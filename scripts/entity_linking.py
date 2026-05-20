from pronto import Ontology
import pandas as pd
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import warnings
from collections import Counter
import tiktoken
from openai import OpenAI
import time
import os
from constants import API_KEY, TEMP, GPT_MODEL
warnings.filterwarnings("ignore")
from constants import model_name

text_model = SentenceTransformer(model_name)
client_open_ai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", API_KEY))


text_embeddings = torch.load('hpo_embeddings_text.pt', weights_only=False)
text_embeddings_def = torch.load('hpo_embeddings_text_def.pt', weights_only=False)
synonyms_embeddings = torch.load('hpo_embeddings_text_syn.pt', weights_only=False)
graph_embeddings = torch.load('hpo_embeddings_graph.pt', weights_only=False)
latent_embeddings = torch.load('hpo_embeddings_latent_space.pt', weights_only=False)

f = open("term_labels.json")
term_labels = json.load(f)

ontology = Ontology("hp.owl")
terms = list(ontology.terms())
terms_list = [t for t in terms if t.id in term_labels]


f = open("term_definitions_valid.json")
term_definitions_valid = json.load(f)

f = open("dict_definitions_valid.json")
dict_definitions_valid = json.load(f)

f = open("all_synonyms_hpo.json")
all_synonyms_hpo = json.load(f)

f = open("all_synonyms.json")
all_synonyms = json.load(f)

dict_labels_valid = {term.id: term.name for term in terms}


def get_raw_response(query, content, client_open_ai):

    prompt = f"""

    Role and Instructions:
    Act as a genomics assistant: your aim is to support geneticists in the task of phenotypes extraction from text. 
    Your task is to identify, given a list of terms as candidates, the most fitting one for the query given. 
    Query is a description of a symptom or condition extracted from a text. List of candidates is a list of up to four terms. 
    Of course, query and candidates can be different. Your aim is to identify the most similar, for a semantic reason.
    If you don’t detect any fitting term at all or if you don’t know, don’t try to make up an answer, just write ’None’.
    Query:
    {query}
    List of possible phenotypes between you have to choose:
    {content}.
    Safety Measures:
    Avoid speculation: extract phenotypes known or explicitly available from the list.
    """

    submit_function = [
    {
        "name": "identify_patient_phenotypes",
        "description": "Most fitting term from the provided list.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_1_candidate": {
                    "type": "string",
                    "description": "Most fitting term from the provided list.",
                }
            },
            "required": ["top_1_candidate"]
            }
        }
    ]

    model = GPT_MODEL
    messages = [{"role": "user", "content": prompt}]

    if GPT_MODEL == 'gpt-4o':
        gpt_funct = client_open_ai.chat.completions.create(
            model=model,
            messages=messages,
            functions=submit_function,
            function_call="auto",
            temperature=TEMP,
            timeout=60,
            stream=False
        )
    else: #reasoning model
        gpt_funct = client_open_ai.chat.completions.create(
            model=model,
            messages=messages,
            functions=submit_function,
            function_call="auto",
            timeout=60,
            stream=False
        )
    return gpt_funct


def generate_results(query, content, client_open_ai):

    if content is None or len(content) == 0:
        return "N/A"

    try:
        response = get_raw_response(query, content, client_open_ai)

        func_args = json.loads(response.choices[0].message.function_call.arguments)
        return func_args

    except Exception as e:
        print(e)
        return "N/A"


def find_top_candidate_rag(query, list_candidates, list_labels_candidates, client_open_ai):

    answer = generate_results(query, list_labels_candidates, client_open_ai) #, encoding)

    if answer and type(answer) == dict:
        return answer['top_1_candidate']
    else:
        return 'None'


def find_top_candidate(list_candidates, list_labels_candidates):

    counting = Counter(list_labels_candidates)
    list_counts = [v for v in counting.values()]
    list_equal_terms = {}
    average_cos = {}

    for elem in list(set(list_labels_candidates)):
        list_equal_terms[elem] = [el[elem] for el in list_candidates for k, v in el.items() if k == elem]

    for k, v in list_equal_terms.items():
        average_cos[k] = np.average(v)

    for elem in list_candidates:
        for k, v in elem.items():
            if v == 1.0:
                return k, False

    if 3 in list_counts or 4 in list_counts:
        return max(counting, key=counting.get), False
    elif 2 in list_counts and 1 not in list_counts:
        return max(average_cos, key=average_cos.get), True
    elif 2 in list_counts and 1 in list_counts:
        return max(counting, key=counting.get), True
    elif len(list(set(list_counts))) == 1 and 1 in list(set(list_counts)):
        max_dict = max(list_candidates[0:3], key=lambda d: list(d.values())[0])
        max_key = list(max_dict.keys())[0]

        return max_key, True
    else:
        return None, False


def entity_linking_from_term(query, client_open_ai, debug=False):

    list_candidates = []
    list_labels_candidates = []
    dict_label_id = {}
    category_candidates = {}

    query_emb = text_model.encode([query], convert_to_numpy=True)
    query_emb = normalize(query_emb, axis=1)

    # --------------------------------------------------
    # 1. Similarity in label text-embedding space
    # --------------------------------------------------
    cos_sims = np.dot(query_emb, text_embeddings.T)[0]

    best_idx = np.argmax(cos_sims)
    best_term = terms_list[best_idx]

    label_candidate = term_labels[best_term.id]
    label_candidate_hpo = best_term.id
    label_candidate_score = round(cos_sims[best_idx], 3)

    list_candidates.append({label_candidate: label_candidate_score})
    category_candidates["Label"] = label_candidate
    list_labels_candidates.append(label_candidate)
    dict_label_id[label_candidate] = label_candidate_hpo

    # --------------------------------------------------
    # 2. Similarity in definition text-embedding space
    # --------------------------------------------------
    cos_sims_def = np.dot(query_emb, text_embeddings_def.T)[0]

    best_idx_def = np.argmax(cos_sims_def)
    best_term_def = term_definitions_valid[best_idx_def]

    hpo_max_def = [
        key
        for key, el in dict_definitions_valid.items()
        if str(el) == best_term_def
    ][0]

    definition_candidate = term_labels[hpo_max_def]
    definition_candidate_hpo = hpo_max_def
    definition_candidate_score = round(cos_sims_def[best_idx_def], 3)

    list_candidates.append({definition_candidate: definition_candidate_score})
    category_candidates["Definition"] = definition_candidate
    list_labels_candidates.append(definition_candidate)
    dict_label_id[definition_candidate] = definition_candidate_hpo

    # --------------------------------------------------
    # 3. Similarity in synonym text-embedding space
    # --------------------------------------------------
    cos_sims_syn = np.dot(query_emb, synonyms_embeddings.T)[0]

    top_k_syn = 10
    top_k_idx_syn = np.argsort(cos_sims_syn)[-top_k_syn:][::-1]

    best_syn_idx = top_k_idx_syn[0]
    synonym_candidate_hpo = all_synonyms_hpo[best_syn_idx]
    synonym_candidate = term_labels[synonym_candidate_hpo]
    synonym_candidate_score = round(cos_sims_syn[best_syn_idx], 3)

    list_candidates.append({synonym_candidate: synonym_candidate_score})
    category_candidates["Synonym"] = synonym_candidate
    list_labels_candidates.append(synonym_candidate)
    dict_label_id[synonym_candidate] = synonym_candidate_hpo
    print('label_candidate', label_candidate, 'synonym_candidate', synonym_candidate, 'definition_candidate:', definition_candidate)

    # --------------------------------------------------
    # 4. Latent-space reranking of top-K label candidates
    # --------------------------------------------------
    K = 10
    top_k_idx = np.argsort(cos_sims)[-K:][::-1]

    top_k_text_sims = cos_sims[top_k_idx]

    # Avoid negative or unstable weights
    top_k_weights = np.maximum(top_k_text_sims, 0)

    if top_k_weights.sum() == 0:
        top_k_weights = np.ones_like(top_k_weights) / len(top_k_weights)
    else:
        top_k_weights = top_k_weights / top_k_weights.sum()

    latent_embeddings_norm = normalize(latent_embeddings, axis=1)

    query_latent = np.sum(
        latent_embeddings_norm[top_k_idx] * top_k_weights[:, None],
        axis=0
    )

    query_latent = normalize(query_latent.reshape(1, -1), axis=1)

    latent_sims = np.dot(query_latent, latent_embeddings_norm.T)[0]

    # Important change:
    # choose the best latent-space term ONLY among the original top-K text candidates
    latent_best_idx = top_k_idx[np.argmax(latent_sims[top_k_idx])]
    latent_best_term = terms_list[latent_best_idx]

    latent_candidate = term_labels[latent_best_term.id]
    latent_candidate_hpo = latent_best_term.id
    latent_candidate_score = round(latent_sims[latent_best_idx], 5)

    list_candidates.append({latent_candidate: latent_candidate_score})

    print(list_candidates)
    category_candidates["Aligned"] = latent_candidate
    list_labels_candidates.append(latent_candidate)
    dict_label_id[latent_candidate] = latent_candidate_hpo

    if debug:
        print("\nTop K label-space terms used for latent projection:")
        for rank, idx in enumerate(top_k_idx, start=1):
            term = terms_list[idx]
            hpo_id = term.id
            label = term_labels[hpo_id]
            print(
                f"{rank}. {label} ({hpo_id}) | "
                f"text_cosine={cos_sims[idx]:.4f} | "
                f"weight={top_k_weights[rank - 1]:.4f} | "
                f"latent_cosine={latent_sims[idx]:.5f}"
            )

        print("\nSelected latent reranked candidate:")
        print(
            f"{latent_candidate} ({latent_candidate_hpo}) | "
            f"latent_cosine={latent_candidate_score}"
        )

        print("\nlist_candidates:")
        print(list_candidates)

    # --------------------------------------------------
    # 5. Voting / candidate selection
    # --------------------------------------------------
    top_1, rag_necessary = find_top_candidate(
        list_candidates,
        list_labels_candidates
    )

    top_1_hpo = dict_label_id.get(top_1)

    top_1_rag = top_1
    top_1_hpo_rag = top_1_hpo

    if rag_necessary:
        top_1_rag = find_top_candidate_rag(
            query,
            list_candidates,
            list_labels_candidates,
            client_open_ai
        )

    if top_1_rag and top_1_rag not in ["None", "N/A"]:
        top_1_hpo_rag = dict_label_id.get(top_1_rag)

        # Fallback only if RAG returns a valid label not present among candidates
        if top_1_hpo_rag is None:
            matches = [
                key
                for key, value in dict_labels_valid.items()
                if value == top_1_rag
            ]
            top_1_hpo_rag = matches[0] if matches else None

    elif top_1_rag == "None":
        top_1_hpo_rag = "None"

    matching_keys_top_1 = [
        key
        for key, value in category_candidates.items()
        if value == top_1
    ]

    matching_keys_top_1_rag = [
        key
        for key, value in category_candidates.items()
        if value == top_1_rag
    ]

    return (
        top_1,
        top_1_hpo,
        top_1_rag,
        top_1_hpo_rag,
        matching_keys_top_1,
        matching_keys_top_1_rag,
    )



