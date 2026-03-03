import json
import pandas as pd
import tiktoken
from openai import OpenAI
import time
import os
from constants import API_KEY, TEMP, GPT_MODEL


client_open_ai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", API_KEY))


def get_raw_response(content, client_open_ai):

    prompt = f"""

    Role and Instructions:
    Act as a genomics assistant: your aim is to support geneticists in the task of phenotypes extraction from text. 
    Your task is to produce a list of entities extracted from a clinical description, in the raw text provided, both for phenotypes and for negative phenotypes. 
    Given a raw text, you must identify the spans related to possible phenotypes, either explicitly or implicitly.
    You should keep in the span all words related to the phenotype that should be informative (such as negation or adjective). 
    Into negative phenotypes, include also descriptions like "recovered without any neurological disability", where is specified that the patient does not have that symptom.
    Avoid to keep descriptions that are not mainly related to phenotypes, as "male child" or "female newborn". 
    Avoid to keep descriptions related to mutations and mode of inheritance, as "mutational burden" or inheritance qualifiers.
    Avoid to keep descriptions as "asymptomatic".
    Avoid to keep patient that have no phenotypes specified.
    You may reformulate the span if needed. 
    If you recognize more than one patient in the text, please list all the phenotypes owned by each patient.
    If you don’t detect any span or if you don’t know, don’t try to make up an answer, just write ’None’.
    If you detect phenotypes without patient specifications, use "not specified" for patient but include phenotypes.
    Raw text:
    {content}.
    Safety Measures:
    Avoid redundancy.
    Avoid speculation: extract phenotypes known or explicitly available from the text.
    Comprehensive use of text: Utilize all relevant information across the entire text to ensure diversity.
    """

    submit_function = [
    {
        "name": "extract_patient_phenotypes",
        "description": "Extract phenotypes and negative phenotypes for each patient from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "patients": {
                    "type": "array",
                    "description": "List of patients with their phenotypes and negative phenotypes.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the patient."
                            },
                            "phenotypes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of phenotypes for the patient."
                            },
                            "negative_phenotypes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of negative phenotypes for the patient."
                            }
                        },
                        "required": ["name", "phenotypes", "negative_phenotypes"]
                    }
                }
            },
            "required": ["patients"]
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


def generate_question_answer(content, client_open_ai):

    if content is None or len(content) == 0:
        return "N/A"

    try:
        response = get_raw_response(content, client_open_ai)
        func_args = json.loads(response.choices[0].message.function_call.arguments)
        return func_args

    except Exception as e:
        print(e)
        return "N/A"


def concept_recognition_from_text(content, client_open_ai, json_path, output_file_path):

    start_generating = time.time()

    answer = generate_question_answer(content, client_open_ai)
    end_generating = time.time()

    time_answer = end_generating - start_generating

    with open(json_path+'.json', 'w') as fp:
        json.dump(answer, fp)

    if answer and answer != "N/A" and answer != "None":

        if answer == {'patients': []}:
            answer = {'patients': [{'name': 'not specified', 'phenotypes': [], 'negative_phenotypes': []}]}

        df = pd.json_normalize(answer, record_path='patients')
        df_positive = df.explode('phenotypes')
        df_positive['negative_phenotype'] = 0
        df_positive = df_positive.rename(columns={'phenotypes': 'phenotype'})

        df_negative = df.explode('negative_phenotypes')
        df_negative['negative_phenotype'] = 1
        df_negative = df_negative.rename(columns={'negative_phenotypes': 'phenotype'})

        df_negative = df_negative.dropna(subset=['phenotype'])

        df_final = pd.concat([df_positive[['name', 'phenotype', 'negative_phenotype']], df_negative[['name', 'phenotype', 'negative_phenotype']]])
        df_final = df_final.rename(columns={'name': 'patient'})
        df_final = df_final.reset_index(drop=True)

        df_final.to_csv(output_file_path+'.csv', sep=',')

        return df_final
    else:
        return None

