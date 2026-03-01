from pronto import Ontology
import json
import torch
import torch.nn as nn
import ot  # pot package for optimal transport
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")

from constants import model_name
text_model = SentenceTransformer(model_name)
####################
# 1. Reading owl ontology and KG creation
####################

time_start = time.time()
ontology = Ontology("./hp.owl")
terms = list(ontology.terms())

term_to_idx = {term.id: idx for idx, term in enumerate(terms)}

term_labels = {term.id: term.name for term in terms if term.name}
term_definitions = {term.id: term.definition for term in terms}
term_synonyms = {
    term.id: [syn.description for syn in term.synonyms]
    for term in terms
}


list_obsolete = []
list_obsolete_hpo = []

for key in term_labels.copy().keys():
    if 'HP' not in key:
        del term_labels[key]

for key in term_definitions.copy().keys():
    if 'HP' not in key:
        del term_definitions[key]

for key in term_synonyms.copy().keys():
    if 'HP' not in key:
        del term_synonyms[key]

for key, elem in term_labels.items():
    if 'obsolete' in elem:
        list_obsolete.append(key)


for key in list_obsolete_hpo:
    term_labels.pop(key)
    term_definitions.pop(key)
    term_synonyms.pop(key)

for key in term_labels.copy().keys():
    if key in list_obsolete:
        del term_labels[key]

for key in term_definitions.copy().keys():
    if key in list_obsolete:
        del term_definitions[key]

for key in term_synonyms.copy().keys():
    if key in list_obsolete:
        del term_synonyms[key]


with open("./list_obsolete_hpo.json", "w") as json_file:
    json.dump(list_obsolete, json_file)


filtered_terms = [t for t in terms if t.id in term_labels]
filtered_terms_def = [t for t in terms if t.id in term_definitions]
filtered_terms_syn = [t for t in terms if t.id in term_synonyms]

print(len([item for item in term_synonyms.values() if item == []]))
print(len([item for item in term_definitions.values() if item == None]))
print(len(filtered_terms), len(filtered_terms_def), len(filtered_terms_syn))


dict_definitions_valid = {key:val for key, val in term_definitions.items() if val != None}
term_definitions_valid = [str(el) for el in dict_definitions_valid.values()]
term_syn_valid = {key:val for key, val in term_synonyms.items() if val != []}
term_missing_syn = {key:val for key, val in term_synonyms.items() if val == []}
term_missing_def = {key:val for key, val in term_definitions.items() if val == None}


print('missinf def', len(term_missing_def))
print('missing syns', len(term_missing_syn))

term_to_idx = {term.id: idx for idx, term in enumerate(filtered_terms)}

edges = []
for term in filtered_terms:
    idx = term_to_idx[term.id]
    # parents at distance=1 (is_a)
    for parent in term.superclasses(distance=1):
        if parent.id in term_to_idx:
            src = idx
            dst = term_to_idx[parent.id]
            edges.append((src, dst))

terms_list = filtered_terms

labels_list = [term_labels[t.id] for t in terms_list]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

###########################
# 2a. Text embedding for terms with SentenceTransformers
###########################

text_embeddings = text_model.encode(labels_list, convert_to_numpy=True)
text_embeddings = normalize(text_embeddings, axis=1)

text_embeddings_def = text_model.encode(term_definitions_valid, convert_to_numpy=True)
text_embeddings_def = normalize(text_embeddings_def, axis=1)

all_synonyms = []
all_synonyms_hpo = []
for hpo_id, syns in term_syn_valid.items():
    for s in syns:
        all_synonyms.append(s)
        all_synonyms_hpo.append(hpo_id)

synonyms_embeddings = text_model.encode(all_synonyms, convert_to_numpy=True)
synonyms_embeddings = normalize(synonyms_embeddings, axis=1)

torch.save(text_embeddings, './hpo_embeddings_text.pt')
torch.save(text_embeddings_def, './hpo_embeddings_text_def.pt')
torch.save(synonyms_embeddings, './hpo_embeddings_text_syn.pt')

###########################
# 2b. KG embedding with GraphSAGE
###########################

num_nodes = len(terms_list)
dim = text_embeddings.shape[1]

x = torch.tensor(text_embeddings, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

class GraphSAGEModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModule, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


graph_model = GraphSAGEModule(dim, 256, dim)
with torch.no_grad():
    graph_embeddings = graph_model(data.x, data.edge_index).numpy()
graph_embeddings = normalize(graph_embeddings, axis=1)
torch.save(graph_embeddings, './hpo_embeddings_graph.pt')


###################
# 3. Optimal Transport: alignment of two embeddings
###########################

N = text_embeddings.shape[0]
C = cdist(text_embeddings, graph_embeddings, metric='sqeuclidean')

a = np.ones(N) / N
b = np.ones(N) / N
G = ot.sinkhorn(a, b, C, reg=0.1)
latent_embeddings = np.dot(G, graph_embeddings)

torch.save(latent_embeddings, './hpo_embeddings_latent_space.pt')

time_end = time.time()

total_time = time_end - time_start

print(total_time, 's')
