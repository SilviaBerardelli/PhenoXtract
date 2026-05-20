from pronto import Ontology
import json
import torch
import torch.nn as nn
import ot  # pot package for optimal transport
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, to_undirected
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

# term_to_idx = {term.id: idx for idx, term in enumerate(terms)}

term_labels = {term.id: term.name for term in terms if term.name}
term_definitions = {term.id: term.definition for term in terms}
term_synonyms = {
    term.id: [syn.description for syn in term.synonyms]
    for term in terms
}


###QUI NUOVA PARTE
# Keep only real HPO terms and remove obsolete HPO terms
valid_hpo_ids = {
    term.id
    for term in terms
    if term.id.startswith("HP:")
    and term.name is not None
    and "obsolete" not in term.name.lower()
}

obsolete_hpo_ids = [
    term.id
    for term in terms
    if term.id.startswith("HP:")
    and term.name is not None
    and "obsolete" in term.name.lower()
]

term_labels = {
    hpo_id: label
    for hpo_id, label in term_labels.items()
    if hpo_id in valid_hpo_ids
}

term_definitions = {
    hpo_id: definition
    for hpo_id, definition in term_definitions.items()
    if hpo_id in valid_hpo_ids
}

term_synonyms = {
    hpo_id: synonyms
    for hpo_id, synonyms in term_synonyms.items()
    if hpo_id in valid_hpo_ids
}

with open("./list_obsolete_hpo.json", "w") as json_file:
    json.dump(obsolete_hpo_ids, json_file, ensure_ascii=False, indent=4)

assert set(term_labels.keys()) == set(term_definitions.keys())
assert set(term_labels.keys()) == set(term_synonyms.keys())

assert all(hpo_id.startswith("HP:") for hpo_id in term_labels.keys())
assert all("obsolete" not in label.lower() for label in term_labels.values())

print("Valid non-obsolete HPO terms:", len(term_labels))
print("Obsolete HPO terms removed:", len(obsolete_hpo_ids))

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
# 2b. KG embedding with TRAINED GraphSAGE
###########################

num_nodes = len(terms_list)
dim = text_embeddings.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Node features: text embeddings of HPO labels
x = torch.tensor(text_embeddings, dtype=torch.float, device=device)

# Convert ontology graph to undirected for neighborhood aggregation
# This lets information flow both child -> parent and parent -> child
edge_index = to_undirected(edge_index, num_nodes=num_nodes).to(device)

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


graph_model = GraphSAGEModule(
    in_channels=dim,
    hidden_channels=256,
    out_channels=dim
).to(device)

optimizer = torch.optim.Adam(
    graph_model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

num_epochs = 200

for epoch in range(1, num_epochs + 1):
    graph_model.train()
    optimizer.zero_grad()

    # Compute node embeddings
    z = graph_model(data.x, data.edge_index)
    z = F.normalize(z, p=2, dim=1)

    # Positive edges: actual ontology edges
    pos_edge_index = data.edge_index
    src_pos = pos_edge_index[0]
    dst_pos = pos_edge_index[1]

    pos_scores = (z[src_pos] * z[dst_pos]).sum(dim=1)

    # Negative edges: randomly sampled non-edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method="sparse"
    )

    src_neg = neg_edge_index[0]
    dst_neg = neg_edge_index[1]

    neg_scores = (z[src_neg] * z[dst_neg]).sum(dim=1)

    # Binary edge reconstruction loss
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores,
        torch.ones_like(pos_scores)
    )

    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores,
        torch.zeros_like(neg_scores)
    )

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{num_epochs} | Loss: {loss.item():.4f}")


# Final trained graph embeddings
graph_model.eval()

with torch.no_grad():
    graph_embeddings = graph_model(data.x, data.edge_index)
    graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
    graph_embeddings = graph_embeddings.cpu().numpy()

torch.save(
    graph_embeddings,
    './hpo_embeddings_graph.pt'
)

print("Graph embeddings saved. Cleaning training objects before OT...")


file_path = "./term_labels.json"
with open(file_path, 'w') as file:
    json.dump(term_labels, file, ensure_ascii=False, indent=4)

file_path = "./term_definitions_valid.json"
with open(file_path, 'w') as file:
    json.dump(term_definitions_valid, file, ensure_ascii=False, indent=4)

file_path = "./dict_definitions_valid.json"
with open(file_path, 'w') as file:
    json.dump(dict_definitions_valid, file, ensure_ascii=False, indent=4)

file_path = "./all_synonyms_hpo.json"
with open(file_path, 'w') as file:
    json.dump(all_synonyms_hpo, file, ensure_ascii=False, indent=4)

file_path = "./all_synonyms.json"
with open(file_path, 'w') as file:
    json.dump(all_synonyms, file, ensure_ascii=False, indent=4)

def sinkhorn_transport_apply_blockwise(
    C,
    graph_embeddings,
    reg=0.1,
    num_iter=1000,
    stop_thr=1e-9,
    block_size=1000,
    eps=1e-16,
):
    """
    Memory-efficient Sinkhorn that does not materialize
    K = exp(-C/reg) nor G = diag(u) K diag(v).

    Returns:
        latent_embeddings = G @ graph_embeddings
    """

    N = C.shape[0]

    a = np.full(N, 1.0 / N, dtype=np.float64)
    b = np.full(N, 1.0 / N, dtype=np.float64)

    u = np.ones(N, dtype=np.float64)
    v = np.ones(N, dtype=np.float64)

    # -----------------------------------------
    # Sinkhorn iterations:
    # u = a / (K @ v)
    # v = b / (K.T @ u)
    # -----------------------------------------
    for it in range(num_iter):
        u_prev = u.copy()

        # Compute K @ v blockwise
        Kv = np.empty(N, dtype=np.float64)

        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)

            C_block = C[i:i_end]  # shape: block_size x N
            K_block = np.exp(-C_block / reg, dtype=np.float64)

            Kv[i:i_end] = K_block @ v

        u = a / np.maximum(Kv, eps)

        # Compute K.T @ u blockwise
        KTu = np.zeros(N, dtype=np.float64)

        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)

            C_block = C[i:i_end]  # shape: block_size x N
            K_block = np.exp(-C_block / reg, dtype=np.float64)

            KTu += K_block.T @ u[i:i_end]

        v = b / np.maximum(KTu, eps)

        # Convergence check on u
        err = np.linalg.norm(u - u_prev, ord=1)

        if it % 50 == 0:
            print(f"Sinkhorn iter {it:04d} | err={err:.6e}")

        if err < stop_thr:
            print(f"Sinkhorn converged at iteration {it} | err={err:.6e}")
            break

    # -----------------------------------------
    # Compute latent_embeddings = G @ graph_embeddings
    # where G = diag(u) K diag(v)
    # -----------------------------------------
    weighted_graph = v[:, None] * graph_embeddings

    latent_embeddings = np.empty(
        (N, graph_embeddings.shape[1]),
        dtype=np.float64
    )

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)

        C_block = C[i:i_end]
        K_block = np.exp(-C_block / reg, dtype=np.float64)

        latent_embeddings[i:i_end] = (
            u[i:i_end, None] *
            (K_block @ weighted_graph)
        )

    return latent_embeddings
###################
# 3. Optimal Transport: alignment of two embeddings
###########################
text_embeddings = torch.load("./hpo_embeddings_text.pt", weights_only=False)
graph_embeddings = torch.load("./hpo_embeddings_graph.pt", weights_only=False)

n_text = text_embeddings.shape[0]
n_graph = graph_embeddings.shape[0]
N = text_embeddings.shape[0]
C = np.empty((n_text, n_graph), dtype=np.float32)


chunk_size = 1000 #1000  # da regolare

for i in range(0, n_text, chunk_size):
    i_end = min(i + chunk_size, n_text)
    C[i:i_end] = cdist(
        text_embeddings[i:i_end],
        graph_embeddings,
        metric='sqeuclidean'
    ).astype(np.float32, copy=False)


latent_embeddings = sinkhorn_transport_apply_blockwise(
    C=C,
    graph_embeddings=graph_embeddings,
    reg=0.1,
    num_iter=1000,
    stop_thr=1e-9,
    block_size=1000,
)

torch.save(
    latent_embeddings,
    './hpo_embeddings_latent_space.pt'
)

time_end = time.time()

total_time = time_end - time_start

print(total_time, 's')

