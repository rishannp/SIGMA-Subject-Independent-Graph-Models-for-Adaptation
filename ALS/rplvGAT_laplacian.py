import os
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, GraphNorm
from torch_geometric.seed import seed_everything
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import scipy.stats as st

# ---------------------------
# Set Seed for Reproducibility
# ---------------------------
seed_everything(12345)

# ---------------------------
# ALS Channel Coordinates for Surface Laplacian
# ---------------------------
ALS_coords = np.array([
    [0.950, 0.309], [0.950, -0.309],
    [0.587, 0.809], [0.673, 0.545],
    [0.719, 0.0],   [0.673, -0.545],
    [0.587, -0.809],[0.0, 0.999],
    [0.0, 0.719],   [0.0, 0.0],
    [0.0, -0.719],  [0.0, -0.999],
    [-0.587, 0.809],[-0.673, 0.545],
    [-0.719, 0.0],  [-0.673, -0.545],
    [-0.587, -0.809],[-0.950, 0.309],
    [-0.950, -0.309]
])

# ---------------------------
# Surface Laplacian (Finite-Difference)
# ---------------------------
def compute_surface_laplacian(eeg_epoch, coords=ALS_coords):
    tri = Delaunay(coords)
    neighbors = {i: set() for i in range(len(coords))}
    for simplex in tri.simplices:
        for k in range(3):
            u, v = simplex[k], simplex[(k+1)%3]
            neighbors[u].add(v)
            neighbors[v].add(u)
    lap = np.zeros_like(eeg_epoch)
    for ch in range(eeg_epoch.shape[1]):
        nbrs = list(neighbors[ch])
        if nbrs:
            lap[:, ch] = eeg_epoch[:, ch] - eeg_epoch[:, nbrs].mean(axis=1)
        else:
            lap[:, ch] = eeg_epoch[:, ch]
    return lap

# ---------------------------
# Graph Laplacian Denoising
# ---------------------------
def graph_denoise_adjacency(A, k_eigen=5, tau=0.1):
    D = np.diag(A.sum(axis=1))
    L = D - A
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)[:k_eigen]
    P = vecs[:, idx] @ vecs[:, idx].T
    A_d = P @ A @ P
    return np.clip(A_d, 0, None)

# ---------------------------
# PLV Computation with Preprocessing Modes
# ---------------------------
def plvfcn(eegData, mode='raw'):
    """
    mode: 'raw', 'spatial', 'graph', 'both'
      - raw: compute PLV on raw data
      - spatial: apply surface Laplacian then PLV
      - graph: compute PLV then graph denoise
      - both: spatial + graph
    """
    eegData = eegData[:, :19]
    # spatial filter
    if mode in ('spatial', 'both'):
        eegData = compute_surface_laplacian(eegData)
    # compute PLV
    numE = eegData.shape[1]
    T = eegData.shape[0]
    P = np.zeros((numE, numE))
    for i in range(numE):
        for j in range(i+1, numE):
            ph1 = np.angle(sig.hilbert(eegData[:, i]))
            ph2 = np.angle(sig.hilbert(eegData[:, j]))
            P[i, j] = abs(np.sum(np.exp(1j*(ph2-ph1))) / T)
            P[j, i] = P[i, j]
    # graph denoise
    if mode in ('graph', 'both'):
        P = graph_denoise_adjacency(P)
    np.fill_diagonal(P, 0)
    return P

# ---------------------------
# Compute PLV for Subject Data
# ---------------------------
def compute_plv(subject_data, mode='raw'):
    idx = ['L', 'R']
    numE = 19
    trials = subject_data.shape[1]
    plv = {field: np.zeros((numE, numE, trials)) for field in idx}
    for field in idx:
        for t in range(trials):
            epoch = subject_data[field][0, t][:, :19]
            plv[field][:, :, t] = plvfcn(epoch, mode)
    l, r = plv['L'], plv['R']
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((np.zeros((trials,1)), np.ones((trials,1))), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y

# ---------------------------
# Create Graphs from PLV
# ---------------------------
def create_graphs(plv, threshold=0.1):
    graphs = []
    for t in range(plv.shape[2]):
        A = plv[:, :, t]
        G = nx.Graph()
        G.add_nodes_from(range(A.shape[0]))
        for u in range(A.shape[0]):
            for v in range(A.shape[1]):
                if u != v and A[u, v] > threshold:
                    G.add_edge(u, v, weight=A[u, v])
        graphs.append(G)
    return graphs

# ---------------------------
# Data Loading and Processing
# ---------------------------
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
all_plvs = {}
for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    S1 = subject_raw[:, :-1]
    # compute for desired mode
    mode = 'graph'  # change to 'spatial','graph','both' as needed
    plv, y = compute_plv(S1, mode)
    threshold = 0.1
    graphs = create_graphs(plv, threshold)
    numElectrodes = 19
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
    adj = torch.tensor(adj, dtype=torch.float32)
    edge_indices = []
    for i in range(adj.shape[2]):
        src, tgt = [], []
        for u in range(adj.shape[0]):
            for v in range(adj.shape[1]):
                if adj[u, v, i] >= threshold:
                    src.append(u); tgt.append(v)
                else:
                    src.append(0); tgt.append(0)
        edge_indices.append(torch.tensor([src, tgt], dtype=torch.long))
    edge_indices = torch.stack(edge_indices, dim=-1)
    data_list = []
    for i in range(adj.shape[2]):
        data_list.append(Data(x=adj[:, :, i], edge_index=edge_indices[:, :, i], y=y[i, 0]))
    size = len(data_list)
    half = size // 2
    combined = []
    for i in range(half):
        combined.extend([data_list[i], data_list[i+half]])
    all_plvs[f'S{subject_number}'] = combined
all_data = []
for subject, data_list in all_plvs.items():
    for data in data_list:
        data.subject = int(subject.strip('S'))
        all_data.append(data)

# ---------------------------
# LOSO Split Function
# ---------------------------
def split_data_by_subject(data_list, test_subject):
    train_data = [d for d in data_list if d.subject != test_subject]
    test_data = [d for d in data_list if d.subject == test_subject]
    return train_data, test_data

# ---------------------------
# Define SimpleGAT
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True)
        self.gn1 = GraphNorm(32 * num_heads)
        self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True)
        self.gn2 = GraphNorm(16 * num_heads)
        self.conv3 = GATv2Conv(16 * num_heads, 8, heads=num_heads, concat=False)
        self.gn3 = GraphNorm(8)
        self.lin = nn.Linear(8, out_channels)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        logits = self.lin(x)
        return logits, x

# ---------------------------
# LOSO Pipeline: Train & Evaluate
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 25
loso_results = {}
for test_subject in subject_numbers:
    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    model = SimpleGAT(in_channels=19, hidden_channels=32, out_channels=2, num_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_test_acc, best_epoch = 0, 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.num_graphs
        test_acc = correct / total if total else 0
        print(f"Subject {test_subject}, Epoch {epoch+1}/{num_epochs}, Test Acc: {test_acc*100:.2f}%")
        if test_acc > best_test_acc:
            best_test_acc, best_epoch = test_acc, epoch+1
    print(f"Subject {test_subject} Best Test Accuracy: {best_test_acc*100:.2f}% at Epoch {best_epoch}")
    loso_results[test_subject] = (best_test_acc, best_epoch)
print("\nLOSO Summary:")
avg_acc = np.mean([acc for acc,_ in loso_results.values()])
print(f"Average Test Accuracy: {avg_acc*100:.2f}%")
# Confidence Interval
acc_list = [acc for acc,_ in loso_results.values()]
n, mean_acc, std_acc = len(acc_list), np.mean(acc_list), np.std(acc_list, ddof=1)
t_crit = st.t.ppf(0.975, df=n-1)
moe = t_crit * (std_acc / np.sqrt(n))
print(f"95% CI: [{(mean_acc-moe)*100:.2f}%, {(mean_acc+moe)*100:.2f}%]")
