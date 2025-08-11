# 3s 0.5s overlapping trials.

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
import matplotlib.pyplot as plt

# ---------------------------
# Set Seed for Reproducibility
# ---------------------------
seed_everything(12345)

# ---------------------------
# Utility Functions for Data Preparation
# ---------------------------
def plvfcn(eegData):
    # Use only the first 19 electrodes
    eegData = eegData[:, :19]
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    # Set diagonal to zero (mimicking A = |PLV| - I)
    np.fill_diagonal(plvMatrix, 0)
    return plvMatrix

def compute_plv(subject_data, fs=256):
    """
    Chop each trial into 1 s windows with 0.5 s overlap,
    compute 19×19 PLV per window, label with original trial label.
    """
    num_elec = 19
    idx      = ['L','R']
    all_plvs = []
    all_labels = []

    for label, field in enumerate(idx):      # 0=L, 1=R
        trials = subject_data[field][0]      # list of trials
        for trial in trials:
            data      = trial[:, :num_elec]
            n_samples = data.shape[0]
            # number of sliding windows
            n_windows = 1 + (n_samples - win_len) // step

            for w in range(n_windows):
                start = w * step
                seg   = data[start:start+win_len, :]
                plv_mat = plvfcn(seg)
                all_plvs.append(plv_mat)
                all_labels.append(label)

    plv_array = np.stack(all_plvs, axis=2)     # (19,19,N_windows_total)
    y         = torch.tensor(all_labels, dtype=torch.long)
    return plv_array, y


def create_graphs(plv, threshold):
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs      = 256           # samples/sec
win_len = fs*3            # 1 sec window = 256 samples
step    = fs // 2       # 0.5 sec overlap = 128 samples

# Process each subject individually and store PyG Data objects per subject.
all_plvs = {}
for subject_number in subject_numbers:
    print(f'Processing Subject S{subject_number}')
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f'Subject{subject_number}']
    S1 = subject_raw[:, :-1]  # Use all columns except the last one
    plv, y = compute_plv(S1, fs=256)
    threshold = 0.1
    graphs = create_graphs(plv, threshold)
    numElectrodes = 19
    # Build an adjacency tensor from the graphs.
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i in range(len(graphs)):
        adj[:, :, i] = nx.to_numpy_array(graphs[i])
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # Build edge indices for each graph.
    edge_indices = []
    for i in range(adj.shape[2]):
        source_nodes = []
        target_nodes = []
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                if adj[row, col, i] >= threshold:
                    source_nodes.append(row)
                    target_nodes.append(col)
                else:
                    # Placeholder to maintain shape (can be omitted if handled differently)
                    source_nodes.append(0)
                    target_nodes.append(0)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
    edge_indices = torch.stack(edge_indices, dim=-1)
    
    # Create PyG Data objects for each trial.
    data_list = []
    for i in range(np.size(adj, 2)):
        data_list.append(Data(x=adj[:, :, i], edge_index=edge_indices[:, :, i], y=y[i]))
    
    # Rearranging data (assuming two conditions interleaved)
    size = len(data_list)
    idx = size // 2
    combined = []
    for i in range(idx):
        combined.extend([data_list[i], data_list[i+idx]])
    all_plvs[f'S{subject_number}'] = combined

# Combine all subjects’ Data objects into one list and add subject attribute.
all_data = []
for subject, data_list in all_plvs.items():
    for data in data_list:
        data.subject = int(subject.strip('S'))
        all_data.append(data)


# LOSO Split Function
def split_data_by_subject(data_list, test_subject):
    train_data = [d for d in data_list if d.subject != test_subject]
    test_data  = [d for d in data_list if d.subject == test_subject]
    return train_data, test_data

# ---------------------------
# Define the Simple GAT Model (Feature Extractor and Classifier)
# ---------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(SimpleGAT, self).__init__()
        # First GAT layer: 19 -> 32 per head, concatenated output -> 32*num_heads
        self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True)
        self.gn1 = GraphNorm(32 * num_heads)
        
        # Second GAT layer: reduce from 32*num_heads to 16 per head, concatenated -> 16*num_heads
        self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True)
        self.gn2 = GraphNorm(16 * num_heads)
        
        # Third GAT layer: reduce from 16*num_heads to 8, without concatenation (averaging heads)
        self.conv3 = GATv2Conv(16 * num_heads, 8, heads=num_heads, concat=False)
        self.gn3 = GraphNorm(8)
        
        # Final linear layer: maps from 8 to the desired number of output classes
        self.lin = nn.Linear(8, out_channels)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.gn1(self.conv1(x, edge_index)))
        x = F.relu(self.gn2(self.conv2(x, edge_index)))
        x = F.relu(self.gn3(self.conv3(x, edge_index)))
        
        # Global mean pooling to produce graph-level representation.
        x = global_mean_pool(x, batch)
        features = x  # Extracted features.
        logits = self.lin(x)
        return logits, features

# class SimpleGAT(nn.Module): # With Dropout
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
#         super(SimpleGAT, self).__init__()
#         self.dropout = dropout

#         self.conv1 = GATv2Conv(in_channels, 32, heads=num_heads, concat=True, dropout=dropout)
#         self.gn1 = GraphNorm(32 * num_heads)

#         self.conv2 = GATv2Conv(32 * num_heads, 16, heads=num_heads, concat=True, dropout=dropout)
#         self.gn2 = GraphNorm(16 * num_heads)

#         self.conv3 = GATv2Conv(16 * num_heads, 8, heads=num_heads, concat=False, dropout=dropout)
#         self.gn3 = GraphNorm(8)

#         self.lin = nn.Linear(8, out_channels)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = F.relu(self.gn1(self.conv1(x, edge_index)))
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = F.relu(self.gn2(self.conv2(x, edge_index)))
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = F.relu(self.gn3(self.conv3(x, edge_index)))
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = global_mean_pool(x, batch)
#         features = x
#         logits = self.lin(x)
#         return logits, features


# ---------------------------
# LOSO Pipeline: Train GAT and Evaluate at Each Epoch
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #If we use the clusters
num_epochs = 25
loso_results = {}

for test_subject in subject_numbers:
    print(f"\n=== LOSO Fold: Test Subject {test_subject} ===")
    train_data, test_data = split_data_by_subject(all_data, test_subject)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)
    
    model = SimpleGAT(in_channels=19, hidden_channels=32, out_channels=2, num_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0
    best_epoch    = 0
    
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
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on test set at this epoch.
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y).sum().item()
                total   += batch.num_graphs
        test_acc = correct / total if total > 0 else 0
        
        print(
            f"Subject {test_subject}, "
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"Test Acc: {test_acc*100:.2f}%"
        )
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch    = epoch + 1
    
    print(
        f"Subject {test_subject}, Last Epoch ({num_epochs}) "
        f"Test Acc: {test_acc*100:.2f}%"
    )
    
    print(
        f"Subject {test_subject} Best Test Accuracy: "
        f"{best_test_acc*100:.2f}% at Epoch {best_epoch}"
    )
    loso_results[test_subject] = (best_test_acc, best_epoch, test_acc)
    
# ---------------------------
# Summary of LOSO Results (using final‐epoch accuracy)
# ---------------------------
print("\nLOSO Summary (GAT Classification), Final‐Epoch Acc:")
for subj, (best_acc, best_epoch, final_acc) in sorted(loso_results.items()):
    print(f"Subject {subj}: Final Test Accuracy = {final_acc*100:.2f}%")

# Compute average and CI on the final‐epoch accuracies
final_accs = [final_acc for _, (_, _, final_acc) in loso_results.items()]
avg_final_acc = np.mean(final_accs)
print(f"\nAverage Final Test Accuracy across LOSO folds: {avg_final_acc*100:.2f}%")

#%%

import scipy.stats as st

n = len(final_accs)
mean_acc = avg_final_acc
std_acc  = np.std(final_accs, ddof=1)

# 95% CI via t‐distribution
confidence_level = 0.95
alpha = 1 - confidence_level
t_crit = st.t.ppf(1 - alpha/2, df=n-1)
margin_err = t_crit * (std_acc / np.sqrt(n))

ci_lower = mean_acc - margin_err
ci_upper = mean_acc + margin_err

print(f"95% Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

#%%