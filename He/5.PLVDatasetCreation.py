#!/usr/bin/env python3

import os
import pickle
import numpy as np
import scipy.signal as sig
from tqdm import tqdm
import traceback

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# ---------------------------
# CONFIG
# ---------------------------
server_dir = r'/scratch/uceerjp/'  # your path
cache_path = os.path.join(server_dir, 'plv_graph_dataset.pkl')

window_length_sec = 'full'  # Set to 'full' or a float (e.g., 3.0)
window_overlap = 0.5        # Only used if window_length_sec is float
fs = 1000                   # Sampling frequency (Hz)

# ---------------------------
# PLV Graph Builder
# ---------------------------
def segment_trial(eeg, fs, win_sec, overlap):
    win_len = int(win_sec * fs)
    step = int(win_len * (1 - overlap))
    segments = []
    if eeg.shape[1] < win_len:
        return []
    for start in range(0, eeg.shape[1] - win_len + 1, step):
        end = start + win_len
        segments.append(eeg[:, start:end])
    return segments

def process_pkl_file(filepath):
    filename = os.path.basename(filepath)
    subject_id = int(filename.split('_')[0][1:])
    print(f"\n[INFO] Processing {filename}")

    try:
        with open(filepath, 'rb') as f:
            bci = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        traceback.print_exc()
        return [], [], [], []

    graphs = []
    eeg_data = bci['data']
    meta = bci['TrialData']
    posy = bci['positionx']
    posx = bci['positiony']

    for i in tqdm(range(len(eeg_data)), desc=f"Trials in {filename}", leave=False):
        label = meta[i].get('targetnumber')
        if label not in [1, 2]:
            continue

        eeg = np.array(eeg_data[i])
        if window_length_sec == 'full':
            segments = [eeg]
        else:
            segments = segment_trial(eeg, fs, window_length_sec, window_overlap)

        for seg in segments:
            n_channels, n_times = seg.shape
            plv = np.zeros((n_channels, n_channels))

            for ch1 in range(n_channels):
                for ch2 in range(ch1 + 1, n_channels):
                    phase1 = np.angle(sig.hilbert(seg[ch1]))
                    phase2 = np.angle(sig.hilbert(seg[ch2]))
                    diff = phase2 - phase1
                    plv_val = np.abs(np.sum(np.exp(1j * diff)) / n_times)
                    plv[ch1, ch2] = plv_val
                    plv[ch2, ch1] = plv_val

            edge_index, edge_attr = dense_to_sparse(torch.tensor(plv, dtype=torch.float32))

            graph = Data(
                x=torch.tensor(plv, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label - 1),
                subject=subject_id
            )
            graphs.append(graph)

    return graphs, meta, posx, posy

# ---------------------------
# Load and Generate Graphs
# ---------------------------
if os.path.exists(cache_path):
    print(f"\n[CACHE] Found existing PLV graph cache. Loading from: {cache_path}")
    with open(cache_path, 'rb') as f:
        all_data, subject_numbers, all_meta, all_posx, all_posy = pickle.load(f)
else:
    print(f"\n[INFO] Generating PLV graphs (serial)...")
    all_files = sorted([f for f in os.listdir(server_dir) if f.endswith('.pkl') and f.startswith('S')])
    all_paths = [os.path.join(server_dir, f) for f in all_files]

    all_data = []
    all_meta = []
    all_posx = []
    all_posy = []
    subject_numbers = set()

    for path in tqdm(all_paths, desc="PLV graph generation"):
        graphs, meta, posx, posy = process_pkl_file(path)
        all_data.extend(graphs)
        all_meta.extend(meta)
        all_posx.append(posx)
        all_posy.append(posy)
        if graphs:
            subject_numbers.add(graphs[0].subject)

    subject_numbers = sorted(subject_numbers)
    with open(cache_path, 'wb') as f:
        pickle.dump((all_data, subject_numbers, all_meta, all_posx, all_posy), f)
    print(f"\n✅ Saved {len(all_data)} graphs from {len(subject_numbers)} subjects to {cache_path}")

print(f"\n[INFO] Ready to train with {len(all_data)} trials from {len(subject_numbers)} subjects.")
