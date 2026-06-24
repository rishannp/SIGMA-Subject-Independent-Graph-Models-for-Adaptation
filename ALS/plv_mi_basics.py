#%% Connectivity + stability analysis for Left/Right MI cursor control
#    (PLV, Imaginary PLV, PLI, distance vs metric, surrogate null, shared hubs)

import os
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_1samp

import torch
from torch_geometric.seed import seed_everything

# ---------------------------
# Basic config
# ---------------------------
seed_everything(12345)

data_dir = r"C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data"
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# ---------------------------
# Electrode labels + true xyz coords (on unit-ish sphere)
# ---------------------------
electrode_labels = [
    "FP1", "FP2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"
]

xyz_coords = np.array([
    [0.950,  0.309,  -0.0349],   # FP1
    [0.950, -0.309,  -0.0349],   # FP2
    [0.587,  0.809,  -0.0349],   # F7
    [0.673,  0.545,   0.500],    # F3
    [0.719,  0.000,   0.695],    # Fz
    [0.673, -0.545,   0.500],    # F4
    [0.587, -0.809,  -0.0349],   # F8
    [6.120e-17,  0.999,  -0.0349],  # T7
    [4.400e-17,  0.719,   0.695],   # C3
    [0.000,      0.000,  1.000],    # Cz
    [4.400e-17, -0.719,   0.695],   # C4
    [6.120e-17, -0.999,  -0.0349],  # T8
    [-0.587,  0.809,  -0.0349],     # P7
    [-0.673,  0.545,   0.500],      # P3
    [-0.719, -8.810e-17, 0.695],    # Pz
    [-0.673, -0.545,   0.500],      # P4
    [-0.587, -0.809,  -0.0349],     # P8
    [-0.950,  0.309,  -0.0349],     # O1
    [-0.950, -0.309,  -0.0349]      # O2
])

numElectrodes = len(electrode_labels)

# Indices for key electrodes
idx_C3 = electrode_labels.index("C3")
idx_C4 = electrode_labels.index("C4")
idx_P3 = electrode_labels.index("P3")
idx_P4 = electrode_labels.index("P4")
idx_CZ = electrode_labels.index("Cz")
idx_PZ = electrode_labels.index("Pz")

# ---------------------------
# Distance matrix in 3D
# ---------------------------
dist_matrix = np.zeros((numElectrodes, numElectrodes))
for i in range(numElectrodes):
    for j in range(numElectrodes):
        dist_matrix[i, j] = np.linalg.norm(xyz_coords[i] - xyz_coords[j])


# ---------------------------
# Connectivity metrics (PLV, Imag-PLV, PLI)
# ---------------------------
def plv_metrics(eegData):
    """
    Compute PLV, imaginary PLV and PLI between all pairs for a single trial.
    eegData: [time, channels] (only first 19 channels used).
    Returns:
      plv   : [channels, channels]
      imag  : [channels, channels] (imaginary PLV)
      pli   : [channels, channels]
    """
    eegData = eegData[:, :numElectrodes]
    n_t, n_ch = eegData.shape

    analytic = sig.hilbert(eegData, axis=0)
    phases   = np.angle(analytic)

    plv_mat  = np.zeros((n_ch, n_ch))
    imag_mat = np.zeros((n_ch, n_ch))
    pli_mat  = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phases[:, j] - phases[:, i]
            complex_phase = np.exp(1j * dphi)

            mean_complex = np.mean(complex_phase)
            plv      = np.abs(mean_complex)
            imag_plv = np.imag(mean_complex)
            pli      = np.abs(np.mean(np.sign(np.sin(dphi))))

            plv_mat[i, j]  = plv
            plv_mat[j, i]  = plv
            imag_mat[i, j] = imag_plv
            imag_mat[j, i] = imag_plv
            pli_mat[i, j]  = pli
            pli_mat[j, i]  = pli

    return plv_mat, imag_mat, pli_mat


# ---------------------------
# Surrogate generator for PLV null
# ---------------------------
def make_surrogate_trial(eegData):
    """
    Destroy genuine coupling but keep amplitude structure.
    I circularly shift each channel by a random lag.
    """
    eegData = eegData[:, :numElectrodes]
    n_t, n_ch = eegData.shape
    surrogate = np.zeros_like(eegData)
    for ch in range(n_ch):
        shift = np.random.randint(0, n_t)
        surrogate[:, ch] = np.roll(eegData[:, ch], shift)
    return surrogate


# ---------------------------
# Containers for all results
# ---------------------------

# stability_matrices[subject][cls][metric] = {'std': [ch,ch], 'cv': [ch,ch]}
metrics_names = ["PLV", "ImagPLV", "PLI"]
stability_matrices = {subj: {"L": {}, "R": {}} for subj in subject_numbers}

# Mean connectivity matrices across trials (for distance & LI analyses)
mean_plv_per_subject      = {subj: {"L": None, "R": None} for subj in subject_numbers}
mean_imag_plv_per_subject = {subj: {"L": None, "R": None} for subj in subject_numbers}
mean_pli_per_subject      = {subj: {"L": None, "R": None} for subj in subject_numbers}

# For surrogate analysis: average surrogate PLV for key edges
surrogate_plv_edges = {subj: {"L": [], "R": []} for subj in subject_numbers}


# ---------------------------
# Main subject loop
# ---------------------------
for subject_number in subject_numbers:
    print(f"Processing Subject S{subject_number}")
    mat_fname = os.path.join(data_dir, f"S{subject_number}.mat")
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f"Subject{subject_number}"]

    # The last column is some metadata I don't want (as in my original code)
    S1 = subject_raw[:, :-1]

    for cls in ["L", "R"]:
        print(f"  Class {cls}")

        # I mimic the original MATLAB-ish layout:
        # S1 has fields 'L' and 'R', each is 1 x num_trials, cell-like.
        subject_data = S1
        num_trials = subject_data.shape[1]

        plv_all  = np.zeros((numElectrodes, numElectrodes, num_trials))
        imag_all = np.zeros_like(plv_all)
        pli_all  = np.zeros_like(plv_all)

        surrogate_plv_this_class = []

        for t in range(num_trials):
            x = subject_data[cls][0, t][:, :numElectrodes]  # [time, channels]

            plv_mat, imag_mat, pli_mat = plv_metrics(x)
            plv_all[:, :, t]  = plv_mat
            imag_all[:, :, t] = imag_mat
            pli_all[:, :, t]  = pli_mat

            # Surrogate PLV for key edges (C3-P3, C4-P4, Cz-Pz)
            sur_x = make_surrogate_trial(x)
            sur_plv, _, _ = plv_metrics(sur_x)
            surrogate_plv_this_class.append([
                sur_plv[idx_C3, idx_P3],
                sur_plv[idx_C4, idx_P4],
                sur_plv[idx_CZ, idx_PZ],
            ])

        surrogate_plv_edges[subject_number][cls] = np.array(surrogate_plv_this_class)

        # ----- Stability (std + CV) over trials for each metric -----
        for metric_name, data_all in zip(
            ["PLV", "ImagPLV", "PLI"],
            [plv_all, imag_all, pli_all]
        ):
            std_matrix = np.zeros((numElectrodes, numElectrodes))
            cv_matrix  = np.zeros((numElectrodes, numElectrodes))

            for i in range(numElectrodes):
                for j in range(i + 1, numElectrodes):
                    ts = data_all[i, j, :]
                    std_val  = np.std(ts)
                    mean_val = np.mean(ts)
                    cv_val   = std_val / mean_val if mean_val != 0 else 0.0

                    std_matrix[i, j] = std_val
                    std_matrix[j, i] = std_val
                    cv_matrix[i, j]  = cv_val
                    cv_matrix[j, i]  = cv_val

            stability_matrices[subject_number][cls][metric_name] = {
                "std": std_matrix,
                "cv":  cv_matrix,
            }

        # Store mean connectivity matrices over trials for later analyses
        mean_plv_per_subject[subject_number][cls]      = np.mean(plv_all,  axis=2)
        mean_imag_plv_per_subject[subject_number][cls] = np.mean(imag_all, axis=2)
        mean_pli_per_subject[subject_number][cls]      = np.mean(pli_all,  axis=2)

print("All subjects processed.")

# ============================================================
# 1) PLV lateralisation index C3-P3 vs C4-P4
# ============================================================

def compute_lateralisation_indices():
    """
    Lateralisation index for PLV:
      LI = (C3P3 - C4P4) / (C3P3 + C4P4)
    """
    li = {"L": [], "R": []}
    for cls in ["L", "R"]:
        for subj in subject_numbers:
            plv_mean = mean_plv_per_subject[subj][cls]
            c3p3 = plv_mean[idx_C3, idx_P3]
            c4p4 = plv_mean[idx_C4, idx_P4]
            denom = c3p3 + c4p4 + 1e-10
            li_val = (c3p3 - c4p4) / denom
            li[cls].append(li_val)
    return li

li_dict = compute_lateralisation_indices()

for cls in ["L", "R"]:
    vals = np.array(li_dict[cls])
    print(f"PLV lateralisation LI for class {cls}:")
    print("  Values per subject:", vals)
    tstat, pval = ttest_1samp(vals, 0.0)
    print(f"  Mean LI = {vals.mean():.4f}, t-test vs 0: t = {tstat:.3f}, p = {pval:.4f}\n")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["L", "R"], [np.mean(li_dict["L"]), np.mean(li_dict["R"])])
ax.axhline(0, color="k", linewidth=0.8)
ax.set_ylabel("Lateralisation Index (C3P3 - C4P4 / sum)")
ax.set_title("PLV Lateralisation: C3P3 vs C4P4")
plt.tight_layout()
plt.show()

# ============================================================
# 2) Distance vs PLV / Imag_PLV / PLI
# ============================================================

def distance_vs_metric_analysis(metric_dict, metric_name="PLV"):
    """
    metric_dict[subject][cls] = [ch,ch] matrix.
    I want to see how |metric| scales with 3D distance.
    """
    for cls in ["L", "R"]:
        all_d = []
        all_m = []
        for subj in subject_numbers:
            mat = metric_dict[subj][cls]
            for i in range(numElectrodes):
                for j in range(i + 1, numElectrodes):
                    all_d.append(dist_matrix[i, j])
                    all_m.append(np.abs(mat[i, j]))
        all_d = np.array(all_d)
        all_m = np.array(all_m)
        rho, pval = spearmanr(all_d, all_m)
        print(f"{metric_name} vs distance for class {cls}: Spearman rho = {rho:.3f}, p = {pval:.4e}")

        plt.figure(figsize=(5, 4))
        plt.scatter(all_d, all_m, alpha=0.3, s=10)
        plt.xlabel("Inter-electrode distance (3D)")
        plt.ylabel(f"|{metric_name}|")
        plt.title(f"{metric_name} vs distance, class {cls}")
        plt.tight_layout()
        plt.show()

print("\n--- Distance vs metric analysis ---")
distance_vs_metric_analysis(mean_plv_per_subject,      metric_name="PLV")
distance_vs_metric_analysis(mean_imag_plv_per_subject, metric_name="Imag_PLV")
distance_vs_metric_analysis(mean_pli_per_subject,      metric_name="PLI")


def highlight_edge_vs_distance(metric_dict, edge_pairs, metric_name="PLV"):
    """
    Overlay key edges (e.g. C3-P3, C4-P4, Cz-Pz) on distance-vs-metric scatter.
    """
    for cls in ["L", "R"]:
        all_d = []
        all_m = []
        for subj in subject_numbers:
            mat = metric_dict[subj][cls]
            for i in range(numElectrodes):
                for j in range(i + 1, numElectrodes):
                    all_d.append(dist_matrix[i, j])
                    all_m.append(np.abs(mat[i, j]))
        all_d = np.array(all_d)
        all_m = np.array(all_m)

        plt.figure(figsize=(5, 4))
        plt.scatter(all_d, all_m, alpha=0.2, s=10, label="All pairs")

        for (lab_i, lab_j) in edge_pairs:
            i = electrode_labels.index(lab_i)
            j = electrode_labels.index(lab_j)
            d_vals = []
            m_vals = []
            for subj in subject_numbers:
                mat = metric_dict[subj][cls]
                d_vals.append(dist_matrix[i, j])
                m_vals.append(np.abs(mat[i, j]))
            d_vals = np.array(d_vals)
            m_vals = np.array(m_vals)
            plt.scatter(d_vals, m_vals, s=60, label=f"{lab_i}-{lab_j}")

        plt.xlabel("Inter-electrode distance (3D)")
        plt.ylabel(f"|{metric_name}|")
        plt.title(f"{metric_name} vs distance with key edges, class {cls}")
        plt.legend()
        plt.tight_layout()
        plt.show()

key_edges = [("C3", "P3"), ("C4", "P4"), ("Cz", "Pz")]
highlight_edge_vs_distance(mean_plv_per_subject,      key_edges, metric_name="PLV")
highlight_edge_vs_distance(mean_imag_plv_per_subject, key_edges, metric_name="Imag_PLV")
highlight_edge_vs_distance(mean_pli_per_subject,      key_edges, metric_name="PLI")

# ============================================================
# 3) Surrogate null for key edges (PLV)
# ============================================================

print("\n--- Surrogate PLV vs real PLV for C3-P3, C4-P4, Cz-Pz ---")

for cls in ["L", "R"]:
    real_vals = {"C3P3": [], "C4P4": [], "CZPZ": []}
    sur_vals  = {"C3P3": [], "C4P4": [], "CZPZ": []}

    for subj in subject_numbers:
        mat_real = mean_plv_per_subject[subj][cls]
        real_vals["C3P3"].append(mat_real[idx_C3, idx_P3])
        real_vals["C4P4"].append(mat_real[idx_C4, idx_P4])
        real_vals["CZPZ"].append(mat_real[idx_CZ, idx_PZ])

        sur_arr = surrogate_plv_edges[subj][cls]  # [trials, 3]
        sur_vals["C3P3"].append(np.mean(sur_arr[:, 0]))
        sur_vals["C4P4"].append(np.mean(sur_arr[:, 1]))
        sur_vals["CZPZ"].append(np.mean(sur_arr[:, 2]))

    print(f"\nClass {cls}:")
    for edge_name in ["C3P3", "C4P4", "CZPZ"]:
        r = np.array(real_vals[edge_name])
        s = np.array(sur_vals[edge_name])
        tstat, pval = ttest_1samp(r - s, 0.0)
        print(f"  Edge {edge_name}: real vs surrogate mean diff t = {tstat:.3f}, p = {pval:.4f}")
        print(f"    Real mean = {r.mean():.4f}, Surrogate mean = {s.mean():.4f}")

# ============================================================
# 4) Shared stability / CV analysis across subjects
#    for PLV, ImagPLV, PLI
# ============================================================

def get_most_shared_pairs(mean_matrix, std_matrix, top_k=20):
    """
    Given the mean and std matrices (over subjects) for CV values,
    return top_k electrode pairs (i<j) with lowest std (most consistent).
    Each element: ((i, j), mean_cv, std_cv)
    """
    pairs = []
    n = mean_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(((i, j), mean_matrix[i, j], std_matrix[i, j]))
    pairs.sort(key=lambda x: x[2])  # sort by std (ascending)
    return pairs[:top_k]


top_k = 10  # how many shared pairs I want to print/visualise
top_cv_matrices = {m: {"L": None, "R": None} for m in metrics_names}
overall_mean_cv = {m: {"L": None, "R": None} for m in metrics_names}

for metric_name in metrics_names:
    print(f"\n=== Shared stability analysis for {metric_name} ===")
    for cls in ["L", "R"]:
        # Collect CV matrices for all subjects
        all_cv_values = np.zeros((numElectrodes, numElectrodes, len(subject_numbers)))
        for k, subj in enumerate(subject_numbers):
            cv_mat = stability_matrices[subj][cls][metric_name]["cv"]
            all_cv_values[:, :, k] = cv_mat

        mean_cv = np.mean(all_cv_values, axis=2)
        std_cv  = np.std(all_cv_values, axis=2)

        overall_mean_cv[metric_name][cls] = mean_cv.copy()

        top_shared_pairs = get_most_shared_pairs(mean_cv, std_cv, top_k=top_k)

        print(f"  Class {cls}: top {top_k} shared pairs (lowest inter-subject CV std):")
        for (i, j), m_cv, s_cv in top_shared_pairs:
            print(f"    ({electrode_labels[i]}, {electrode_labels[j]}): mean CV = {m_cv:.4f}, std CV = {s_cv:.4f}")

        # Build a 19x19 matrix with mean CV only at top shared pairs
        top_matrix = np.zeros((numElectrodes, numElectrodes))
        for (i, j), m_cv, s_cv in top_shared_pairs:
            top_matrix[i, j] = m_cv
            top_matrix[j, i] = m_cv
        top_cv_matrices[metric_name][cls] = top_matrix

# Visualise top shared pairs matrices for each metric and class
for metric_name in metrics_names:
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for idx, cls in enumerate(["L", "R"]):
        im = axs[idx].imshow(top_cv_matrices[metric_name][cls],
                             cmap="viridis", interpolation="nearest")
        axs[idx].set_title(f"{metric_name}: Top shared pairs, class {cls}")
        axs[idx].set_xticks(np.arange(numElectrodes))
        axs[idx].set_yticks(np.arange(numElectrodes))
        axs[idx].set_xticklabels(electrode_labels, rotation=90)
        axs[idx].set_yticklabels(electrode_labels)
        axs[idx].set_xlabel("Electrodes")
        axs[idx].set_ylabel("Electrodes")
        fig.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04, label="Mean CV")
    fig.tight_layout()
    plt.show()

# Also visualise overall mean CV matrices (all pairs) for comparison
for metric_name in metrics_names:
    for cls in ["L", "R"]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(overall_mean_cv[metric_name][cls],
                       cmap="RdYlGn", interpolation="nearest")
        ax.set_title(f"{metric_name}: Overall mean CV, class {cls}")
        ax.set_xticks(np.arange(numElectrodes))
        ax.set_yticks(np.arange(numElectrodes))
        ax.set_xticklabels(electrode_labels, rotation=90)
        ax.set_yticklabels(electrode_labels)
        ax.set_xlabel("Electrodes")
        ax.set_ylabel("Electrodes")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean CV")
        fig.tight_layout()
        plt.show()

# ============================================================
# 5) Average connectivity matrices (for hub visualization)
# ============================================================

def plot_metric_matrix(metric_dict, cls, metric_name="PLV"):
    """
    Plot the across-subject average connectivity matrix for one class.
    This is just to see the hub structure visually.
    """
    mats = []
    for subj in subject_numbers:
        mats.append(metric_dict[subj][cls])
    mats = np.stack(mats, axis=0)
    avg_mat = np.mean(mats, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_mat, cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(numElectrodes))
    ax.set_yticks(np.arange(numElectrodes))
    ax.set_xticklabels(electrode_labels, rotation=90)
    ax.set_yticklabels(electrode_labels)
    ax.set_title(f"{metric_name} average matrix, class {cls}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

for cls in ["L", "R"]:
    plot_metric_matrix(mean_plv_per_subject,      cls, metric_name="PLV")
    plot_metric_matrix(mean_imag_plv_per_subject, cls, metric_name="Imag_PLV")
    plot_metric_matrix(mean_pli_per_subject,      cls, metric_name="PLI")

print("Analysis complete.")
