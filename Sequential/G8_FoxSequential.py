"""
Sequential Fox's Algorithm with CSV Logging and Plot Generation
Author: Group 8
Course: Software and Parallel Development
"""

import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# CSV Helper
# -----------------------------
def save_csv_row(filepath, row):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------
# Plot Helper
# -----------------------------
def generate_plots(csv_path, output_folder="plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_path)

    # ---- Runtime vs N ----
    plt.figure()
    for q in sorted(df["q"].unique()):
        subset = df[df["q"] == q]
        plt.plot(subset["N"], subset["time"], marker="o", label=f"q={q}")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Runtime (seconds)")
    plt.title("Sequential Fox: Runtime vs N")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_N.png")
    plt.close()

    # ---- Runtime vs q ----
    plt.figure()
    for N in sorted(df["N"].unique()):
        subset = df[df["N"] == N]
        plt.plot(subset["q"], subset["time"], marker="o", label=f"N={N}")
    plt.xlabel("q (process grid size)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Sequential Fox: Runtime vs q")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_q.png")
    plt.close()

    # ---- Relative Error ----
    plt.figure()
    plt.plot(df["N"], df["error"], marker="x", linestyle="--")
    plt.xlabel("N")
    plt.ylabel("Relative Error")
    plt.title("Sequential Fox: Relative Error vs N")
    plt.grid()
    plt.savefig(f"{output_folder}/relative_error.png")
    plt.close()


# -----------------------------
# Sequential Fox Algorithm
# -----------------------------
def fox_sequential(A, B, q):
    N = A.shape[0]
    n_local = N // q
    C = np.zeros((N, N), dtype=np.float32)

    # Split into blocks
    A_blocks = []
    B_blocks = []
    for i in range(q):
        rowA, rowB = [], []
        for j in range(q):
            rowA.append(A[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local])
            rowB.append(B[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local])
        A_blocks.append(rowA)
        B_blocks.append(rowB)

    # Fox algorithm (sequential)
    for stage in range(q):
        new_B = [[None]*q for _ in range(q)]

        for i in range(q):
            k = (i + stage) % q
            A_bcast = A_blocks[i][k]

            for j in range(q):
                C[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local] += \
                    A_bcast @ B_blocks[k][j]

        # Circular shift B upward
        for i in range(q):
            for j in range(q):
                new_B[i][j] = B_blocks[(i+1) % q][j]
        B_blocks = new_B

    return C


# -----------------------------
# Full Experiment Runner
# -----------------------------
def run_sequential_tests():
    csv_path = "sequential/sequential.csv"

    # Clear previous CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Final parameters (standard for entire project)
    Ns = [256, 512, 1024]
    qs = [1, 2, 4]

    print("\nRunning Sequential Fox Algorithm...\n")

    for N in Ns:
        for q in qs:
            if N % q != 0:
                print(f"Skipping N={N}, q={q} (not divisible)")
                continue

            np.random.seed(0)
            A = np.random.rand(N, N).astype(np.float32)
            B = np.random.rand(N, N).astype(np.float32)

            start = time.perf_counter()
            C = fox_sequential(A, B, q)
            end = time.perf_counter()
            runtime = end - start

            # Correctness check
            C_ref = A @ B
            error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)

            print(f"N={N:<5} | q={q:<2} | Time={runtime:.4f}s | Error={error:.2e}")

            save_csv_row(csv_path, {
                "N": N,
                "q": q,
                "time": runtime,
                "error": error,
                "hardware": "CPU (sequential)"
            })

    # After experiments, generate plots
    generate_plots(csv_path, "sequential/plots")
    print("\nPlots saved to sequential/plots/")
    print(f"CSV saved at: {csv_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_sequential_tests()
