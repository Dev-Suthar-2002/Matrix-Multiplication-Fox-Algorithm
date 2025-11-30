"""
OpenMP-Style Fox's Algorithm using Python Multiprocessing
Author: Group 8
Course: Software & Parallel Development
"""

import numpy as np
import time
import csv
import os
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# CSV Helper
# ==========================================================
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


# ==========================================================
# Plot Helper
# ==========================================================
def generate_plots(csv_path, output_folder="openmp/plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_path)

    # ------------------------------------
    # Runtime vs N
    # ------------------------------------
    plt.figure()
    for q in sorted(df["q"].unique()):
        subset = df[df["q"] == q]
        plt.plot(subset["N"], subset["time"], marker="o", label=f"q={q}")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Runtime (seconds)")
    plt.title("OpenMP Fox: Runtime vs Matrix Size")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_N.png")
    plt.close()

    # ------------------------------------
    # Runtime vs q
    # ------------------------------------
    plt.figure()
    for N in sorted(df["N"].unique()):
        subset = df[df["N"] == N]
        plt.plot(subset["q"], subset["time"], marker="o", label=f"N={N}")
    plt.xlabel("q (Grid Size)")
    plt.ylabel("Runtime (seconds)")
    plt.title("OpenMP Fox: Runtime vs q")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_q.png")
    plt.close()

    # ------------------------------------
    # Runtime vs #Processes
    # ------------------------------------
    plt.figure()
    for N in sorted(df["N"].unique()):
        subset = df[df["N"] == N]
        for q in sorted(subset["q"].unique()):
            sub = subset[subset["q"] == q]
            plt.plot(sub["procs"], sub["time"], marker="o",
                     label=f"N={N}, q={q}")
    plt.xlabel("# Processes Used")
    plt.ylabel("Runtime (seconds)")
    plt.title("OpenMP Fox: Runtime vs Number of Processes")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_procs.png")
    plt.close()

    # ------------------------------------
    # Relative Error Plot
    # ------------------------------------
    plt.figure()
    plt.plot(df["N"], df["error"], marker="x", linestyle="--")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Relative Error")
    plt.title("OpenMP Fox: Relative Error")
    plt.grid()
    plt.savefig(f"{output_folder}/relative_error.png")
    plt.close()


# ==========================================================
# Worker Function
# ==========================================================
def fox_worker(args):
    """
    Calculates one row i of C for all j.
    Equivalent to OpenMP private(i) parallel for collapse(1).
    """
    i, q, A_blocks, B_blocks, n_local = args
    C_row = [np.zeros((n_local, n_local), dtype=np.float32) for _ in range(q)]

    with threadpool_limits(limits=1):  # ensure np.dot is 1-threaded
        B_local = [row[:] for row in B_blocks]  # local copy

        for stage in range(q):
            k = (i + stage) % q
            A_bcast = A_blocks[i][k]

            for j in range(q):
                C_row[j] += A_bcast @ B_local[k][j]

            # circular shift B UP
            new_B = [[None]*q for _ in range(q)]
            for r in range(q):
                for c in range(q):
                    new_B[r][c] = B_local[(r+1) % q][c]
            B_local = new_B

    return i, C_row


# ==========================================================
# OpenMP-Style Fox
# ==========================================================
def fox_openmp(A, B, q, nprocs):
    N = A.shape[0]
    n_local = N // q

    # Build blocks
    A_blocks = []
    B_blocks = []
    for i in range(q):
        rowA, rowB = [], []
        for j in range(q):
            rowA.append(A[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local])
            rowB.append(B[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local])
        A_blocks.append(rowA)
        B_blocks.append(rowB)

    # Parallel execution per row (i)
    args = [(i, q, A_blocks, B_blocks, n_local) for i in range(q)]

    with Pool(processes=min(nprocs, q)) as pool:
        results = pool.map(fox_worker, args)

    # Rebuild C
    C = np.zeros((N, N), dtype=np.float32)
    results.sort(key=lambda x: x[0])  # sort by row index
    C_blocks = [row for _, row in results]

    for i in range(q):
        for j in range(q):
            C[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local] = C_blocks[i][j]

    return C


# ==========================================================
# Full Experiment Runner
# ==========================================================
def run_openmp_tests():
    csv_path = "openmp/openmp.csv"

    # Clear old CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # FINAL TEST PARAMETERS (same across project)
    Ns = [256, 512, 1024]
    qs = [1, 2, 4]
    procs_list = [1, 2, 4, 8]

    print("\nRunning OpenMP-Style Fox Algorithm...\n")

    for N in Ns:
        for q in qs:
            if N % q != 0:
                print(f"Skipping N={N}, q={q}")
                continue

            for p in procs_list:
                # Create data
                np.random.seed(0)
                A = np.random.rand(N, N).astype(np.float32)
                B = np.random.rand(N, N).astype(np.float32)

                # Run timed test
                start = time.perf_counter()
                C = fox_openmp(A, B, q, p)
                end = time.perf_counter()

                runtime = end - start
                C_ref = A @ B
                error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)

                print(f"N={N:<5} | q={q:<2} | procs={p:<2} | "
                      f"Time={runtime:.4f}s | Error={error:.2e}")

                save_csv_row(csv_path, {
                    "N": N,
                    "q": q,
                    "procs": p,
                    "time": runtime,
                    "error": error,
                    "hardware": "CPU multiprocessing"
                })

    generate_plots(csv_path)
    print("\nPlots saved to openmp/plots/")
    print(f"CSV saved at: {csv_path}")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    run_openmp_tests()
