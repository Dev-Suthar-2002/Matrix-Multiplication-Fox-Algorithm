"""
MPI Fox's Algorithm with CSV Logging + Plot Generation
Author: Group 8
Course: Software and Parallel Development
"""

from mpi4py import MPI
import numpy as np
import argparse
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits


# ================================================================
# CSV Helper
# ================================================================
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


# ================================================================
# Plot Helper (Root Only)
# ================================================================
def generate_plots(csv_path, output_folder="mpi/plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_path)

    # Runtime vs N
    plt.figure()
    for q in sorted(df["q"].unique()):
        subset = df[df["q"] == q]
        plt.plot(subset["N"], subset["time"], marker="o", label=f"q={q}")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Runtime (seconds)")
    plt.title("MPI Fox: Runtime vs N")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_N.png")
    plt.close()

    # Runtime vs q
    plt.figure()
    for N in sorted(df["N"].unique()):
        subset = df[df["N"] == N]
        plt.plot(subset["q"], subset["time"], marker="o", label=f"N={N}")
    plt.xlabel("q (Grid Size)")
    plt.ylabel("Runtime (seconds)")
    plt.title("MPI Fox: Runtime vs q")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_q.png")
    plt.close()

    # Runtime vs Processes
    plt.figure()
    for N in sorted(df["N"].unique()):
        subset = df[df["N"] == N]
        plt.plot(subset["procs"], subset["time"], marker="o",
                 label=f"N={N}")
    plt.xlabel("Total MPI Processes")
    plt.ylabel("Runtime (seconds)")
    plt.title("MPI Fox: Runtime vs #MPI Processes")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_procs.png")
    plt.close()

    # Relative Error
    plt.figure()
    plt.plot(df["N"], df["error"], marker="x", linestyle="--")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Relative Error")
    plt.title("MPI Fox: Relative Error")
    plt.grid()
    plt.savefig(f"{output_folder}/relative_error.png")
    plt.close()


# ================================================================
# MPI Fox Implementation
# ================================================================
def fox_mpi(N, q):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != q * q:
        if rank == 0:
            print(f"ERROR: Need {q*q} MPI processes for q={q}")
        return None

    n_local = N // q

    A_local = np.zeros((n_local, n_local), dtype=np.float32)
    B_local = np.zeros((n_local, n_local), dtype=np.float32)
    C_local = np.zeros((n_local, n_local), dtype=np.float32)

    # Root creates full matrix and flattens blocks
    if rank == 0:
        np.random.seed(0)
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)

        blocks_A = []
        blocks_B = []
        for i in range(q):
            for j in range(q):
                blocks_A.append(
                    A[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local].flatten()
                )
                blocks_B.append(
                    B[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local].flatten()
                )

        sendbuf_A = np.ascontiguousarray(np.concatenate(blocks_A))
        sendbuf_B = np.ascontiguousarray(np.concatenate(blocks_B))
    else:
        sendbuf_A = None
        sendbuf_B = None

    recvbuf_A = np.zeros(n_local * n_local, dtype=np.float32)
    recvbuf_B = np.zeros(n_local * n_local, dtype=np.float32)

    # Scatter blocks
    comm.Scatter(sendbuf_A, recvbuf_A, root=0)
    comm.Scatter(sendbuf_B, recvbuf_B, root=0)

    A_local = recvbuf_A.reshape((n_local, n_local))
    B_local = recvbuf_B.reshape((n_local, n_local))

    # Determine row/col communicators
    my_row = rank // q
    my_col = rank % q

    row_comm = comm.Split(color=my_row, key=my_col)
    col_comm = comm.Split(color=my_col, key=my_row)

    # =======================
    # FOX Algorithm
    # =======================
    with threadpool_limits(limits=1):
        for stage in range(q):
            root = (my_row + stage) % q

            # Broadcast A block across the row
            if my_col == root:
                A_bcast = A_local.copy()
            else:
                A_bcast = np.zeros((n_local, n_local), dtype=np.float32)

            row_comm.Bcast(A_bcast, root=root)

            # Multiply
            C_local += A_bcast @ B_local

            # Shift B upward
            above = ((my_row - 1) % q) * q + my_col
            below = ((my_row + 1) % q) * q + my_col
            comm.Sendrecv_replace(B_local, dest=above, source=below)

    # Gather results
    flat_C = C_local.flatten()
    recvbuf_C = None
    if rank == 0:
        recvbuf_C = np.zeros(N * N, dtype=np.float32)

    comm.Gather(flat_C, recvbuf_C, root=0)

    # Reconstruct matrix on root
    if rank == 0:
        C = np.zeros((N, N), dtype=np.float32)
        idx = 0
        for i in range(q):
            for j in range(q):
                C[i*n_local:(i+1)*n_local, j*n_local:(j+1)*n_local] = \
                    recvbuf_C[idx*n_local*n_local:(idx+1)*n_local*n_local].reshape((n_local, n_local))
                idx += 1
        return C
    return None


# ================================================================
# Experiment Runner (Root)
# ================================================================
def run_mpi_tests():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    csv_path = "mpi/mpi.csv"

    if rank == 0:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # parameters (same across whole project)
    Ns = [256, 512, 1024]
    qs = [1, 2, 4]  # implies 1,4,16 MPI processes

    for N in Ns:
        for q in qs:
            if q * q != size:
                if rank == 0:
                    print(f"Skipping q={q}, needs {q*q} processes, current={size}")
                continue

            if N % q != 0:
                continue

            comm.Barrier()
            start = MPI.Wtime()

            C = fox_mpi(N, q)

            comm.Barrier()
            end = MPI.Wtime()
            runtime = end - start

            if rank == 0:
                # correctness
                np.random.seed(0)
                A = np.random.rand(N, N).astype(np.float32)
                B = np.random.rand(N, N).astype(np.float32)
                C_ref = A @ B
                error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)

                print(f"MPI Fox | N={N:<5}, q={q:<2}, procs={size:<3}, "
                      f"Time={runtime:.4f}s | Error={error:.2e}")

                save_csv_row(csv_path, {
                    "N": N,
                    "q": q,
                    "procs": size,
                    "time": runtime,
                    "error": error,
                    "hardware": "MPI Cluster/Local"
                })

    # Only root generates plots
    if rank == 0:
        generate_plots(csv_path)
        print("\nPlots saved to mpi/plots/")
        print(f"CSV saved at: {csv_path}")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    run_mpi_tests()
