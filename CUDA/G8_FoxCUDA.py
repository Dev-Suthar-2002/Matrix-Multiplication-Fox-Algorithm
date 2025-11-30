"""
fox_cuda_numba.py
Numba-CUDA implementation of Fox's Algorithm (single GPU)
Author: Group 8
"""

import numpy as np
import time
import os
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd

from numba import cuda, float32

# ----------------------------
# CSV helper
# ----------------------------
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

# ----------------------------
# Plot helper
# ----------------------------
def generate_plots(csv_path, output_folder="cuda/plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df = pd.read_csv(csv_path)
    # Runtime vs N
    plt.figure()
    for q in sorted(df["q"].unique()):
        sub = df[df["q"] == q]
        plt.plot(sub["N"], sub["time"], marker="o", label=f"q={q}")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Runtime (s)")
    plt.title("CUDA Fox (Numba) - Runtime vs N")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_N.png")
    plt.close()

    # Runtime vs q
    plt.figure()
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        plt.plot(sub["q"], sub["time"], marker="o", label=f"N={N}")
    plt.xlabel("q (grid size)")
    plt.ylabel("Runtime (s)")
    plt.title("CUDA Fox (Numba) - Runtime vs q")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_q.png")
    plt.close()

    # Runtime vs procs (GPU is single device)
    plt.figure()
    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        plt.plot(sub["procs"], sub["time"], marker="o", label=f"N={N}")
    plt.xlabel("Processes / Devices")
    plt.ylabel("Runtime (s)")
    plt.title("CUDA Fox (Numba) - Runtime vs procs")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/runtime_vs_procs.png")
    plt.close()

    # Relative error
    plt.figure()
    plt.plot(df["N"], df["error"], marker="x", linestyle="--")
    plt.xlabel("Matrix Size N")
    plt.ylabel("Relative Error")
    plt.title("CUDA Fox (Numba) - Relative Error vs N")
    plt.grid()
    plt.savefig(f"{output_folder}/relative_error.png")
    plt.close()

# ----------------------------
# Tiled matrix multiplication kernel
# Each kernel computes: C += A @ B for one block (n x n)
# Uses 2D grid and 2D blocks with TILE x TILE tiles
# ----------------------------
TILE = 16  # choose 16 or 32 depending on GPU capabilities

@cuda.jit
def matmul_tiled(A, B, C, n):
    # shared memory tiles
    sA = cuda.shared.array((TILE, TILE), dtype=float32)
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y

    row = by * bdy + ty
    col = bx * bdx + tx

    tmp = 0.0

    # Loop over tiles
    tiles = (n + TILE - 1) // TILE
    for t in range(tiles):
        # Load A tile
        a_row = row
        a_col = t * TILE + tx
        if a_row < n and a_col < n:
            sA[ty, tx] = A[a_row, a_col]
        else:
            sA[ty, tx] = 0.0

        # Load B tile
        b_row = t * TILE + ty
        b_col = col
        if b_row < n and b_col < n:
            sB[ty, tx] = B[b_row, b_col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        # multiply tile
        for k in range(TILE):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < n and col < n:
        # accumulate into C (C += tmp)
        C[row, col] += tmp

# ----------------------------
# Helper: launch kernel for one block multiply-add
# ----------------------------
def launch_block_matmul(A_dev, B_dev, C_dev, n):
    # grid and block dims
    threads_per_block = (TILE, TILE)
    blocks_per_grid_x = math.ceil(n / threads_per_block[0])
    blocks_per_grid_y = math.ceil(n / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    matmul_tiled[blocks_per_grid, threads_per_block](A_dev, B_dev, C_dev, n)
    # don't sync here; caller will sync after stage if desired

# ----------------------------
# Fox algorithm on GPU (single device)
# ----------------------------
def fox_cuda_numba(A_cpu, B_cpu, q):
    """
    A_cpu, B_cpu: host numpy arrays (N x N)
    q: grid size
    returns: C_cpu (host numpy array), elapsed_time
    """
    N = A_cpu.shape[0]
    if N % q != 0:
        raise ValueError("N must be divisible by q")
    n = N // q

    # Partition blocks and copy each block to device (keep device arrays persistent)
    A_dev_blocks = [[None] * q for _ in range(q)]
    B_dev_blocks = [[None] * q for _ in range(q)]
    C_dev_blocks = [[None] * q for _ in range(q)]

    # allocate device arrays for each block
    for i in range(q):
        for j in range(q):
            A_blk = np.ascontiguousarray(A_cpu[i*n:(i+1)*n, j*n:(j+1)*n])
            B_blk = np.ascontiguousarray(B_cpu[i*n:(i+1)*n, j*n:(j+1)*n])
            C_blk = np.zeros((n, n), dtype=np.float32)

            A_dev_blocks[i][j] = cuda.to_device(A_blk)
            B_dev_blocks[i][j] = cuda.to_device(B_blk)
            C_dev_blocks[i][j] = cuda.to_device(C_blk)

    # run stages on device, avoiding host-device transfers in the inner loop
    start = time.perf_counter()

    for stage in range(q):
        # For each block-row i, choose A_bcast = A[i][k] with k = (i+stage)%q
        for i in range(q):
            k = (i + stage) % q
            A_bcast_dev = A_dev_blocks[i][k]  # device array (n x n)

            for j in range(q):
                B_dev = B_dev_blocks[k][j]
                C_dev = C_dev_blocks[i][j]
                # launch kernel to perform: C_dev += A_bcast_dev @ B_dev
                launch_block_matmul(A_bcast_dev, B_dev, C_dev, n)

        # synchronize to ensure all kernels for this stage complete
        cuda.synchronize()

        # circular shift of B blocks upward (rotate rows)
        newB = [[None] * q for _ in range(q)]
        for ii in range(q):
            for jj in range(q):
                newB[ii][jj] = B_dev_blocks[(ii+1) % q][jj]
        B_dev_blocks = newB

    # Done: copy C blocks back to host and assemble
    cuda.synchronize()
    C_cpu = np.zeros((N, N), dtype=np.float32)
    for i in range(q):
        for j in range(q):
            C_cpu[i*n:(i+1)*n, j*n:(j+1)*n] = C_dev_blocks[i][j].copy_to_host()

    elapsed = time.perf_counter() - start
    return C_cpu, elapsed

# ----------------------------
# Experiment runner
# ----------------------------
def run_cuda_tests():
    csv_path = "cuda/cuda.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # test parameters (standardized)
    Ns = [512, 1024, 2048]
    qs = [1, 2, 4]

    # check GPU availability
    try:
        n_devices = len(cuda.gpus)
        if n_devices == 0:
            print("No CUDA GPU found. Exiting CUDA tests.")
            return
    except Exception as e:
        print("CUDA not available or numba cuda couldn't initialize:", e)
        return

    print("Running CUDA Fox (Numba) tests...\n")
    for N in Ns:
        for q in qs:
            if N % q != 0:
                print(f"Skipping N={N}, q={q} (not divisible)")
                continue

            # create test matrices on host
            np.random.seed(0)
            A = np.random.rand(N, N).astype(np.float32)
            B = np.random.rand(N, N).astype(np.float32)

            try:
                # run fox on GPU
                C_gpu, runtime = fox_cuda_numba(A, B, q)

                # reference and error
                C_ref = A @ B
                rel_err = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)

                print(f"N={N:<5} | q={q:<2} | Time={runtime:.4f}s | RelErr={rel_err:.2e}")

                save_csv_row(csv_path, {
                    "N": N,
                    "q": q,
                    "procs": 1,            # single GPU device for this script
                    "time": runtime,
                    "error": rel_err,
                    "hardware": f"CUDA (Numba)"
                })

            except Exception as ex:
                print(f"Error running N={N}, q={q}: {ex}")

    # generate plots
    if os.path.exists(csv_path):
        generate_plots(csv_path)
        print("\nPlots saved to cuda/plots/")
        print(f"CSV saved at: {csv_path}")
    else:
        print("No CSV produced; skipping plotting.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    run_cuda_tests()
