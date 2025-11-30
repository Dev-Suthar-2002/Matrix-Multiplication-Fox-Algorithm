# Matrix-Multiplication-Fox-Algorithm

This project implements the FOX Parallel Matrix Multiplication Algorithm using Python for high-performance, distributed, and GPU-accelerated computation. It demonstrates parallel algorithm design using NumPy, Numba CUDA, and MPI4Py, along with performance benchmarking on multi-core and multi-node systems.

## overview
The FOX algorithm is an efficient method for performing matrix multiplication on a 2D process grid. This implementation supports:
- Multi-core CPU parallelism
- CUDA GPU acceleration (Numba CUDA)
- Distributed execution using MPI (MPI4Py)
- Scalable performance across q × q process grids
- Benchmarking for runtime, speedup, and memory usage

## Key Features

✔️ Fully implemented FOX algorithm with block matrix decomposition

✔️ Supports q × q distributed process grid

✔️ Optimized GPU kernel using Numba CUDA

✔️ MPI-based broadcasting and data shifting logic

✔️ Benchmarks for N = 256–2048 with speedup up to 12×

✔️ Handles race conditions, synchronization issues, and CUDA memory errors

✔️ Produces validated results compared to serial baseline multiplication

⚙️ Technologies Used
- Python 3.10+
- NumPy
- Numba (CUDA kernels)
- MPI4Py
- CUDA Toolkit (11 or higher)
- Matplotlib (for plotting benchmark results)

📘 Algorithm Summary
- The FOX algorithm divides matrices into q × q sub-blocks, then:
- Broadcasts appropriate sub-blocks across each row
- Each process performs local matrix multiplication
- Cyclically shifts sub-blocks of matrix B upward
- Repeats for q iterations to compute local sub-results
- Collects and assembles the final output matrix
- This minimizes communication overhead and ensures balanced workload distribution.
