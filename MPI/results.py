import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# READ RESULTS FROM EXCEL (instead of hardcoding)
# ------------------------------------------------
# Make sure the Excel file is in the same folder as this script.
# Expected columns:
# N | q | procs | time | error | hardware
df = pd.read_excel("mpi_results.xlsx", sheet_name="Sheet1")

# ------------------------------------------------
# Runtime vs N (separate lines for q)
# ------------------------------------------------
plt.figure(figsize=(8, 6))
for q in sorted(df["q"].unique()):
    subset = df[df["q"] == q]
    plt.plot(subset["N"], subset["time"], marker="o", label=f"q={q}")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Runtime (seconds)")
plt.title("MPI Fox Algorithm - Runtime vs N")
plt.grid(True)
plt.legend()
plt.savefig("mpi_runtime_vs_N.png")
plt.close()

# ------------------------------------------------
# Runtime vs q (separate lines for N)
# ------------------------------------------------
plt.figure(figsize=(8, 6))
for N in sorted(df["N"].unique()):
    subset = df[df["N"] == N]
    plt.plot(subset["q"], subset["time"], marker="o", label=f"N={N}")
plt.xlabel("q (Grid Size)")
plt.ylabel("Runtime (seconds)")
plt.title("MPI Fox Algorithm - Runtime vs q")
plt.grid(True)
plt.legend()
plt.savefig("mpi_runtime_vs_q.png")
plt.close()

# ------------------------------------------------
# Runtime vs number of processes
# ------------------------------------------------
plt.figure(figsize=(8, 6))
for N in sorted(df["N"].unique()):
    subset = df[df["N"] == N]
    plt.plot(subset["procs"], subset["time"], marker="o", label=f"N={N}")
plt.xlabel("Number of MPI Processes")
plt.ylabel("Runtime (seconds)")
plt.title("MPI Fox Algorithm - Runtime vs Processes")
plt.grid(True)
plt.legend()
plt.savefig("mpi_runtime_vs_procs.png")
plt.close()

# ------------------------------------------------
# Relative Error vs N
# ------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(df["N"], df["error"], marker="x", linestyle="--", color="red")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Relative Error")
plt.title("MPI Fox Algorithm - Relative Error vs N")
plt.grid(True)
plt.savefig("mpi_relative_error.png")
plt.close()

print("Plots generated successfully:")
print(" - mpi_runtime_vs_N.png")
print(" - mpi_runtime_vs_q.png")
print(" - mpi_runtime_vs_procs.png")
print(" - mpi_relative_error.png")
