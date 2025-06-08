import numpy as np
import extreme_matmul
import time
import torch

# Batch dimensions
B, M, K, N = 10, 8164, 8164, 2048

A_batch = np.random.rand(B, M, K).astype(np.float32)
B_batch = np.random.rand(B, K, N).astype(np.float32)

def benchmark(name, matmul_fn, A_batch, B_batch):
    times = []
    for i in range(len(A_batch)):
        A = A_batch[i]
        B = B_batch[i]

        start = time.perf_counter()
        result = matmul_fn(A, B)
        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"{name:<35}: {avg_time:.6f} seconds per matmul (avg over {len(A_batch)})")
    #print(result)
    
print("large matrix: 8164*2048")
benchmark("extreme_matmul.fast_matmul", extreme_matmul.matmul, A_batch, B_batch)
benchmark("NumPy @ operator", np.matmul, A_batch, B_batch)
with torch.no_grad():
  benchmark("torch optim: ", torch.matmul, torch.from_numpy(A_batch), torch.from_numpy(B_batch))



B, M, K, N = 10, 128, 256, 256

A_batch = np.random.rand(B, M, K).astype(np.float32)
B_batch = np.random.rand(B, K, N).astype(np.float32)

print("smaller matrix: 128*256")

def benchmark(name, matmul_fn, A_batch, B_batch):
    times = []
    for i in range(len(A_batch)):
        A = A_batch[i]
        B = B_batch[i]

        start = time.perf_counter()
        result = matmul_fn(A, B)
        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"{name:<35}: {avg_time:.6f} seconds per matmul (avg over {len(A_batch)})")
    #print(result)

# Run benchmarks
benchmark("extreme_matmul.fast_matmul", extreme_matmul.matmul, A_batch, B_batch)
benchmark("NumPy @ operator", np.matmul, A_batch, B_batch)
with torch.no_grad():
  benchmark("torch optim: ", torch.matmul, torch.from_numpy(A_batch), torch.from_numpy(B_batch))