# ⚡ extreme_matmul
Fast matrix multiplication in Python using Intel MKL

*Because sometimes NumPy just isn't fast enough.*

## What's This About?

Ever found yourself waiting around for large matrix multiplications to finish? Yeah, me too. That's why I built **Extreme MatMul** - a custom Python extension that leverages Intel's Math Kernel Library (MKL) to make matrix multiplication blazingly fast.

This project started as a deep dive into understanding how high-performance computing libraries work under the hood. What I discovered was pretty eye-opening: with the right approach, we can achieve some serious performance gains over standard NumPy operations.

## Performance Results

Here's what really matters - the numbers don't lie:

### Large Matrices (8164 × 8164 × 2048)
```
===============================================
extreme_matmul.fast_matmul    : 1.74 seconds ⚡
NumPy @ operator             : 7.14 seconds 🐌
torch optim                  : 1.97 seconds
===============================================
```

### Smaller Matrices (128 × 256 × 256)
```
===============================================
extreme_matmul.fast_matmul    : 0.35ms ⚡
NumPy @ operator             : 19.25ms
torch optim                  : 9.96ms
===============================================

```
<p align="right"><small><i>(Tests were ran on Intel(R) Xeon(R) CPU @ 2.00GHz)</i></small></p>

**That's roughly 4x faster than NumPy for large matrices and 55x faster for smaller ones!**


## How It Works

The magic happens through a few key optimizations:

1. **Intelligent Algorithm Selection**: For tiny matrices (≤32×32), we use a simple triple-loop implementation that's actually faster due to reduced overhead. For everything else, we call Intel MKL's optimized BLAS routines.

2. **Direct MKL Integration**: Instead of going through NumPy's abstraction layers, we talk directly to Intel's Math Kernel Library - the same engine that powers many scientific computing applications.

3. **Memory Layout Optimization**: The code ensures data is contiguous in memory and properly aligned for SIMD operations.

4. **Flexible Broadcasting**: Supports 1D, 2D, and 3D arrays with proper broadcasting rules, just like NumPy.

## Installation

### Prerequisites
You'll need Intel MKL installed on your system. Here's how to get everything set up:

```bash
# Update your system
sudo apt update
sudo apt install -y gpg-agent wget

# Add Intel's repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# Install Intel MKL
sudo apt update
sudo apt install intel-oneapi-mkl-devel

# Set up the environment and install
source /opt/intel/oneapi/setvars.sh
pip install git+https://github.com/dellano54/extreme-matmul.git
```

### Build from Source
```bash
git clone https://github.com/yourusername/extreme-matmul.git
cd extreme-matmul
source /opt/intel/oneapi/setvars.sh  # Make sure MKL is in your environment
python setup.py build_ext --inplace
```

## Usage

It's designed to be a drop-in replacement for NumPy's matrix multiplication:

```python
import numpy as np
import extreme_matmul

# Create some test matrices
A = np.random.rand(1000, 1000).astype(np.float32)
B = np.random.rand(1000, 1000).astype(np.float32)

# Use extreme_matmul instead of np.matmul
result = extreme_matmul.matmul(A, B)

# That's it! Same API, much faster performance
```

### Supported Operations

- **Vector × Vector**: Dot product
- **Matrix × Vector**: Matrix-vector multiplication  
- **Matrix × Matrix**: Standard matrix multiplication
- **Batch Operations**: 3D arrays with batch dimensions
- **Mixed Dimensions**: Flexible broadcasting like NumPy

### Important Notes

- **Float32 Only**: Currently optimized for 32-bit floating point operations
- **Memory Requirements**: Arrays are converted to contiguous format if needed
- **Array Limits**: Supports 1D to 3D arrays (batch processing for 3D)

## Running the Benchmarks

Want to see the performance difference yourself?

```python
python benchmark.py
```

This will run the same tests I used to generate the performance numbers above. The benchmark tests both large and small matrix scenarios to show how the algorithm selection works.

## Technical Deep Dive

### Algorithm Selection Strategy
The code uses a size-based heuristic to choose between algorithms:
- **Tiny matrices** (≤32×32): Simple triple-loop implementation
- **Larger matrices**: Intel MKL's `cblas_sgemm` with full optimizations

### Memory Management
- Automatic conversion to contiguous arrays when needed
- Proper reference counting to prevent memory leaks
- Aligned memory access for optimal SIMD performance

### Error Handling
Comprehensive input validation including:
- Type checking (float32 requirement)
- Dimension compatibility verification
- Memory allocation error handling

## Why This Matters

This project demonstrates several important concepts:

1. **The Power of Specialized Libraries**: MKL is heavily optimized for Intel processors with years of engineering behind it
2. **Algorithm Selection**: Sometimes simpler algorithms win for small inputs due to reduced overhead
3. **C Extensions**: How to write efficient Python extensions that rival compiled languages
4. **Memory Layout**: The importance of cache-friendly data structures

## Limitations & Future Work

- **Platform Dependency**: Currently requires Intel MKL (Linux/Intel processors)
- **Data Type Limitation**: Only supports float32 (could be extended)
- **GPU Support**: No CUDA implementation yet (interesting future direction)

## Contributing

Found a bug or have an idea for improvement? Feel free to open an issue or submit a pull request. This started as a learning project, but I'm always interested in making it better!

## License

MIT License - feel free to use this in your own projects.

---

*Built with curiosity and a need for speed. If you find this useful, consider giving it a star! ⭐*
