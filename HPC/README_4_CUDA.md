# HPC Practical 4: GPU Computing with CUDA — Vector Addition and Matrix Multiplication

## Objective

Implement GPU-parallel programs using CUDA for two fundamental operations: **Vector Addition** and **Matrix Multiplication**. Compare CPU sequential execution time vs GPU parallel execution time to understand how GPUs accelerate data-parallel computations.

---

## Theory / Background

### What is CUDA?
CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model created by **NVIDIA**. It allows you to write C/C++ code that runs on NVIDIA GPUs (Graphics Processing Units) — massively parallel processors with thousands of cores.

**Key Idea:** A GPU has thousands of smaller, simpler cores optimized for doing many simple calculations simultaneously (vs a CPU which has fewer but more powerful cores). CUDA lets you use those GPU cores for general computing tasks — not just graphics.

### Why Use CUDA / GPU Computing?
| CPU | GPU |
|---|---|
| Few cores (4–64) | Thousands of cores (thousands to tens of thousands) |
| Designed for complex, serial tasks | Designed for simple, massively parallel tasks |
| High clock speed | Lower clock speed per core |
| Low latency | High throughput |
| General purpose | Ideal for data-parallel workloads |

**Data-parallel workload:** When you need to do the SAME operation on thousands/millions of independent data elements — like adding two vectors element by element, or computing each pixel of an image. Each element can be processed by a separate GPU core simultaneously.

### CPU vs GPU Architecture:
- **CPU:** Small number of powerful cores (4–16 for laptops), large cache, designed to minimize latency for complex branching logic
- **GPU:** Thousands of simpler cores (e.g., RTX 3080 has 8704 CUDA cores), small cache per core, designed to maximize throughput for simple repetitive operations

### CUDA Programming Model — Key Concepts:

#### Host vs Device:
- **Host:** The CPU and its RAM (regular computer memory) — `int *a = new int[n]`
- **Device:** The GPU and its VRAM (GPU memory) — `int *d_a; cudaMalloc(&d_a, ...)`
- Data must be explicitly **copied** from host to device before computation and back after

#### Kernel:
A CUDA **kernel** is a function that runs on the GPU. Defined with `__global__`:
```cpp
__global__ void add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```
When you "launch" a kernel, thousands of GPU threads execute it simultaneously.

#### Thread Hierarchy:
CUDA organizes threads in a two-level hierarchy:
```
Grid
 └── Block 0   Block 1   Block 2   ...
      ├── Thread 0
      ├── Thread 1
      ├── Thread 2
      └── ...
```
- **Thread:** The smallest execution unit. Each thread runs the kernel independently.
- **Block (Thread Block):** A group of threads that can share fast "shared memory" and synchronize with each other. Max 1024 threads per block.
- **Grid:** A collection of blocks. Together, Grid = all threads launched for one kernel call.

#### Thread Indexing (1D — for vectors):
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x` — which block is this? (0, 1, 2, ...)
- `blockDim.x` — how many threads per block? (e.g., 256)
- `threadIdx.x` — which thread within the block? (0–255)
- Together: gives each thread a unique global index `i`

Example with 256 threads/block:
- Thread 0 of Block 0: i = 0×256 + 0 = 0
- Thread 1 of Block 0: i = 0×256 + 1 = 1
- Thread 0 of Block 1: i = 1×256 + 0 = 256
- Thread 1 of Block 1: i = 1×256 + 1 = 257

#### Thread Indexing (2D — for matrices):
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```
For a matrix, each thread computes one cell `(row, col)` of the result.

#### Key CUDA API Functions:
| Function | Purpose |
|---|---|
| `cudaMalloc(&ptr, size)` | Allocate memory on GPU |
| `cudaMemcpy(dst, src, size, direction)` | Copy data between CPU and GPU |
| `cudaFree(ptr)` | Free GPU memory |
| `cudaDeviceSynchronize()` | Wait for all GPU threads to finish |
| `kernel<<<blocks, threads>>>(args)` | Launch kernel with specified grid/block configuration |

#### Memory Transfer Directions:
- `cudaMemcpyHostToDevice` — CPU RAM → GPU VRAM (before computation)
- `cudaMemcpyDeviceToHost` — GPU VRAM → CPU RAM (after computation)

---

## Program A: Vector Addition

### Objective
Add two vectors A and B element-wise to produce C: `C[i] = A[i] + B[i]` for all i. This is trivially parallel because each element of C depends only on A[i] and B[i] — no dependency between different i values.

### Step-by-Step Code Walkthrough

#### CUDA Kernel
```cpp
__global__ void add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```
- `__global__` marks this function as a CUDA kernel — called from CPU, runs on GPU
- Each GPU thread computes exactly ONE element: `c[i] = a[i] + b[i]`
- The `if (i < n)` check prevents out-of-bounds access when n is not a multiple of 256

#### Main Function — Sequential Part
```cpp
int *a = new int[n];
int *b = new int[n];
int *c = new int[n];

// ... input a and b ...

// Sequential addition (CPU)
auto start = high_resolution_clock::now();
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];          // One element at a time
}
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << "Sequential Time: " << duration.count() << " microseconds\n";
```
Standard CPU loop — one element processed per clock cycle.

#### Main Function — CUDA Part
```cpp
// Step 1: Allocate GPU memory
int *d_a, *d_b, *d_c;
cudaMalloc(&d_a, n * sizeof(int));    // Allocate n integers on GPU
cudaMalloc(&d_b, n * sizeof(int));
cudaMalloc(&d_c, n * sizeof(int));

// Step 2: Copy data from CPU to GPU
cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

// Step 3: Configure and launch kernel
int threads = 256;                          // 256 threads per block
int blocks = (n + threads - 1) / threads;  // Enough blocks to cover all n elements

start = high_resolution_clock::now();
add<<<blocks, threads>>>(d_a, d_b, d_c, n);   // Launch kernel
cudaDeviceSynchronize();                        // Wait for GPU to finish
stop = high_resolution_clock::now();

duration = duration_cast<microseconds>(stop - start);
cout << "Parallel CUDA Time: " << duration.count() << " microseconds\n";

// Step 4: Copy result from GPU to CPU
cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

// Step 5: Free GPU memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

**Why `blocks = (n + threads - 1) / threads`?**
This is ceiling division. If n=10 and threads=4, we need ⌈10/4⌉ = 3 blocks (12 threads total). The extra 2 threads check `if (i < n)` and do nothing. This ensures every element is covered.

---

## Program B: Matrix Multiplication

### Objective
Multiply two N×N matrices A and B to produce C: `C[i][j] = Σ A[i][k] × B[k][j]` for all valid i, j. Each cell of C can be computed independently — perfect for parallelization.

### Step-by-Step Code Walkthrough

#### CUDA Kernel (2D)
```cpp
__global__ void multiply(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];   // Dot product of row and column
        }
        C[row * N + col] = sum;
    }
}
```
- Each thread is responsible for exactly ONE cell (row, col) of the result matrix C
- It computes the dot product of row `row` of A with column `col` of B
- Matrices are stored as flat 1D arrays: `A[row][col]` → `A[row * N + col]`
- 2D thread indexing: `row` from y-dimension, `col` from x-dimension

#### 2D Grid Configuration
```cpp
dim3 threads(16, 16);               // 16×16 = 256 threads per block (2D block)
dim3 blocks((N + 15) / 16, (N + 15) / 16);  // Enough 2D blocks to cover N×N matrix

multiply<<<blocks, threads>>>(d_A, d_B, d_C, N);
```
- `dim3` is a 3D vector (x, y, z). We use x and y for the 2D thread grid.
- 16×16 = 256 threads per block (within the 1024 max limit)
- The 2D block grid covers the entire N×N output matrix

#### Sequential Multiplication (CPU)
```cpp
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```
Classic O(N³) triple loop. For large matrices, this is very slow.

---

## How to Compile and Run

### Requirements:
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (`nvcc` compiler)
- Google Colab provides free GPU access (Runtime → Change runtime type → GPU)

### Compilation:
```bash
nvcc vector_add.cu -o vector_add
./vector_add

nvcc matrix_mul.cu -o matrix_mul
./matrix_mul
```

### In Google Colab (from the notebook):
The CUDA code is written as string literals and compiled using magic commands:
```python
%%writefile add.cu
// ... CUDA code ...
!nvcc add.cu -o add && ./add
```

---

## Output Explanation

### Vector Addition Output (n=10):
```
Sequential Time: 0 microseconds
Parallel CUDA Time: 3 microseconds

Result Vector: [sum of each pair]
```
For tiny n=10, sequential is actually faster (or same) because GPU kernel launch overhead (~2–5 μs) dominates. For large n (millions), GPU would be thousands of times faster.

### Matrix Multiplication Output (2×2 matrix):
```
Sequential Time: 0 microseconds
Parallel CUDA Time: 6 microseconds

Result Matrix:
[correct product values]
```
Similarly, for small matrices the overhead exceeds the gain. For large matrices (N=1000+), GPU multiplication would be hundreds of times faster.

**Why does the tiny example show CUDA slower?**
Every CUDA program has fixed overhead:
1. `cudaMalloc` — allocate GPU memory: ~1–10 μs
2. `cudaMemcpy` (host→device) — transfer data: depends on size
3. Kernel launch — schedule threads: ~2–5 μs
4. `cudaDeviceSynchronize` — wait for completion: ~1 μs
5. `cudaMemcpy` (device→host) — get results back: depends on size

For 10 integers, this overhead is enormous relative to the 10 additions done. For 10 million integers, the 10M parallel additions far outweigh the constant overhead.

---

## Conclusion

We implemented CUDA programs for vector addition and matrix multiplication, demonstrating GPU parallel computing. The GPU programs launch thousands of threads simultaneously, each computing one element independently.

Key takeaways:
- `__global__` marks a CUDA kernel — it runs on the GPU but is launched from the CPU
- Threads use `blockIdx` and `threadIdx` to compute their unique element index
- Data must be explicitly transferred between CPU (host) and GPU (device) memory
- For small data, CUDA overhead dominates and CPU is faster; for large data (millions of elements), GPU is dramatically faster
- Matrix multiplication is perfect for GPU — each output cell is independent and computed by its own thread

---

## Viva Questions & Answers

**Q1: What is CUDA and why is it used?**
> CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. It lets programmers write C/C++ code that runs on GPU's thousands of cores simultaneously. It is used for data-parallel tasks like image processing, machine learning, scientific simulations, and physics — anywhere the same operation must be applied to many independent data elements at once.

**Q2: What is the difference between a CPU and a GPU?**
> A CPU has few (4–64) but powerful cores with large caches, designed for complex sequential tasks with branching. A GPU has thousands of simpler cores (e.g., 8704 for RTX 3080), designed for throughput — running thousands of simple operations simultaneously. CPU minimizes latency for any single task; GPU maximizes throughput for many parallel tasks.

**Q3: What is a CUDA kernel?**
> A kernel is a function marked with `__global__` that runs on the GPU. It is launched from CPU code with the `<<<blocks, threads>>>` syntax. When launched, thousands of GPU threads each execute the same kernel function but with different thread IDs, allowing each to work on a different data element.

**Q4: Explain `int i = blockIdx.x * blockDim.x + threadIdx.x`.**
> This computes a unique global thread ID for 1D indexing. `blockIdx.x` is which block (0, 1, 2...). `blockDim.x` is threads per block (e.g., 256). `threadIdx.x` is thread index within the block (0–255). Multiplying block index by block size and adding thread index gives a unique ID for every thread across all blocks.

**Q5: What is `cudaMalloc` and why is it needed?**
> `cudaMalloc(&ptr, size)` allocates memory on the GPU (device memory / VRAM). It is needed because GPU and CPU have separate physical memory spaces. Data in CPU RAM (`int *a = new int[n]`) cannot be directly accessed by GPU kernels. You must allocate separate space on the GPU and copy data there.

**Q6: What is `cudaMemcpy` and what are its directions?**
> `cudaMemcpy(destination, source, size, direction)` copies data between CPU and GPU memory. Directions:
> - `cudaMemcpyHostToDevice`: CPU RAM → GPU VRAM (before kernel — to feed input data)
> - `cudaMemcpyDeviceToHost`: GPU VRAM → CPU RAM (after kernel — to retrieve results)

**Q7: What is `cudaDeviceSynchronize()` and why is it used before measuring time?**
> GPU kernels execute asynchronously — the CPU launches the kernel and immediately continues without waiting for it to finish. `cudaDeviceSynchronize()` makes the CPU wait until all GPU threads complete. We call it before recording the stop time, otherwise we'd measure only the kernel launch time (microseconds) not the actual computation time.

**Q8: What is `__global__` in CUDA? What other qualifiers exist?**
> `__global__`: Called from CPU, runs on GPU. This is the kernel qualifier. Other qualifiers:
> - `__device__`: Called from GPU only, runs on GPU (helper functions within kernels)
> - `__host__`: Called from CPU, runs on CPU (the default — same as regular C++ functions)
> A function can be both `__host__ __device__` to work on both.

**Q9: Why do we use 2D thread blocks (dim3 threads(16,16)) for matrix multiplication?**
> A matrix is a 2D structure. Using 2D thread organization (x for columns, y for rows) makes thread-to-element mapping intuitive: thread(row, col) computes result matrix cell C[row][col]. While 1D indexing could work mathematically, 2D blocks match the 2D structure of the problem.

**Q10: What is the maximum number of threads per block in CUDA?**
> 1024 threads per block (hardware limit). Our matrix multiplication uses 16×16 = 256 threads per block (well within limit). Our vector addition uses 256 threads per block. We could use up to 32×32 = 1024 for matrices, but 16×16 is common and well-balanced.

**Q11: Why is vector addition ideal for GPU parallelism?**
> Each output element `c[i] = a[i] + b[i]` depends ONLY on `a[i]` and `b[i]` — no dependency on any other index. This means n elements can all be computed simultaneously by n threads with zero synchronization. Perfect data parallelism with no overhead. It is an embarrassingly parallel problem.

**Q12: What does `cudaFree()` do and why is it important?**
> `cudaFree(ptr)` releases GPU memory previously allocated with `cudaMalloc`. GPU memory (VRAM) is limited (e.g., 8GB). If you don't free it, it stays occupied for the entire program. For programs that process many datasets, forgetting `cudaFree` causes GPU memory leaks, eventually crashing the program.

**Q13: How is matrix multiplication done in parallel with CUDA?**
> Each thread computes one cell C[row][col] of the result matrix. Thread (row, col) iterates over all k from 0 to N-1, computing the dot product: `sum += A[row][k] * B[k][col]`. Since different (row, col) threads access different rows of A and different columns of B, there are no write conflicts. All N×N cells are computed simultaneously.

**Q14: Why is CUDA slower than CPU for small input sizes?**
> CUDA has fixed overhead: GPU memory allocation (~μs), data transfer CPU→GPU (~μs per MB), kernel launch (~μs), synchronization, and data transfer GPU→CPU. For tiny inputs (n=10 integers), these overheads are much larger than the time saved by parallel computation. The break-even point is typically n > 10,000–100,000 where the parallel speedup outweighs overhead.
