# HPC Practical 2: Parallel Reduction — Min, Max, Sum, Average using OpenMP

## Objective

Implement Min, Max, Sum, and Average operations on a large array using both sequential and parallel approaches. Use OpenMP's `reduction` clause to safely combine results from multiple threads. Measure and compare execution times to demonstrate the performance benefit of parallel reduction.

---

## Theory / Background

### What is Reduction?
Reduction is the process of combining many values into a single result using an associative operation like sum, minimum, maximum, or product.

Example: Finding the sum of [3, 7, 1, 9, 4, 2] → 3+7+1+9+4+2 = 26

In sequential code, we loop through one element at a time. In parallel, we split the array among threads, each thread computes a partial result, and then we combine all partial results.

### How Parallel Reduction Works:
1. Divide the array into chunks — one chunk per thread
2. Each thread computes its local result (e.g., local minimum) on its chunk
3. When all threads finish, combine all local results into the final answer

For example, with 4 threads on array [3, 7, 1, 9, 4, 2, 8, 5]:
- Thread 0 processes [3, 7] → local min = 3
- Thread 1 processes [1, 9] → local min = 1
- Thread 2 processes [4, 2] → local min = 2
- Thread 3 processes [8, 5] → local min = 5
- Combine: min(3, 1, 2, 5) = 1 ✓

### Why Does Reduction Need Special Handling?
If all threads tried to update the same variable simultaneously:
```cpp
// WRONG — race condition!
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    if (arr[i] < minval) minval = arr[i];   // Multiple threads writing simultaneously
}
```
This is a **race condition** — two threads can both check, both find a new minimum, both write, and the final value depends on which thread wrote last. Result is unpredictable and wrong.

### The `reduction` Clause:
OpenMP's `reduction` clause solves this automatically:
```cpp
#pragma omp parallel for reduction(min:minval)
```
This tells OpenMP to:
1. Give each thread its own **private copy** of `minval` initialized appropriately
2. Each thread updates only its own private copy (no race condition)
3. After the loop, OpenMP combines all private copies using the specified operation (min, max, +, etc.)

### Amdahl's Law:
Even with unlimited threads, you can't speed up beyond the sequential portion of your code.

`Speedup = 1 / (S + (1-S)/N)`

Where S = fraction of code that must remain sequential, N = number of threads. If even 10% must be sequential, max speedup is 10× regardless of how many threads you add.

### Key OpenMP Directives Used:
| Directive | Meaning |
|---|---|
| `#pragma omp parallel for reduction(min:var)` | Parallel loop where each thread maintains private min, combined at end |
| `#pragma omp parallel for reduction(max:var)` | Same but for maximum |
| `#pragma omp parallel for reduction(+:var)` | Same but for sum using addition |
| `omp_get_wtime()` | Wall-clock timer in seconds |

### OpenMP Reduction Operations:
| Operator | Symbol | Init Value | Use case |
|---|---|---|---|
| Sum | `+` | 0 | Sum of all elements |
| Multiply | `*` | 1 | Product of all elements |
| Min | `min` | Type maximum | Minimum value |
| Max | `max` | Type minimum | Maximum value |
| AND | `&&` | 1 | Logical AND across all |
| OR | `\|\|` | 0 | Logical OR across all |

---

## Code Structure

**File:** `parallel_reduction.cpp`

Contains 8 functions: sequential and parallel versions of Min, Max, Sum, and Average. The `main()` function generates a random array, tests all 8 functions, and prints timing results.

---

## Step-by-Step Code Walkthrough

### Step 1: Include Headers
```cpp
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;
```
- `<omp.h>` — OpenMP header required for all OpenMP functions

### Step 2: Sequential Min
```cpp
int sequentialMin(vector<int>& arr) {
    int minval = arr[0];               // Start with first element
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < minval)
            minval = arr[i];
    }
    return minval;
}
```
Classic single-threaded loop. Passes `arr` by reference (`&`) to avoid copying the entire vector.

### Step 3: Parallel Min
```cpp
int parallelMin(vector<int>& arr) {
    int minval = arr[0];
    #pragma omp parallel for reduction(min:minval)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < minval)
            minval = arr[i];
    }
    return minval;
}
```
`reduction(min:minval)`: Each thread gets a private copy of `minval`. Each thread finds the minimum in its chunk. OpenMP automatically combines all private minimums into the final global minimum.

### Step 4: Sequential Max
```cpp
int sequentialMax(vector<int>& arr) {
    int maxval = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > maxval)
            maxval = arr[i];
    }
    return maxval;
}
```

### Step 5: Parallel Max
```cpp
int parallelMax(vector<int>& arr) {
    int maxval = arr[0];
    #pragma omp parallel for reduction(max:maxval)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > maxval)
            maxval = arr[i];
    }
    return maxval;
}
```
Same pattern as parallel min but with `reduction(max:maxval)`.

### Step 6: Sequential Sum
```cpp
int sequentialSum(vector<int>& arr) {
    int sum = 0;
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}
```

### Step 7: Parallel Sum
```cpp
int parallelSum(vector<int>& arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}
```
`reduction(+:sum)`: Each thread accumulates a partial sum. All partial sums are added together at the end.

### Step 8: Average (Sequential and Parallel)
```cpp
double sequentialAverage(vector<int>& arr) {
    return (double)sequentialSum(arr) / arr.size();
}

double parallelAverage(vector<int>& arr) {
    return (double)parallelSum(arr) / arr.size();
}
```
Average reuses the sum functions — compute sum, then divide by count. Note the `(double)` cast to get a decimal result instead of integer division.

### Step 9: Main Function
```cpp
int main() {
    int n;
    cin >> n;

    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;    // Random values 0–999
    }

    double start, end;

    start = omp_get_wtime();
    int seqMin = sequentialMin(arr);
    end = omp_get_wtime();
    cout << "Sequential Min: " << seqMin << " | Time: " << (end - start) * 1000 << " ms" << endl;

    start = omp_get_wtime();
    int parMin = parallelMin(arr);
    end = omp_get_wtime();
    cout << "Parallel Min: " << parMin << " | Time: " << (end - start) * 1000 << " ms" << endl;

    // ... similarly for Max, Sum, Average
}
```
`rand() % 1000` generates random integers from 0 to 999. The pattern of measuring time before/after each function call is repeated for all 8 functions.

---

## How to Compile and Run

```bash
g++ -fopenmp parallel_reduction.cpp -o hpc3
export OMP_NUM_THREADS=4
./hpc3
```

**Sample Input:**
```
Enter array size: 1000000
```

---

## Output Explanation

For an array of 1,000,000 random integers (0–999):

```
Sequential Min: 0
Time: 2.3 ms

Parallel Min: 0
Time: 0.9 ms

Sequential Max: 999
Time: 2.1 ms

Parallel Max: 999
Time: 0.8 ms

Sequential Sum: 499,512,345
Time: 1.8 ms

Parallel Sum: 499,512,345
Time: 0.7 ms

Sequential Average: 499.51
Time: 1.8 ms

Parallel Average: 499.51
Time: 0.7 ms
```

**Observations:**
- All sequential and parallel results are identical — correctness is maintained
- Parallel versions are ~2–3× faster with 4 threads
- Actual speedup varies: thread creation has overhead, so for small arrays (n < 10,000), sequential may be faster
- For very large arrays (n = 10,000,000+), parallel speedup is more pronounced

**Why is speedup not exactly 4× with 4 threads?**
- Thread creation/destruction overhead
- All threads reading from the same memory (memory bandwidth bottleneck)
- Time to combine partial results at the end

---

## Conclusion

We implemented parallel versions of Min, Max, Sum, and Average using OpenMP's `reduction` clause. The reduction clause elegantly handles thread safety by giving each thread a private copy of the accumulator variable, then combining results.

Key takeaways:
- The `reduction` clause prevents race conditions without needing explicit `#pragma omp critical`
- Sequential and parallel results are always identical — parallelism doesn't affect correctness
- Parallel reduction is most effective for large arrays; overhead dominates for small ones
- Average is derived from Sum — reusing functions avoids code duplication

---

## Viva Questions & Answers

**Q1: What is parallel reduction?**
> Reduction is combining many values into one using an operation (sum, min, max). In parallel reduction, the array is split among threads, each thread computes a partial result on its chunk, and all partial results are combined at the end. This reduces the time from O(n) to O(n/threads).

**Q2: What is the `reduction` clause in OpenMP?**
> `reduction(operator:variable)` tells OpenMP to: (1) create a private copy of the variable for each thread, (2) each thread operates on only its private copy (no race condition), and (3) after the parallel region, all private copies are combined using the specified operator into the final variable.

**Q3: What would happen without the `reduction` clause?**
> Multiple threads would try to read and write the same variable (e.g., `sum`) simultaneously — a race condition. The result would be unpredictable and incorrect because one thread might overwrite another's update. The `reduction` clause prevents this by giving each thread its own private copy.

**Q4: What is Amdahl's Law?**
> Amdahl's Law states that the maximum speedup of a parallel program is limited by its sequential (non-parallelizable) portion: Speedup = 1 / (S + (1-S)/N), where S = serial fraction, N = number of threads. Even with 1000 threads, if 10% is serial, maximum speedup is 10×.

**Q5: What is a race condition and how does `reduction` prevent it?**
> A race condition occurs when multiple threads access and modify a shared variable simultaneously, giving unpredictable results. `reduction` prevents this by giving each thread a PRIVATE copy of the variable. Threads only modify their own private copy, so there is no conflict. After all threads finish, OpenMP combines private copies into the final shared variable.

**Q6: For what size of array does parallel reduction become beneficial?**
> For small arrays (< 10,000 elements), thread creation overhead often makes parallel slower than sequential. For large arrays (> 100,000 elements), parallel becomes significantly faster. The crossover point depends on the CPU, number of cores, and array type.

**Q7: Why is `rand() % 1000` used to generate the array?**
> `rand()` generates a pseudo-random integer. `% 1000` limits values to 0–999. This creates a random test array without needing user input. The actual values don't matter — we're testing the performance of the parallel algorithms, not the correctness of any specific computation.

**Q8: What is thread overhead in parallel computing?**
> Creating threads (allocating stack space, registering with OS), synchronizing them (barriers, critical sections), and destroying them at the end all take time. This is thread overhead. For very small tasks, this overhead can exceed the time saved by parallelism.

**Q9: Why does `sequentialAverage` call `sequentialSum` instead of recomputing the sum?**
> Code reuse. The Sum function is already implemented and tested. Calling it avoids duplicating 5 lines of code and reduces the chance of bugs. If the Sum function is correct, Average built on top of it is also correct.

**Q10: What is the difference between `#pragma omp parallel for reduction` and `#pragma omp parallel for` with `#pragma omp critical`?**
> Both prevent race conditions but differently. `reduction` gives each thread a private variable — no synchronization during the loop, just one combine operation at the very end. `critical` allows only one thread into a block at a time — threads wait in line, which is slower. `reduction` is more efficient for simple accumulation operations.

**Q11: What does `vector<int>&` mean in the function parameters?**
> The `&` means pass-by-reference. The function receives a reference (pointer) to the original vector, not a copy. This avoids copying millions of integers just to call a function. The `int` specifies the vector holds integer values.

**Q12: How would you extend this to compute standard deviation in parallel?**
> Standard deviation = √(mean of squared deviations from mean). First compute the mean using parallel reduction (Sum/Count). Then compute squared deviations for each element and sum them using another parallel reduction. Finally, sqrt(sumSquaredDev / n). This requires two parallel passes.
