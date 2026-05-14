# HPC Practical 3: Parallel Bubble Sort and Merge Sort using OpenMP

## Objective

Implement parallel versions of two classical sorting algorithms — **Bubble Sort** and **Merge Sort** — using OpenMP. Compare execution time of sequential vs parallel for both algorithms, and understand how different algorithms can be parallelized in different ways.

---

## Theory / Background

### What is Sorting?
Sorting arranges elements in a specific order (ascending or descending). It is one of the most fundamental operations in computer science and a key benchmark for comparing algorithms.

### What is Bubble Sort?
Bubble Sort repeatedly compares adjacent elements and swaps them if they are in the wrong order. After each full pass, the largest unsorted element "bubbles up" to its correct position.

**Sequential Algorithm:**
```
for i = 0 to n-1:
    for j = 0 to n-i-2:
        if arr[j] > arr[j+1]: swap(arr[j], arr[j+1])
```

**Time Complexity:**
- Best case: O(n) — already sorted
- Worst/Average: O(n²)
- Space: O(1) — in-place

### Why Can't We Simply Parallelize the Inner Loop of Bubble Sort?
Look at a standard bubble sort's inner loop: it compares pairs (0,1), (1,2), (2,3), etc. The problem is that these pairs are NOT independent — if swap(0,1) moves a new value to position 1, then comparing (1,2) depends on that result. We cannot parallelize sequential adjacent swaps safely.

### Odd-Even Transposition Sort (Parallel Bubble Sort):
The solution is the **odd-even transposition** approach:
- **Even phase:** Compare pairs at positions (0,1), (2,3), (4,5), (6,7)... — these pairs are independent! Swapping (0,1) doesn't affect (2,3).
- **Odd phase:** Compare pairs at positions (1,2), (3,4), (5,6), (7,8)... — these are also independent!

By alternating even and odd phases, all comparisons within a phase can be done in parallel. After n rounds, the array is sorted.

```cpp
for (int i = 0; i < n; i++) {
    // Phase: even if i is even, odd if i is odd
    // Parallel: compare non-overlapping pairs
    #pragma omp parallel for
    for (int j = (i % 2); j < n - 1; j += 2) {
        if (arr[j] > arr[j + 1]) swap(arr[j], arr[j + 1]);
    }
}
```

### What is Merge Sort?
Merge Sort is a divide-and-conquer algorithm:
1. **Divide:** Split the array in half recursively until subarrays have size 1
2. **Conquer:** Subarrays of size 1 are trivially sorted
3. **Combine:** Merge pairs of sorted subarrays into a larger sorted array

**Time Complexity:**
- All cases: O(n log n)
- Space: O(n) — needs auxiliary arrays for merging

Merge Sort is fundamentally more efficient than Bubble Sort for large arrays.

### Parallelizing Merge Sort:
Merge Sort divides the array into two halves and sorts them independently — the left half and right half don't share data. This is **naturally parallel**: both halves can be sorted by different threads simultaneously.

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    parallelMergeSort(arr, left, mid);      // Thread A sorts left half

    #pragma omp section
    parallelMergeSort(arr, mid + 1, right); // Thread B sorts right half
}
merge(arr, left, mid, right);               // Must be sequential
```

The `merge` step cannot easily be parallelized because it reads from two arrays and writes to one in a specific order.

### Key OpenMP Directives Used:
| Directive | Meaning |
|---|---|
| `#pragma omp parallel for` | Distributes loop iterations across threads |
| `#pragma omp parallel sections` | Defines a block where each `section` is run by a different thread |
| `#pragma omp section` | Marks an independent piece of work within a `parallel sections` block |
| `omp_get_wtime()` | Measures wall-clock time |

### Thread Explosion in Recursive Parallel:
Each call to `parallelMergeSort` creates new threads for its two recursive calls. With n = 1,000,000, the recursion depth is ~20, potentially creating 2^20 (over 1 million) threads! This is called **thread explosion** and exhausts system resources.

The fix (commented in the code) is to limit parallelism to a certain depth:
```cpp
if (depth < 4) {
    // parallelize
} else {
    // use sequential
}
```

This limits the tree to 2^4 = 16 parallel branches — manageable for a 4-core CPU.

---

## Code Structure

**File:** `parallel_sort.cpp`

Implements 4 functions: `sequentialBubbleSort`, `parallelBubbleSort`, `sequentialMergeSort`, `parallelMergeSort`, plus a `merge` helper.

---

## Step-by-Step Code Walkthrough

### Sequential Bubble Sort
```cpp
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {           // n-1 passes
        for (int j = 0; j < n - i - 1; j++) {   // Unsorted portion shrinks each pass
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```
Classic O(n²) bubble sort. After pass `i`, the last `i` elements are in their correct positions. So the inner loop runs `n-i-1` times.

### Parallel Bubble Sort (Odd-Even Transposition)
```cpp
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = (i % 2); j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```
- Outer loop `i` alternates phase: when `i=0 (even)` → `j` starts at 0; when `i=1 (odd)` → `j` starts at 1
- `j += 2` skips every other position — only non-overlapping pairs are compared
- `#pragma omp parallel for` distributes the non-overlapping pairs across threads safely
- Requires `n` outer iterations (not `n-1` like sequential) to guarantee full sortedness

**Why are pairs non-overlapping?**
- Even phase: pairs (0,1), (2,3), (4,5)... — indices jump by 2
- If thread A swaps (0,1) and thread B swaps (2,3), they touch different array positions — no conflict

### Merge Helper Function
```cpp
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;    // Size of left subarray
    int n2 = right - mid;       // Size of right subarray

    vector<int> L(n1), R(n2);   // Temporary arrays

    // Copy data to temp arrays
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    // Merge: pick the smaller element from L or R
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else               arr[k++] = R[j++];
    }

    // Copy remaining elements
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}
```
Creates temp copies of left and right halves. Uses two pointers (`i` for L, `j` for R) to merge in sorted order. Always picks the smaller of the two current elements.

### Sequential Merge Sort
```cpp
void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {                          // Base case: single element
        int mid = (left + right) / 2;
        sequentialMergeSort(arr, left, mid);     // Sort left half
        sequentialMergeSort(arr, mid + 1, right); // Sort right half
        merge(arr, left, mid, right);             // Merge both halves
    }
}
```
Classic recursive divide-and-conquer. Recursion stops when `left >= right` (subarray has 0 or 1 elements).

### Parallel Merge Sort
```cpp
void parallelMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, left, mid);       // Thread 1 sorts left

            #pragma omp section
            parallelMergeSort(arr, mid + 1, right);  // Thread 2 sorts right
        }

        merge(arr, left, mid, right);   // Sequential merge after both halves done
    }
}
```
- `#pragma omp parallel sections` creates two threads
- Each `#pragma omp section` is an independent work unit assigned to one thread
- Both halves are sorted simultaneously (true parallelism)
- `merge` happens after BOTH sections complete (implicit barrier at end of `parallel sections`)

### Main Function
```cpp
int main() {
    int SIZE;
    cin >> SIZE;

    vector<int> original(SIZE);
    for (int i = 0; i < SIZE; i++) {
        original[i] = rand() % 1000;
    }

    // Make 4 copies so each sort works on the same original data
    vector<int> a1 = original;  // Sequential Bubble Sort
    vector<int> a2 = original;  // Parallel Bubble Sort
    vector<int> a3 = original;  // Sequential Merge Sort
    vector<int> a4 = original;  // Parallel Merge Sort

    double start, end;

    start = omp_get_wtime();
    sequentialBubbleSort(a1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << (end - start) * 1000 << " ms" << endl;

    // ... similarly for all 4 algorithms
}
```
Creates 4 copies of the same array so each algorithm starts with identical unsorted data for fair comparison.

---

## How to Compile and Run

```bash
g++ -fopenmp parallel_sort.cpp -o hpc2
export OMP_NUM_THREADS=4
./hpc2
```

**Sample Input:**
```
Enter array size: 10000
```

---

## Output Explanation

For array size 10,000:

```
Sequential Bubble Sort Time: 487.3 ms
Parallel Bubble Sort Time:   198.2 ms
Sequential Merge Sort Time:  2.4 ms
Parallel Merge Sort Time:    1.8 ms
```

**Key Observations:**

1. **Bubble Sort:** Parallel is significantly faster than sequential (by ~2–3× with 4 threads) because many comparisons happen simultaneously in each phase.

2. **Merge Sort:** Both sequential and parallel are extremely fast for n=10,000 (O(n log n) is very efficient). Parallel merge sort shows modest improvement because creating threads has overhead that is large relative to the already-fast sequential merge sort.

3. **Bubble Sort vs Merge Sort:** Merge Sort is orders of magnitude faster than Bubble Sort for large arrays. Sequential Merge Sort (2.4 ms) is faster than Parallel Bubble Sort (198 ms) for n=10,000! This shows that **a better algorithm beats a worse algorithm even when parallelized**.

4. **Parallel Merge Sort Limitation:** Thread explosion in deep recursion can actually make parallel merge sort slower for large arrays without depth limiting.

---

## Conclusion

We implemented 4 sorting variants: sequential and parallel Bubble Sort (using odd-even transposition) and sequential and parallel Merge Sort (using parallel sections for recursive calls).

Key takeaways:
- Parallel Bubble Sort uses odd-even transposition to make comparisons independent and parallelizable
- Parallel Merge Sort naturally parallelizes by sorting left and right halves simultaneously
- Algorithm choice matters more than parallelism — sequential Merge Sort beats parallel Bubble Sort
- Thread explosion in recursive parallel algorithms can worsen performance; depth limiting is needed
- The merge step remains sequential — a fundamental limitation of merge sort parallelization

---

## Viva Questions & Answers

**Q1: What is Bubble Sort and what is its time complexity?**
> Bubble Sort repeatedly compares and swaps adjacent elements. In each pass, the largest unsorted element moves ("bubbles") to its correct position. Time complexity: O(n²) in average and worst case, O(n) best case (already sorted). Space complexity: O(1) — in-place sort.

**Q2: Why can't you simply parallelize the inner loop of sequential Bubble Sort?**
> In sequential bubble sort, the inner loop compares (0,1), then (1,2), then (2,3)... These are sequential dependencies — comparing (1,2) uses the result of potentially swapping (0,1). If threads run simultaneously, thread B comparing (1,2) might read a stale value before thread A finishes swapping (0,1). This causes a race condition.

**Q3: How does Odd-Even Transposition Sort parallelize Bubble Sort?**
> In the even phase, we compare pairs (0,1), (2,3), (4,5)... — these pairs don't share any indices, so all comparisons are independent and can run in parallel. In the odd phase, we compare (1,2), (3,4), (5,6)... — again all independent. By alternating these phases for n rounds, the array gets sorted while each phase is fully parallelizable.

**Q4: What is `#pragma omp parallel sections` and how is it different from `#pragma omp parallel for`?**
> `parallel sections` divides work into distinct, independent blocks (sections), each assigned to a different thread. `parallel for` distributes iterations of a loop. Use `parallel sections` when you have different blocks of work (like sorting left and right halves). Use `parallel for` when you have a loop with independent iterations.

**Q5: What is Merge Sort and what is its time complexity?**
> Merge Sort is a divide-and-conquer algorithm. It recursively splits the array in half until subarrays have 1 element (base case), then merges sorted pairs back together. Time complexity: O(n log n) for all cases. Space complexity: O(n) — needs extra memory for temporary arrays during merging.

**Q6: Why is the merge step kept sequential in parallel merge sort?**
> The merge step reads from two subarrays and writes to one, in a specific interleaved order that depends on comparisons as it goes. Each element written depends on the previous comparison result. There is no simple way to divide this work among threads without creating dependencies or complex synchronization. So merge remains sequential.

**Q7: What is thread explosion in recursive parallel merge sort?**
> Each call to `parallelMergeSort` spawns 2 new threads. Each of those spawns 2 more, etc. For an array of n elements, recursion depth is log₂(n). This creates up to 2^(log n) = n threads! For n=1,000,000, that's over a million threads — far more than the CPU can handle, causing massive overhead.

**Q8: How do you prevent thread explosion?**
> Limit the depth at which parallelism is applied. If `depth < 4`, use `parallel sections`; otherwise use sequential recursive calls. This limits the parallel tree to 2^4 = 16 branches. Beyond depth 4, sequential merge sort is used (which is efficient at small sizes anyway).

**Q9: Why is Merge Sort always better than Bubble Sort for large n?**
> Bubble Sort is O(n²) — for n=1,000,000, that's 10^12 operations. Merge Sort is O(n log n) — for n=1,000,000, that's ~20,000,000 operations. Even with 1000 parallel threads, parallel Bubble Sort is still O(n²/threads), while sequential Merge Sort is O(n log n). At large n, Merge Sort wins every time.

**Q10: In main(), why are 4 separate arrays created?**
> Each sort modifies the array in-place. If all 4 algorithms used the same array, the second sort would start with an already-sorted array (best case for Bubble Sort), giving unfair timing comparisons. By copying the original random array into 4 separate copies (a1, a2, a3, a4), each algorithm starts with the same unsorted data.

**Q11: What does `j += 2` do in the parallel bubble sort inner loop?**
> `j += 2` makes the loop skip every other position: j takes values 0, 2, 4, 6... (even phase) or 1, 3, 5, 7... (odd phase). This ensures the pairs being compared never share an index — pair (0,1) and pair (2,3) have no common element. Non-overlapping pairs can be swapped in parallel without race conditions.

**Q12: What is the role of the `merge` function? Explain it line by line.**
> It merges two sorted subarrays (left half: indices `left` to `mid`, right half: `mid+1` to `right`) into one sorted array. Steps: (1) Copy both halves into temporary arrays L and R. (2) Compare elements at the front of L and R, place the smaller one into the original array, advance that pointer. (3) After one half is exhausted, copy remaining elements from the other half. The result is one sorted subarray from `left` to `right`.
