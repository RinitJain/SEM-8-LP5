// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use
// existing algorithms and measure the performance of sequential and parallel algorithms.

#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

// ---------------- SEQUENTIAL BUBBLE SORT ----------------
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// ---------------- PARALLEL BUBBLE SORT ----------------
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

// ---------------- MERGE FUNCTION ----------------
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

// ---------------- SEQUENTIAL MERGE SORT ----------------
void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// ---------------- PARALLEL MERGE SORT ----------------
void parallelMergeSort(vector<int>& arr, int left, int right) {  // add int depth = 0 if want to limit depth of parallelism

    if (left < right) {

        int mid = (left + right) / 2;
        // if (depth < 4) { // optional depth check to limit parallelism, cause generally system has limited thread so 4 cores is like 2^4 = 16 threads, so we can limit to that
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, left, mid);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }
        // else {
        //     sequentialMergeSort(arr, left, mid);
        //     sequentialMergeSort(arr, mid + 1, right);    
        // }    

        merge(arr, left, mid, right);
    }
}

int main() {

    int SIZE;

    cout << "Enter array size: ";
    cin >> SIZE;

    vector<int> original(SIZE);

    // Random array generation
    for (int i = 0; i < SIZE; i++) {
        original[i] = rand() % 1000; // OR cin >> original[i] if want to input manually
    }

    vector<int> a1 = original;
    vector<int> a2 = original;
    vector<int> a3 = original;
    vector<int> a4 = original;

    double start, end;

    // ---------------- SEQUENTIAL BUBBLE SORT ----------------
    start = omp_get_wtime();
    sequentialBubbleSort(a1);
    end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- PARALLEL BUBBLE SORT ----------------
    start = omp_get_wtime();
    parallelBubbleSort(a2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- SEQUENTIAL MERGE SORT ----------------
    start = omp_get_wtime();
    sequentialMergeSort(a3, 0, SIZE - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- PARALLEL MERGE SORT ----------------
    start = omp_get_wtime();
    parallelMergeSort(a4, 0, SIZE - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << (end - start) * 1000 << " ms" << endl;

    return 0;  
}

/*

COMPILATION:
g++ -fopenmp hpc2.cpp -o hpc2

SET THREADS:
export OMP_NUM_THREADS=4

RUN:
./hpc2

*/