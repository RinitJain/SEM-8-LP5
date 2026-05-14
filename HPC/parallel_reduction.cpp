// Implement Min, Max, Sum and Average operations using Parallel Reduction.
// Measure the performance of sequential and parallel algorithms.

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// ---------------- SEQUENTIAL MIN ----------------
int sequentialMin(vector<int>& arr) {
    int minval = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < minval)
            minval = arr[i];
    }
    return minval;
}

// ---------------- PARALLEL MIN ----------------
int parallelMin(vector<int>& arr) {
    int minval = arr[0];
    #pragma omp parallel for reduction(min:minval)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < minval)
            minval = arr[i];
    }
    return minval;
}

// ---------------- SEQUENTIAL MAX ----------------
int sequentialMax(vector<int>& arr) {
    int maxval = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > maxval)
            maxval = arr[i];
    }
    return maxval;
}

// ---------------- PARALLEL MAX ----------------
int parallelMax(vector<int>& arr) {
    int maxval = arr[0];
    #pragma omp parallel for reduction(max:maxval)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > maxval)
            maxval = arr[i];
    }
    return maxval;
}

// ---------------- SEQUENTIAL SUM ----------------
int sequentialSum(vector<int>& arr) {
    int sum = 0;
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

// ---------------- PARALLEL SUM ----------------
int parallelSum(vector<int>& arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

// ---------------- SEQUENTIAL AVERAGE ----------------
double sequentialAverage(vector<int>& arr) {
    return (double)sequentialSum(arr) / arr.size();
}

// ---------------- PARALLEL AVERAGE ----------------
double parallelAverage(vector<int>& arr) {
    return (double)parallelSum(arr) / arr.size();
}

int main() {

    int n;

    cout << "Enter array size: ";
    cin >> n;

    vector<int> arr(n);

    // Generate random array
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;  // OR cin >> arr[i] if want to input manually
    }

    double start, end;

    // ---------------- MIN ----------------
    start = omp_get_wtime();
    int seqMin = sequentialMin(arr);
    end = omp_get_wtime();
    cout << "\nSequential Min: " << seqMin << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    start = omp_get_wtime();
    int parMin = parallelMin(arr);
    end = omp_get_wtime();
    cout << "\nParallel Min: " << parMin << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- MAX ----------------
    start = omp_get_wtime();
    int seqMax = sequentialMax(arr);
    end = omp_get_wtime();
    cout << "\nSequential Max: " << seqMax << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    start = omp_get_wtime();
    int parMax = parallelMax(arr);
    end = omp_get_wtime();
    cout << "\nParallel Max: " << parMax << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- SUM ----------------
    start = omp_get_wtime();
    int seqSum = sequentialSum(arr);
    end = omp_get_wtime();
    cout << "\nSequential Sum: " << seqSum << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    start = omp_get_wtime();
    int parSum = parallelSum(arr);
    end = omp_get_wtime();
    cout << "\nParallel Sum: " << parSum << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- AVERAGE ----------------
    start = omp_get_wtime();
    double seqAvg = sequentialAverage(arr);
    end = omp_get_wtime();
    cout << "\nSequential Average: " << seqAvg << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    start = omp_get_wtime();
    double parAvg = parallelAverage(arr);
    end = omp_get_wtime();
    cout << "\nParallel Average: " << parAvg << endl;
    cout << "Time: " << (end - start) * 1000 << " ms" << endl;

    return 0;
}

/*

COMPILATION:
g++ -fopenmp hpc3.cpp -o hpc3

SET THREADS:
export OMP_NUM_THREADS=4

RUN:
./hpc3

*/