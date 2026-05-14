#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <omp.h>
using namespace std;

class ParallelSorting {
public:
  vector<int> arr;

  ParallelSorting(vector<int> v) { arr = v; }

  void print_array() {
    for (int i = 0; i < arr.size(); i++) {
      cout << arr[i] << " ";
    }
    cout << endl;
  }

  void merge(int left, int mid, int right) {
    int i = left, j = mid + 1;
    vector<int> temp;

    while (i <= mid && j <= right) {
      if (arr[i] <= arr[j])
        temp.push_back(arr[i++]);
      else
        temp.push_back(arr[j++]);
    }

    while (i <= mid)
      temp.push_back(arr[i++]);
    while (j <= right)
      temp.push_back(arr[j++]);

    for (int k = 0; k < temp.size(); k++) {
      arr[left + k] = temp[k];
    }
  }

  void serial_bubblesort() {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
      for (int j = 0; j < n - i - 1; j++) {
        if (arr[j] > arr[j + 1])
          swap(arr[j], arr[j + 1]);
      }
    }
  }

  void serial_mergesort(int left, int right) {
    if (left < right) {
      int mid = left + (right - left) / 2;

      serial_mergesort(left, mid);
      serial_mergesort(mid + 1, right);

      merge(left, mid, right);
    }
  }

  void parallel_bubblesort() {
    int n = arr.size();
    bool sorted = false;

    // odd-even transposition sort pattern
    while (!sorted) {
      bool local_sorted = true;

// even-indexed passes (0,2,4,...)
#pragma omp parallel for reduction(&& : local_sorted)
      for (int i = 0; i < n - 1; i += 2) {
        if (arr[i] > arr[i + 1]) {
          swap(arr[i], arr[i + 1]);
          local_sorted = false;
        }
      }

// odd-indexed passes (1,3,5,...)
#pragma omp parallel for reduction(&& : local_sorted)
      for (int i = 1; i < n - 1; i += 2) {
        if (arr[i] > arr[i + 1]) {
          swap(arr[i], arr[i + 1]);
          local_sorted = false;
        }
      }

      // stop when no swaps were made in both passes
      sorted = local_sorted;
    }
  }

  void parallel_mergesort(int left, int right, int depth = 0) {
    if (left < right) {
      int mid = left + (right - left) / 2;

      // Without a depth limit, each recursive call would spawn new parallel
      // tasks. Since merge sort is divide-and-conquer, this leads to
      // exponential growth in threads (e.g., depth 0: 1 thread, depth 1: 2,
      // depth 2: 4, ..., depth 10: 1024).

      // At some point (deep in the recursion), the subarrays are so small that
      // parallelizing their sort takes more time than sorting them sequentially
      // due to thread management overhead. Hence we parallelize for depth <= 4.
      if (depth <= 4) {
#pragma omp parallel sections
        {
#pragma omp section
          parallel_mergesort(left, mid, depth + 1);

#pragma omp section
          parallel_mergesort(mid + 1, right, depth + 1);
        }
      } else {
        parallel_mergesort(left, mid, depth + 1);
        parallel_mergesort(mid + 1, right, depth + 1);
      }

      merge(left, mid, right);
    }
  }
};

int main() {
  vector<int> arr = {5, 3, 10, 20, 0, -1};
  ParallelSorting ps1(arr);

  cout << "Original array....." << endl;
  ps1.print_array();

  cout << "Bubble Sort..." << endl;
  ps1.parallel_bubblesort();
  ps1.print_array();

  arr = {5, 3, 10, 20, 0, -1};
  ParallelSorting ps2(arr);
  cout << "Merge Sort...." << endl;
  ps2.parallel_mergesort(0, arr.size() - 1, 0);
  ps2.print_array();

  return 0;
}