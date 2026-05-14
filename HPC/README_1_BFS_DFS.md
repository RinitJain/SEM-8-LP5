# HPC Practical 1: Parallel Breadth First Search (BFS) and Depth First Search (DFS) using OpenMP

## Objective

Implement both BFS and DFS graph traversal algorithms in sequential and parallel versions using OpenMP. Measure and compare the execution time of sequential vs parallel implementations to understand the performance benefits of parallelism.

---

## Theory / Background

### What is OpenMP?
OpenMP (Open Multi-Processing) is a set of compiler directives, library routines, and environment variables that enables **shared-memory parallel programming** in C, C++, and Fortran. It allows you to parallelize loops and sections of code by simply adding `#pragma omp` directives — without writing thread management code manually.

**Key idea:** OpenMP creates a team of threads that run in parallel on the CPU's multiple cores. All threads share the same memory space.

### Why Use OpenMP?
- **Easy to use:** Add pragma directives to existing sequential code
- **Portable:** Works on Linux, Windows, macOS with GCC/MSVC/Clang
- **Shared memory:** All threads can access the same arrays/variables (no message passing needed)
- **Scalable:** Set number of threads via environment variable `OMP_NUM_THREADS`

### What is BFS (Breadth First Search)?
BFS explores a graph **level by level** — it visits all neighbors of the starting node before moving to their neighbors.

**Algorithm:**
1. Start at node S, mark it visited, add to queue
2. While queue is not empty:
   - Dequeue a node
   - Visit all its unvisited neighbors, mark them visited, add them to queue

**Data Structure Used:** Queue (FIFO — First In, First Out)

**Characteristic:** Finds the shortest path in unweighted graphs

### What is DFS (Depth First Search)?
DFS explores a graph by going **as deep as possible** along one path before backtracking.

**Algorithm:**
1. Start at node S, mark it visited, print it
2. For each unvisited neighbor, recursively apply DFS

**Data Structure Used:** Stack (implicit via recursion call stack)

**Characteristic:** Good for detecting cycles, topological sorting, connected components

### BFS vs DFS:
| Property | BFS | DFS |
|---|---|---|
| Traversal order | Level by level (breadth-first) | Deep along one path first |
| Data structure | Queue | Stack (recursion) |
| Finds shortest path | Yes (unweighted) | No |
| Memory usage | More (stores all nodes at a level) | Less (only current path) |
| Use case | Shortest path, social networks | Cycle detection, maze solving |

### What is a Race Condition?
When multiple threads access the same variable simultaneously and at least one writes to it, the result depends on the **unpredictable order of execution**. This is a race condition and causes bugs.

Example: Two threads both check `if (!visited[5])` and find it false. Both mark it visited and add it to the queue — the node is added twice. To prevent this, we use **critical sections**.

### Key OpenMP Directives Used:
| Directive | Meaning |
|---|---|
| `#pragma omp parallel` | Creates a team of threads; all threads execute the block in parallel |
| `#pragma omp single` | Only ONE thread (arbitrarily chosen) executes this block; others wait |
| `#pragma omp for` | Distributes loop iterations across available threads |
| `#pragma omp critical` | Only ONE thread at a time can execute this block (mutual exclusion) |
| `omp_get_thread_num()` | Returns the ID of the current thread (0, 1, 2, ...) |
| `omp_get_wtime()` | Returns current wall-clock time in seconds (used for timing) |

---

## Code Structure

**File:** `parallel_bfs-dfs_graph.cpp`

The code defines a `Graph` class with an adjacency list representation and implements 4 methods: `sequentialBFS`, `parallelBFS`, `sequentialDFS`, `parallelDFS`.

---

## Step-by-Step Code Walkthrough

### Graph Class Setup
```cpp
class Graph {
    int V;                        // Number of vertices
    vector<vector<int>> adj;      // Adjacency list

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);             // Create V empty neighbor lists
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);      // Undirected: add edge in both directions
    }
```
`adj` is a vector of vectors. `adj[3]` = list of all neighbors of node 3. Adding edge (u,v) to an undirected graph means both `adj[u]` and `adj[v]` must be updated.

### Sequential BFS
```cpp
void sequentialBFS(int start) {
    vector<bool> visited(V, false);   // All nodes unvisited initially
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();          // Get front of queue
        q.pop();
        cout << node << " ";

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);      // Add unvisited neighbors to queue
            }
        }
    }
}
```
Classic BFS — single threaded, processes one node at a time.

### Parallel BFS
```cpp
void parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int current;

        #pragma omp parallel shared(q, visited)
        {
            #pragma omp single
            {
                current = q.front();   // Only ONE thread dequeues
                q.pop();
                cout << "Thread " << omp_get_thread_num() << " visited " << current << endl;
            }

            #pragma omp for
            for (int i = 0; i < adj[current].size(); i++) {
                int neighbor = adj[current][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {   // Double-check inside critical
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}
```
- `#pragma omp parallel` creates a thread team for each node
- `#pragma omp single` ensures only one thread dequeues (queue is not thread-safe)
- `#pragma omp for` parallelizes the neighbor loop across threads
- `#pragma omp critical` protects the visited array from concurrent writes
- Double-check (`if (!visited[neighbor])`) inside critical prevents duplicate additions

### Sequential DFS
```cpp
void sequentialDFSUtil(int node, vector<bool> &visited) {
    visited[node] = true;
    cout << node << " ";
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            sequentialDFSUtil(neighbor, visited);   // Recursive call
        }
    }
}
```
Classic recursive DFS — mark, print, recurse on unvisited neighbors.

### Parallel DFS
```cpp
void parallelDFSUtil(int node, vector<bool> &visited) {
    #pragma omp critical
    {
        visited[node] = true;
        cout << node << " ";
    }

    #pragma omp parallel for
    for (int i = 0; i < adj[node].size(); i++) {
        int neighbor = adj[node][i];
        if (!visited[neighbor]) {
            parallelDFSUtil(neighbor, visited);
        }
    }
}
```
- Marks and prints the current node inside a critical section (thread-safe)
- Processes neighbors in parallel with `#pragma omp parallel for`
- Note: Parallel DFS is complex and can have issues with race conditions — it demonstrates the concept rather than being production-ready

### Main Function
```cpp
int main() {
    int V, E;
    cin >> V >> E;
    Graph g(V);

    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int startNode;
    cin >> startNode;

    double start, end;

    // Sequential BFS timing
    start = omp_get_wtime();
    g.sequentialBFS(startNode);
    end = omp_get_wtime();
    cout << "\nExecution Time: " << (end - start) * 1000 << " ms" << endl;

    // ... similarly for Parallel BFS, Sequential DFS, Parallel DFS
}
```
`omp_get_wtime()` returns wall-clock time. Difference × 1000 = milliseconds.

---

## How to Compile and Run

**On Linux/macOS:**
```bash
g++ -fopenmp parallel_bfs-dfs_graph.cpp -o hpc1
export OMP_NUM_THREADS=4
./hpc1
```

**On Windows (PowerShell):**
```powershell
g++ -fopenmp parallel_bfs-dfs_graph.cpp -o hpc1.exe
$env:OMP_NUM_THREADS=4
.\hpc1.exe
```

**Sample Input:**
```
Enter number of vertices: 6
Enter number of edges: 7
Enter edges (u v):
0 1
0 2
1 3
1 4
2 5
3 5
4 5
Enter starting vertex: 0
```

---

## Output Explanation

**Sequential BFS output:**
```
Sequential BFS: 0 1 2 3 4 5
Execution Time: 0.05 ms
```
Visits level by level: 0 → 1, 2 → 3, 4, 5

**Parallel BFS output:**
```
Thread 0 visited 0
Thread 2 visited 1
Thread 1 visited 2
...
Execution Time: 0.8 ms
```
Node processing can happen in different thread orders — the traversal order may differ from sequential BFS. The overhead of creating threads may make parallel version slightly slower for small graphs.

**Sequential DFS output:**
```
Sequential DFS: 0 1 3 5 2 4
Execution Time: 0.03 ms
```
Goes deep along one path first: 0→1→3→5→2→4

**Parallel DFS output:**
```
Parallel DFS: 0 1 3 5 2 4
Execution Time: 0.7 ms
```
Same nodes visited but possibly in different order due to thread scheduling.

**Performance Observation:**
For small graphs, parallel is actually SLOWER due to thread creation overhead. Parallel versions benefit from large graphs (many nodes/edges) where the cost of thread creation is amortized.

**Speedup = Sequential Time / Parallel Time**

---

## Conclusion

We implemented parallel BFS and DFS using OpenMP. For small graphs, sequential is faster because thread creation has overhead. For large graphs with many neighbors per node, parallel processing of neighbors provides significant speedup.

Key takeaways:
- `#pragma omp critical` prevents race conditions on shared data (visited array, queue)
- `#pragma omp single` ensures only one thread accesses non-thread-safe structures (queue)
- `#pragma omp for` efficiently distributes neighbor processing across threads
- BFS is more naturally parallelizable than DFS because each level's nodes can be processed in parallel

---

## Viva Questions & Answers

**Q1: What is OpenMP and why is it used?**
> OpenMP (Open Multi-Processing) is an API for shared-memory parallel programming. It uses pragma directives in C/C++ to parallelize code. We use it because it is easy to add parallelism to existing sequential code, works across platforms, and automatically manages thread creation and synchronization on multi-core CPUs.

**Q2: What is the difference between BFS and DFS?**
> BFS explores a graph level by level — it visits all neighbors at depth 1 before going to depth 2. It uses a Queue and finds shortest paths. DFS explores as far as possible along one branch before backtracking. It uses a Stack (recursion) and is better for cycle detection and topological sorting.

**Q3: Why do we need `#pragma omp critical` in parallel BFS?**
> Multiple threads simultaneously check `if (!visited[neighbor])` and can all find it false. Without critical, all threads would mark it visited and add it to the queue multiple times — a race condition. The critical section ensures only ONE thread at a time modifies the visited array and queue, preventing duplicates.

**Q4: What is `#pragma omp single` and why is it used for dequeuing?**
> `#pragma omp single` ensures only ONE thread executes the block (the others wait). A `queue` is not thread-safe — multiple threads calling `q.front()` and `q.pop()` simultaneously could corrupt the queue. By restricting dequeue to one thread, we ensure data integrity.

**Q5: What is a race condition?**
> A race condition occurs when two or more threads access a shared variable simultaneously and at least one writes to it. The result depends on the unpredictable order of thread execution. In BFS, if two threads both check `visited[5]` = false and both add node 5 to the queue, the same node is processed twice — incorrect behavior.

**Q6: What is `omp_get_wtime()` used for?**
> It returns the current wall-clock time in seconds as a `double`. By recording time before and after a block of code, we can measure how long it took: `time = (end - start) * 1000` gives milliseconds. It is more accurate than `clock()` for parallel code because it measures actual elapsed time, not CPU time.

**Q7: Why is parallel version sometimes slower than sequential for small graphs?**
> Creating a team of threads has overhead — memory allocation, synchronization. For a small graph with 6 nodes, the actual work is trivial (microseconds). Thread creation overhead can exceed the work done, making parallel slower. For large graphs with thousands of nodes and edges, parallelism provides significant speedup.

**Q8: What is `OMP_NUM_THREADS` and how do you set it?**
> It is an environment variable that tells OpenMP how many threads to create. `export OMP_NUM_THREADS=4` sets it to 4 threads on Linux. You can also set it in code with `omp_set_num_threads(4)`. If not set, OpenMP uses all available CPU cores by default.

**Q9: What is Speedup and how is it calculated?**
> Speedup = Sequential Execution Time / Parallel Execution Time. If sequential takes 100ms and parallel takes 25ms, speedup = 4×. Ideal speedup with N threads would be N× (linear speedup), but in practice it is less due to synchronization overhead, serial portions of code (Amdahl's Law), and memory bandwidth limits.

**Q10: What is shared memory vs distributed memory in parallel computing?**
> In shared memory (like OpenMP), all threads access the same RAM. Changes made by one thread are immediately visible to others. In distributed memory (like MPI), each process has its own separate memory; communication happens by explicitly sending messages over a network. OpenMP is for single machines with multiple cores; MPI is for clusters of separate machines.

**Q11: What is `#pragma omp parallel for`?**
> It combines `#pragma omp parallel` (create thread team) and `#pragma omp for` (distribute loop iterations). Each thread gets a chunk of loop iterations to execute independently. The work is divided automatically — if there are 4 threads and 100 iterations, each thread roughly handles 25 iterations.

**Q12: Can DFS be perfectly parallelized?**
> DFS is harder to parallelize than BFS because of its recursive, sequential nature. The standard recursive DFS visits one branch at a time. Parallelizing it means recursing into multiple neighbors simultaneously, but this can cause race conditions on the visited array and may create far more threads than available cores (thread explosion). It is possible but requires careful depth limiting.
