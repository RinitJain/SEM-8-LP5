// Design and implement Parallel Breadth First Search and Depth First Search based on
// existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS .
// Measure the performance of sequential and parallel algorithms.
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    // Add edge to graph
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    // ---------------- SEQUENTIAL BFS ----------------
    void sequentialBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            cout << node << " ";

            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    // ---------------- PARALLEL BFS ----------------
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {

            int current;

            #pragma omp parallel shared(q, visited)
            {
                // Only one thread removes element from queue
                #pragma omp single
                {
                    current = q.front();
                    q.pop();

                    cout << "Thread " << omp_get_thread_num() << " visited " << current << endl;
                }

                // Parallel traversal of neighbors
                #pragma omp for
                for (int i = 0; i < adj[current].size(); i++) {

                    int neighbor = adj[current][i];

                    if (!visited[neighbor]) {

                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---------------- SEQUENTIAL DFS ----------------
    void sequentialDFSUtil(int node, vector<bool> &visited) {

        visited[node] = true;
        cout << node << " ";

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                sequentialDFSUtil(neighbor, visited);
            }
        }
    }

    void sequentialDFS(int start) {
        vector<bool> visited(V, false);
        sequentialDFSUtil(start, visited);
    }

    // ---------------- PARALLEL DFS ----------------
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

    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        parallelDFSUtil(start, visited);
    }
};

int main() {

    int V, E;

    cout << "Enter number of vertices: ";
    cin >> V;

    cout << "Enter number of edges: ";
    cin >> E;

    Graph g(V);

    cout << "Enter edges (u v):" << endl;

    for (int i = 0; i < E; i++) {

        int u, v;
        cin >> u >> v;

        g.addEdge(u, v);
    }

    int startNode;

    cout << "Enter starting vertex: ";
    cin >> startNode;

    double start, end;

    // ---------------- SEQUENTIAL BFS ----------------
    cout << "\nSequential BFS: ";
    start = omp_get_wtime();
    g.sequentialBFS(startNode);
    end = omp_get_wtime();
    cout << "\nExecution Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- PARALLEL BFS ----------------
    cout << "\nParallel BFS:" << endl;
    start = omp_get_wtime();
    g.parallelBFS(startNode);
    end = omp_get_wtime();
    cout << "Execution Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- SEQUENTIAL DFS ----------------
    cout << "\nSequential DFS: ";
    start = omp_get_wtime();
    g.sequentialDFS(startNode);
    end = omp_get_wtime();
    cout << "\nExecution Time: " << (end - start) * 1000 << " ms" << endl;

    // ---------------- PARALLEL DFS ----------------
    cout << "\nParallel DFS: ";
    start = omp_get_wtime();
    g.parallelDFS(startNode);
    end = omp_get_wtime();
    cout << "\nExecution Time: " << (end - start) * 1000 << " ms" << endl;

    return 0;
}

/*

COMPILATION:

Linux:
g++ -fopenmp hpc1.cpp -o hpc1
export OMP_NUM_THREADS=4
./hpc1

Windows:
g++ -fopenmp hpc1.cpp -o hpc1.exe
$env:OMP_NUM_THREADS=4
.\hpc1.exe

*/