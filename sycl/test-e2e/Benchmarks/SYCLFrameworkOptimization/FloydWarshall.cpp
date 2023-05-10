// Test checks SYCL Framework optimization feature.
// The program is being compiled with optimization and without.
// Then the execution time of both are being relatively compared.
//
// The source code is taken from
// https://github.com/oneapi-src/oneAPI-samples/blob/master/DirectProgramming/C%2B%2BSYCL/GraphAlgorithms/all-pairs-shortest-paths/src/apsp.cpp
// and adjusted for testing purposes (parts of code has been deleted and
// refactored).

// RUN: %{build} -O0 -o %t.unoptimized.out
// RUN: %{build} -fsycl-optimize-non-user-code -O0 -o %t.optimized.out

// RUN: %{run} %t.unoptimized.out && \
// RUN: %{run} %t.optimized.out && \
// RUN: python %S%{fs-sep}compare.py --files=%t.unoptimized.output,%t.optimized.output --diff=0.5

#include <CL/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace sycl;

// Number of nodes in the graph.
constexpr int nodes = 256;

// Block length and block count (along a single dimension).
constexpr int block_length = 16;
constexpr int block_count = (nodes / block_length);

// Maximum distance between two adjacent nodes.
constexpr int max_distance = 100;
constexpr int infinite = (nodes * max_distance);

// Number of repetitions.
constexpr int repetitions = 2;

// Randomly initialize directed graph.
void InitializeDirectedGraph(int *graph) {
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < nodes; j++) {
      int cell = i * nodes + j;

      if (i == j) {
        graph[cell] = 0;
      } else if (rand() % 2) {
        graph[cell] = infinite;
      } else {
        graph[cell] = rand() % max_distance + 1;
      }
    }
  }
}

void CopyGraph(int *to, int *from) {
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < nodes; j++) {
      int cell = i * nodes + j;
      to[cell] = from[cell];
    }
  }
}

bool VerifyGraphsAreEqual(int *graph1, int *graph2) {
  for (int i = 0; i < nodes; i++) {
    for (int j = 0; j < nodes; j++) {
      int cell = i * nodes + j;
      if (graph1[cell] != graph2[cell]) {
        return false;
      }
    }
  }

  return true;
}

// The basic (sequential) implementation of Floyd Warshall algorithm for
// computing all pairs shortest paths. Is used to verify correctnes of
// the parallel algorithm.
void FloydWarshall(int *graph) {
  for (int k = 0; k < nodes; k++) {
    for (int i = 0; i < nodes; i++) {
      for (int j = 0; j < nodes; j++) {
        if (graph[i * nodes + j] >
            graph[i * nodes + k] + graph[k * nodes + j]) {
          graph[i * nodes + j] = graph[i * nodes + k] + graph[k * nodes + j];
        }
      }
    }
  }
}

typedef local_accessor<int, 2> LocalBlock;

// Inner loop of the blocked Floyd Warshall algorithm. A thread handles one cell
// of a block. To complete the computation of a block, this function is invoked
// by as many threads as there are cells in the block. Each such invocation
// computes as many iterations as there are blocks (along a single dimension).
// Moreover, each thread (simultaneously operating on a block), synchronizes
// between them at the end of each iteration. This is required for correctness
// as a following iteration depends on the previous iteration.
void BlockedFloydWarshallCompute(nd_item<1> &item, const LocalBlock &C,
                                 const LocalBlock &A, const LocalBlock &B,
                                 int i, int j) {
  for (int k = 0; k < block_length; k++) {
    if (C[i][j] > A[i][k] + B[k][j]) {
      C[i][j] = A[i][k] + B[k][j];
    }

    item.barrier(access::fence_space::local_space);
  }
}

// Phase 1 of blocked Floyd Warshall algorithm. It always operates on a block
// on the diagonal of the adjacency matrix of the graph.
void BlockedFloydWarshallPhase1(queue &q, int *graph, int round) {
  // Each group will process one block.
  constexpr auto blocks = 1;
  // Each item/thread in a group will handle one cell of the block.
  constexpr auto block_size = block_length * block_length;

  q.submit([&](handler &h) {
    LocalBlock block(range<2>(block_length, block_length), h);

    h.parallel_for<class KernelPhase1>(
        nd_range<1>(blocks * block_size, block_size), [=](nd_item<1> item) {
          auto tid = item.get_local_id(0);
          auto i = tid / block_length;
          auto j = tid % block_length;

          // Copy data to local memory.
          block[i][j] = graph[(round * block_length + i) * nodes +
                              (round * block_length + j)];
          item.barrier(access::fence_space::local_space);

          // Compute.
          BlockedFloydWarshallCompute(item, block, block, block, i, j);

          // Copy back data to global memory.
          graph[(round * block_length + i) * nodes +
                (round * block_length + j)] = block[i][j];
          item.barrier(access::fence_space::local_space);
        });
  });

  q.wait();
}

// Phase 2 of blocked Floyd Warshall algorithm. It always operates on blocks
// that are either on the same row or on the same column of a diagonal block.
void BlockedFloydWarshallPhase2(queue &q, int *graph, int round) {
  // Each group will process one block.
  constexpr auto blocks = block_count;
  // Each item/thread in a group will handle one cell of the block.
  constexpr auto block_size = block_length * block_length;

  q.submit([&](handler &h) {
    LocalBlock diagonal(range<2>(block_length, block_length), h);
    LocalBlock off_diag(range<2>(block_length, block_length), h);

    h.parallel_for<class KernelPhase2>(
        nd_range<1>(blocks * block_size, block_size), [=](nd_item<1> item) {
          auto gid = item.get_group(0);
          auto index = gid;

          if (index != round) {
            auto tid = item.get_local_id(0);
            auto i = tid / block_length;
            auto j = tid % block_length;

            // Copy data to local memory.
            diagonal[i][j] = graph[(round * block_length + i) * nodes +
                                   (round * block_length + j)];
            off_diag[i][j] = graph[(index * block_length + i) * nodes +
                                   (round * block_length + j)];
            item.barrier(access::fence_space::local_space);

            // Compute for blocks above and below the diagonal block.
            BlockedFloydWarshallCompute(item, off_diag, off_diag, diagonal, i,
                                        j);

            // Copy back data to global memory.
            graph[(index * block_length + i) * nodes +
                  (round * block_length + j)] = off_diag[i][j];

            // Copy data to local memory.
            off_diag[i][j] = graph[(round * block_length + i) * nodes +
                                   (index * block_length + j)];
            item.barrier(access::fence_space::local_space);

            // Compute for blocks at left and at right of the diagonal block.
            BlockedFloydWarshallCompute(item, off_diag, diagonal, off_diag, i,
                                        j);

            // Copy back data to global memory.
            graph[(round * block_length + i) * nodes +
                  (index * block_length + j)] = off_diag[i][j];
            item.barrier(access::fence_space::local_space);
          }
        });
  });

  q.wait();
}

// Phase 3 of blocked Floyd Warshall algorithm. It operates on all blocks except
// the ones that are handled in phase 1 and in phase 2 of the algorithm.
void BlockedFloydWarshallPhase3(queue &q, int *graph, int round) {
  // Each group will process one block.
  constexpr auto blocks = block_count * block_count;
  // Each item/thread in a group will handle one cell of the block.
  constexpr auto block_size = block_length * block_length;

  q.submit([&](handler &h) {
    LocalBlock A(range<2>(block_length, block_length), h);
    LocalBlock B(range<2>(block_length, block_length), h);
    LocalBlock C(range<2>(block_length, block_length), h);

    h.parallel_for<class KernelPhase3>(
        nd_range<1>(blocks * block_size, block_size), [=](nd_item<1> item) {
          auto bk = round;

          auto gid = item.get_group(0);
          auto bi = gid / block_count;
          auto bj = gid % block_count;

          if ((bi != bk) && (bj != bk)) {
            auto tid = item.get_local_id(0);
            auto i = tid / block_length;
            auto j = tid % block_length;

            // Copy data to local memory.
            A[i][j] = graph[(bi * block_length + i) * nodes +
                            (bk * block_length + j)];
            B[i][j] = graph[(bk * block_length + i) * nodes +
                            (bj * block_length + j)];
            C[i][j] = graph[(bi * block_length + i) * nodes +
                            (bj * block_length + j)];

            item.barrier(access::fence_space::local_space);

            // Compute.
            BlockedFloydWarshallCompute(item, C, A, B, i, j);

            // Copy back data to global memory.
            graph[(bi * block_length + i) * nodes + (bj * block_length + j)] =
                C[i][j];
            item.barrier(access::fence_space::local_space);
          }
        });
  });

  q.wait();
}

// Parallel implementation of blocked Floyd Warshall algorithm. It has three
// phases. Given a prior round of these computation phases are complete, phase 1
// is independent; Phase 2 can only execute after phase 1 completes; Similarly
// phase 3 depends on phase 2 so can only execute after phase 2 is complete.
//
// The inner loop of the sequential implementation is similar to:
//   g[i][j] = min(g[i][j], g[i][k] + g[k][j])
// A careful observation shows that for the kth iteration of the outer loop,
// the computation depends on cells either on the kth column, g[i][k] or on the
// kth row, g[k][j] of the graph. Phase 1 handles g[k][k], phase 2 handles
// g[*][k] and g[k][*], and phase 3 handles g[*][*] in that sequence. This cell
// level observations largely propagate to the blocks as well.
void BlockedFloydWarshall(queue &q, int *graph) {
  for (int round = 0; round < block_count; round++) {
    BlockedFloydWarshallPhase1(q, graph, round);
    BlockedFloydWarshallPhase2(q, graph, round);
    BlockedFloydWarshallPhase3(q, graph, round);
  }
}

class TimeInterval {
public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

int main() {
  try {
    queue q{default_selector_v};
    auto device = q.get_device();
    auto work_group_size = device.get_info<info::device::max_work_group_size>();
    auto block_size = block_length * block_length;

    cout << "Device: " << device.get_info<info::device::name>() << "\n";
    if (work_group_size < block_size) {
      cout << "Work group size " << work_group_size
           << " is less than required size " << block_size << "\n";
      return -1;
    }

    // Allocate unified shared memory so that graph data is accessible to both
    // the CPU and the device (e.g., a GPU).
    int *graph = (int *)malloc(sizeof(int) * nodes * nodes);
    int *sequential = malloc_shared<int>(nodes * nodes, q);
    int *parallel = malloc_shared<int>(nodes * nodes, q);

    if ((graph == nullptr) || (sequential == nullptr) ||
        (parallel == nullptr)) {
      if (graph != nullptr)
        free(graph);
      if (sequential != nullptr)
        free(sequential, q);
      if (parallel != nullptr)
        free(parallel, q);

      cout << "Memory allocation failure.\n";
      return -1;
    }

    // Initialize directed graph.
    InitializeDirectedGraph(graph);

    // Measure execution times.
    double elapsed_p = 0;

    cout << "Running " << repetitions << " iterations...\n";
    for (int i = 0; i < repetitions; i++) {
      cout << "Iteration: " << (i + 1) << "\n";

      // Sequential all pairs shortest paths.
      CopyGraph(sequential, graph);
      FloydWarshall(sequential);

      // Parallel all pairs shortest paths.
      CopyGraph(parallel, graph);

      TimeInterval timer_p;
      BlockedFloydWarshall(q, parallel);
      elapsed_p += timer_p.Elapsed();

      // Verify two results are equal.
      if (!VerifyGraphsAreEqual(sequential, parallel)) {
        cout << "Failed to correctly compute all pairs shortest paths!\n";
        return 1;
      }
    }

    elapsed_p /= repetitions;
    cout << "Time: " << elapsed_p << " sec\n";

    // Free unified shared memory.
    free(graph);
    free(sequential, q);
    free(parallel, q);
  } catch (std::exception const &e) {
    cout << "An exception is caught while computing on device.\n";
    terminate();
  }

  return 0;
}
