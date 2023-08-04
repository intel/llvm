#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <condition_variable> // std::conditional_variable
#include <mutex>              // std::mutex, std::unique_lock
#include <numeric>

// Test constants.
constexpr size_t Size = 1024;      // Number of data elements in a buffer.
constexpr unsigned Iterations = 5; // Iterations of graph to execute.
constexpr size_t Offset = 100; // Number of offset elements for Buffer accessors

// Namespace alias to use in test code.
namespace exp_ext = sycl::ext::oneapi::experimental;
// Make tests less verbose by using sycl namespace.
using namespace sycl;

// We have 4 versions of the same kernel sequence for testing a combination
// of graph construction API against memory model. Each submits the same pattern
/// of 4 kernels with a diamond dependency.
//
//                 |    Buffers    |      USM          |
// ----------------|---------------|-------------------|
// Record & Replay | run_kernels() | run_kernels_usm() |
// ----------------|---------------|-------------------|
// Explicit API    | add_kernels() | add_kernels_usm() |

/// Calculates reference data on the host for a given number of executions
/// @param[in] Iterations Number of iterations of kernel sequence to run.
/// @param[in] Size Number of elements in vectors
/// @param[in,out] ReferenceA First input/output.
/// @param[in,out] ReferenceB Second input/output.
/// @param[in,out] ReferenceC Third input/output.
template <typename T>
void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceA,
                              std::vector<T> &ReferenceB,
                              std::vector<T> &ReferenceC) {
  for (size_t n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceA[i]++;
      ReferenceB[i] += ReferenceA[i];
      ReferenceC[i] -= ReferenceA[i];
      ReferenceB[i]--;
      ReferenceC[i]--;
    }
  }
}

/// Test Record and Replay graph construction with buffers.
///
/// @param Q Queue to submit nodes to.
/// @param Size Number of elements in the buffers.
/// @param BufferA First input/output to use in kernels.
/// @param BufferB Second input/output to use in kernels.
/// @param BufferC Third input/output to use in kernels.
///
/// @return An event corresponding to the exit node of the submissions sequence.
template <typename T>
event run_kernels(queue Q, const size_t Size, buffer<T> BufferA,
                  buffer<T> BufferB, buffer<T> BufferC) {
  // Read & write Buffer A.
  Q.submit([&](handler &CGH) {
    auto DataA = BufferA.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { DataA[Id]++; });
  });

  // Reads Buffer A.
  // Read & Write Buffer B.
  Q.submit([&](handler &CGH) {
    auto DataA = BufferA.template get_access<access::mode::read>(CGH);
    auto DataB = BufferB.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> Id) { DataB[Id] += DataA[Id]; });
  });

  // Reads Buffer A.
  // Read & writes Buffer C
  Q.submit([&](handler &CGH) {
    auto DataA = BufferA.template get_access<access::mode::read>(CGH);
    auto DataC = BufferC.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> Id) { DataC[Id] -= DataA[Id]; });
  });

  // Read & write Buffers B and C.
  auto ExitEvent = Q.submit([&](handler &CGH) {
    auto DataB = BufferB.template get_access<access::mode::read_write>(CGH);
    auto DataC = BufferC.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      DataB[Id]--;
      DataC[Id]--;
    });
  });

  return ExitEvent;
}

/// Test Explicit API graph construction with buffers.
///
/// @param Graph Modifiable graph to add commands to.
/// @param Size Number of elements in the buffers.
/// @param BufferA First input/output to use in kernels.
/// @param BufferB Second input/output to use in kernels.
/// @param BufferC Third input/output to use in kernels.
///
/// @return Exit node of the submission sequence.
template <typename T>
exp_ext::node
add_kernels(exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph,
            const size_t Size, buffer<T> BufferA, buffer<T> BufferB,
            buffer<T> BufferC) {
  // Read & write Buffer A
  Graph.add([&](handler &CGH) {
    auto DataA = BufferA.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) { DataA[Id]++; });
  });

  // Reads Buffer A
  // Read & Write Buffer B
  Graph.add([&](handler &CGH) {
    auto DataA = BufferA.template get_access<access::mode::read>(CGH);
    auto DataB = BufferB.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> Id) { DataB[Id] += DataA[Id]; });
  });

  // Reads Buffer A
  // Read & writes Buffer C
  Graph.add([&](handler &CGH) {
    auto DataA = BufferA.template get_access<access::mode::read>(CGH);
    auto DataC = BufferC.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size),
                     [=](item<1> Id) { DataC[Id] -= DataA[Id]; });
  });

  // Read & write Buffers B and C
  auto ExitNode = Graph.add([&](handler &CGH) {
    auto DataB = BufferB.template get_access<access::mode::read_write>(CGH);
    auto DataC = BufferC.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      DataB[Id]--;
      DataC[Id]--;
    });
  });
  return ExitNode;
}

//// Test Explicit API graph construction with USM.
///
/// @param Q Command-queue to make kernel submissions to.
/// @param Size Number of elements in the buffers.
/// @param DataA Pointer to first USM allocation to use in kernels.
/// @param DataB Pointer to second USM allocation to use in kernels.
/// @param DataC Pointer to third USM allocation to use in kernels.
///
/// @return Event corresponding to the exit node of the submission sequence.
template <typename T>
event run_kernels_usm(queue Q, const size_t Size, T *DataA, T *DataB,
                      T *DataC) {
  // Read & write Buffer A
  auto EventA = Q.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataA[LinID]++;
    });
  });

  // Reads Buffer A
  // Read & Write Buffer B
  auto EventB = Q.submit([&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataB[LinID] += DataA[LinID];
    });
  });

  // Reads Buffer A
  // Read & writes Buffer C
  auto EventC = Q.submit([&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataC[LinID] -= DataA[LinID];
    });
  });

  // Read & write Buffers B and C
  auto ExitEvent = Q.submit([&](handler &CGH) {
    CGH.depends_on({EventB, EventC});
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataB[LinID]--;
      DataC[LinID]--;
    });
  });
  return ExitEvent;
}

/// Test Explicit API graph construction with USM.
///
/// @param Graph Modifiable graph to add commands to.
/// @param Size Number of elements in the buffers.
/// @param DataA Pointer to first USM allocation to use in kernels.
/// @param DataB Pointer to second USM allocation to use in kernels.
/// @param DataC Pointer to third USM allocation to use in kernels.
///
/// @return Exit node of the submission sequence.
template <typename T>
exp_ext::node
add_kernels_usm(exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph,
                const size_t Size, T *DataA, T *DataB, T *DataC) {
  // Read & write Buffer A
  auto NodeA = Graph.add([&](handler &CGH) {
    CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
      auto LinID = Id.get_linear_id();
      DataA[LinID]++;
    });
  });

  // Reads Buffer A
  // Read & Write Buffer B
  auto NodeB = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
          auto LinID = Id.get_linear_id();
          DataB[LinID] += DataA[LinID];
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // Reads Buffer A
  // Read & writes Buffer C
  auto NodeC = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
          auto LinID = Id.get_linear_id();
          DataC[LinID] -= DataA[LinID];
        });
      },
      {exp_ext::property::node::depends_on(NodeA)});

  // Read & write data B and C
  auto ExitNode = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> Id) {
          auto LinID = Id.get_linear_id();
          DataB[LinID]--;
          DataC[LinID]--;
        });
      },
      {exp_ext::property::node::depends_on(NodeB, NodeC)});

  return ExitNode;
}

// Values for dotp tests
const float Alpha = 1.0f;
const float Beta = 2.0f;
const float Gamma = 3.0f;

// Reference function for dotp
float dotp_reference_result(size_t N) {
  float Sum = 0.0f;

  for (size_t i = 0; i < N; ++i) {
    Sum += (Alpha * 1.0f + Beta * 2.0f) * (Gamma * 3.0f + Beta * 2.0f);
  }

  return Sum;
}

/* Single use thread barrier which makes threads wait until defined number of
 * threads reach it.
 * std:barrier should be used instead once compiler is moved to C++20 standard.
 */
class Barrier {
public:
  Barrier() = delete;
  explicit Barrier(std::size_t count) : threadNum(count) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    if (--threadNum == 0) {
      cv.notify_all();
    } else {
      cv.wait(lock, [this] { return threadNum == 0; });
    }
  }

private:
  std::mutex mutex;
  std::condition_variable cv;
  std::size_t threadNum;
};
