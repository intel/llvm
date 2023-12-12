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

// Helper functions for wrapping depends_on calls when add_node is used so they
// are not used in the explicit API
template <typename T> inline void depends_on_helper(sycl::handler &CGH, T Dep) {
#ifdef GRAPH_E2E_RECORD_REPLAY
  CGH.depends_on(Dep);
#endif
  (void)CGH;
  (void)Dep;
}

template <typename T>
inline void depends_on_helper(sycl::handler &CGH,
                              std::initializer_list<T> DepList) {
#ifdef GRAPH_E2E_RECORD_REPLAY
  CGH.depends_on(DepList);
#endif
  (void)CGH;
  (void)DepList;
}

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

/// Adds a common series of nodes to a graph forming a diamond dependency
/// structure using USM pointers for the inputs. Can be used for either Explicit
/// or Record and Replay with the choice being dictated by defining one of
/// GRAPH_E2E_EXPLICIT or GRAPH_E2E_RECORD_REPLAY.
///
/// @param Graph Modifiable graph to add commands to.
/// @param Queue Queue to be used for record and replay.
/// @param Size Number of elements in the buffers.
/// @param DataA Pointer to first USM allocation to use in kernels.
/// @param DataB Pointer to second USM allocation to use in kernels.
/// @param DataC Pointer to third USM allocation to use in kernels.
///
/// @return If using Explicit API this will be the last node added, if Record
/// and Replay this will be an event corresponding to the last submission.
template <typename T>
auto add_nodes(exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph,
               queue Queue, const size_t Size, T *DataA, T *DataB, T *DataC) {
#if defined(GRAPH_E2E_EXPLICIT)
  return add_kernels_usm(Graph, Size, DataA, DataB, DataC);
#elif defined(GRAPH_E2E_RECORD_REPLAY)
  Graph.begin_recording(Queue);
  auto ev = run_kernels_usm(Queue, Size, DataA, DataB, DataC);
  Graph.end_recording(Queue);
  return ev;
#else
  assert(0 && "Error: Cannot use add_nodes without selecting an API");
#endif
}

/// Adds a common series of nodes to a graph forming a diamond dependency
/// structure using Buffers for the inputs. Can be used for either Explicit
/// or Record and Replay with the choice being dictated by defining one of
/// GRAPH_E2E_EXPLICIT or GRAPH_E2E_RECORD_REPLAY.
///
/// @param Graph Modifiable graph to add commands to.
/// @param Queue Queue to be used for record and replay.
/// @param Size Number of elements in the buffers.
/// @param BufferA First input/output to use in kernels.
/// @param BufferB Second input/output to use in kernels.
/// @param BufferC Third input/output to use in kernels.
///
/// @return If using Explicit API this will be the last node added, if Record
/// and Replay API this will be an event corresponding to the last submission.
template <typename T>
auto add_nodes(exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph,
               queue Queue, const size_t Size, buffer<T> BufferA,
               buffer<T> BufferB, buffer<T> BufferC) {
#if defined(GRAPH_E2E_EXPLICIT)
  return add_kernels(Graph, Size, BufferA, BufferB, BufferC);
#elif defined(GRAPH_E2E_RECORD_REPLAY)
  Graph.begin_recording(Queue);
  auto ev = run_kernels(Queue, Size, BufferA, BufferB, BufferC);
  Graph.end_recording(Queue);
  return ev;
#else
  assert(0 && "Error: Cannot use add_nodes without selecting an API");
#endif
}

/// Adds a single node to the graph in an API agnostic way. Can be used for
/// either Explicit or Record and Replay with the choice being dictated by
/// defining one of GRAPH_E2E_EXPLICIT or GRAPH_E2E_RECORD_REPLAY.
///
/// @tparam CGFunc Type of the command group function.
/// @tparam DepT Type of all the dependencies.
/// @param Graph Modifiable graph to add commands to.
/// @param Queue Queue to be used for record and replay.
/// @param CGF The command group function representing the node
/// @param Deps Parameter pack of dependencies, if they are Nodes we pass them
/// to explicit API add, otherwise they are ignored.
/// @return If using the Explicit API this will be the node that was added, if
/// Record and Replay this will be an event representing the submission.
template <typename CGFunc, typename... DepT>
auto add_node(exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph,
              queue Queue, CGFunc CGF, DepT... Deps) {
#if defined(GRAPH_E2E_EXPLICIT)
  if constexpr ((std::is_same_v<exp_ext::node, DepT> && ...)) {
    return Graph.add(CGF, {exp_ext::property::node::depends_on(Deps...)});
  } else {
    return Graph.add(CGF);
  }
#elif defined(GRAPH_E2E_RECORD_REPLAY)
  Graph.begin_recording(Queue);
  auto ev = Queue.submit(CGF);
  Graph.end_recording(Queue);
  return ev;
#else
  assert(0 && "Error: Cannot use add_node without selecting an API");
#endif
}

/// Adds an empty node to the graph in an API agnostic way. Can be used for
/// either Explicit or Record and Replay with the choice being dictated by
/// defining one of GRAPH_E2E_EXPLICIT or GRAPH_E2E_RECORD_REPLAY.
///
/// @tparam DepT Type of all the dependencies.
/// @param Graph Modifiable graph to add commands to.
/// @param Queue Queue to be used for record and replay.
/// @param Deps Parameter pack of dependencies, if they are Nodes we pass them
/// to explicit API add, otherwise they are passed as events to queue::submit()
/// for the empty node submission.
/// @return If using the Explicit API this will be the node that was added, if
/// Record and Replay this will be an event representing the submission.
template <typename... DepT>
auto add_empty_node(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph, queue Queue,
    DepT... Deps) {
#if defined(GRAPH_E2E_EXPLICIT)
  if constexpr ((std::is_same_v<exp_ext::node, DepT> && ...)) {
    return Graph.add({exp_ext::property::node::depends_on(Deps...)});
  } else {
    return Graph.add();
  }
#elif defined(GRAPH_E2E_RECORD_REPLAY)
  Graph.begin_recording(Queue);
  auto ev = Queue.submit(
      [&](sycl::handler &CGH) { CGH.depends_on(std::vector<event>{Deps...}); });
  Graph.end_recording(Queue);
  return ev;
#else
  assert(0 && "Error: Cannot use add_empty_node without selecting an API");
#endif
}

// Values for dotp tests
constexpr int Alpha = 1;
constexpr int Beta = 2;
constexpr int Gamma = 3;

// Reference function for dotp
int dotp_reference_result(size_t N) {
  return N * (Alpha * 1 + Beta * 2) * (Gamma * 3 + Beta * 2);
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

template <typename T>
bool inline check_value(const T &Ref, const T &Got,
                        const std::string &VariableName) {
  if (Got != Ref) {
    std::cout << "Unexpected value of " << VariableName << ": " << Got
              << " (got) vs " << Ref << " (expected)" << std::endl;
    return false;
  }

  return true;
}

template <typename T>
bool inline check_value(const size_t index, const T &Ref, const T &Got,
                        const std::string &VariableName) {
  if (Got != Ref) {
    std::cout << "Unexpected value at index " << index << " for "
              << VariableName << ": " << Got << " (got) vs " << Ref
              << " (expected)" << std::endl;
    return false;
  }

  return true;
}

bool are_graphs_supported(queue &Queue) {
  auto Device = Queue.get_device();

  exp_ext::graph_support_level SupportsGraphs =
      Device.get_info<exp_ext::info::device::graph_support>();

  return SupportsGraphs != exp_ext::graph_support_level::unsupported;
}
