// Tests destruction callbacks: resource cleanup on graph destruction, and
// copy/move constraint validation (CopyConstructible required,
// MoveConstructible as rvalue optimization, assignment not required).

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

struct CopyMoveTracker {
  bool *CopiedFlag;
  bool *MovedFlag;
  int Value;

  CopyMoveTracker(int V, bool *Copied, bool *Moved)
      : CopiedFlag(Copied), MovedFlag(Moved), Value(V) {}
  CopyMoveTracker(const CopyMoveTracker &Other)
      : CopiedFlag(Other.CopiedFlag), MovedFlag(Other.MovedFlag),
        Value(Other.Value) {
    *CopiedFlag = true;
  }
  CopyMoveTracker(CopyMoveTracker &&Other)
      : CopiedFlag(Other.CopiedFlag), MovedFlag(Other.MovedFlag),
        Value(Other.Value) {
    *MovedFlag = true;
  }

  // Assignment operators not required by spec
  CopyMoveTracker &operator=(const CopyMoveTracker &) = delete;
  CopyMoveTracker &operator=(CopyMoveTracker &&) = delete;
};

int main() {
  queue Queue{property::queue::in_order{}};

  int ObservedLvalue = 0;
  int ObservedRvalue = 0;
  size_t ObservedSize = 0;
  bool CleanupCallbackInvoked = false;

  const size_t N = 64;
  int *Data = malloc_device<int>(N, Queue);

  {
#ifdef GRAPH_E2E_NATIVE_RECORDING
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::enable_native_recording{}}};
#else
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

    add_node(Graph, Queue, [&](handler &CGH) {
      CGH.parallel_for(range<1>{N},
                       [=](id<1> idx) { Data[idx] = static_cast<int>(idx); });
    });

    // Resource cleanup: free device memory on graph destruction.
    // Also verifies lvalue reference arguments are copied into internal storage
    // and not referenced after registration.
    size_t SizeCopy = N;
    Graph.set_destruction_callback(
        [](int *Ptr, queue Q, size_t &Size, size_t *OutSize, bool *OutInvoked) {
          sycl::free(Ptr, Q);
          *OutSize = Size;
          *OutInvoked = true;
        },
        Data, Queue, SizeCopy, &ObservedSize, &CleanupCallbackInvoked);
    SizeCopy = 0;
    assert(!CleanupCallbackInvoked && "Cleanup callback should not fire yet");

    // Lvalue arg: must be copied into the graph's stored tuple.
    bool LvalueCopied = false, LvalueMoved = false;
    CopyMoveTracker LvalueTracker{42, &LvalueCopied, &LvalueMoved};
    Graph.set_destruction_callback(
        [](CopyMoveTracker T, int *Out) { *Out = T.Value; }, LvalueTracker,
        &ObservedLvalue);
    assert(LvalueCopied && "Lvalue arg should be copied");

    // Rvalue arg: move-constructible optimization should kick in.
    bool RvalueCopied = false, RvalueMoved = false;
    CopyMoveTracker RvalueTracker{99, &RvalueCopied, &RvalueMoved};
    Graph.set_destruction_callback(
        [](CopyMoveTracker T, int *Out) { *Out = T.Value; },
        std::move(RvalueTracker), &ObservedRvalue);
    assert(RvalueMoved && !RvalueCopied &&
           "Rvalue arg should be moved, not copied");

    auto ExecGraph = Graph.finalize();

    std::vector<int> HostData(N);
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.memcpy(HostData.data(), Data, N * sizeof(int));
    Queue.wait();
    for (size_t i = 0; i < N; i++) {
      assert(check_value(i, static_cast<int>(i), HostData[i], "Data"));
    }
  }

  assert(CleanupCallbackInvoked && "Cleanup callback was not invoked");
  assert(ObservedLvalue == 42 &&
         "Lvalue callback should observe original value");
  assert(ObservedRvalue == 99 &&
         "Rvalue callback should observe original value");
  assert(ObservedSize == N &&
         "Cleanup callback should observe pre-mutation size");
  return 0;
}
