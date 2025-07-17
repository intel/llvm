// RUN: %{run-aux} %{build} -Wno-error=deprecated-pragma -o %t.out %hip_options
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{run-aux} %{build} -Wno-error=deprecated-pragma -fpreview-breaking-changes -o %t2.out %hip_options %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: target-amd

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/usm.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  queue Queue;

  const size_t Size = 128;
  int *PtrX = malloc_device<int>(Size, Queue);
  int *PtrY = malloc_device<int>(Size, Queue);

  exp_ext::command_graph Graph{Queue};

  Graph.begin_recording(Queue);

  auto EventA = Queue.submit([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrX[i] = i;
        PtrY[i] = 0;
      }
    });
  });

  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventA);

    CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
      if (!IH.ext_codeplay_has_graph()) {
        assert(false && "Native Handle should have a graph");
      }
      // Newly created stream for this node
      auto NativeStream = IH.get_native_queue<backend::ext_oneapi_hip>();
      // Graph already created with hipGraphCreate
      HIPGraph NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_hip>();

      // Start stream capture
      // After HIP 6.2 we can use hipStreamBeginCaptureToGraph to capture
      // the stream directly in the native graph, rather than needing to
      // instantiate the stream capture as a new graph.
#if HIP_VERSION > 60200000
      auto Res =
          hipStreamBeginCaptureToGraph(NativeStream, NativeGraph, nullptr,
                                       nullptr, 0, hipStreamCaptureModeGlobal);
      assert(Res == hipSuccess);
#else
      auto Res =
          hipStreamBeginCapture(NativeStream, hipStreamCaptureModeGlobal);
      assert(Res == hipSuccess);
#endif

      // Add memcopy node
      Res = hipMemcpyWithStream(PtrY, PtrX, sizeof(int) * Size,
                                hipMemcpyDefault, NativeStream);
      assert(Res == hipSuccess);

#if HIP_VERSION > 60200000
      Res = hipStreamEndCapture(NativeStream, &NativeGraph);
      assert(Res == hipSuccess);
#else
      // hipStreamEndCapture returns a new graph, if we overwrite
      // "NativeGraph" it won't be picked up by the UR runtime, as it's
      // a passed-by-value pointer
      HIPGraph RecordedGraph;
      Res = hipStreamEndCapture(NativeStream, &RecordedGraph);
      assert(Res == hipSuccess);

      // Add graph to native graph as a child node
      // Need to return a node object for the node to be created,
      // can't be nullptr.
      HIPGraphNode Node;
      Res = hipGraphAddChildGraphNode(&Node, NativeGraph, nullptr, 0,
                                      RecordedGraph);
      assert(Res == hipSuccess);
#endif
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventB);
    CGH.parallel_for(range<1>{Size}, [=](id<1> it) { PtrY[it] *= 2; });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostData(Size);
  Queue.copy(PtrY, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    const int Ref = i * 2;
    assert(Ref == HostData[i]);
  }

  free(PtrX, Queue);
  free(PtrY, Queue);

  return 0;
}
