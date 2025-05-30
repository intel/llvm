// RUN: %{build} -o %t.out %cuda_options
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %cuda_options %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: target-nvidia, cuda_dev_kit

#include <cuda.h>
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
      auto NativeStream = IH.get_native_queue<backend::ext_oneapi_cuda>();
      // Graph already created with cuGraphCreate
      CUgraph NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_cuda>();

      // Start stream capture
      // After CUDA 12.3 we can use cuStreamBeginCaptureToGraph to capture
      // the stream directly in the native graph, rather than needing to
      // instantiate the stream capture as a new graph.
#if CUDA_VERSION >= 12030
      auto Res = cuStreamBeginCaptureToGraph(NativeStream, NativeGraph, nullptr,
                                             nullptr, 0,
                                             CU_STREAM_CAPTURE_MODE_GLOBAL);
      assert(Res == CUDA_SUCCESS);
#else
      auto Res =
          cuStreamBeginCapture(NativeStream, CU_STREAM_CAPTURE_MODE_GLOBAL);
      assert(Res == CUDA_SUCCESS);
#endif

      // Add memcopy node
      Res = cuMemcpyAsync((CUdeviceptr)PtrY, (CUdeviceptr)PtrX,
                          Size * sizeof(int), NativeStream);
      assert(Res == CUDA_SUCCESS);

#if CUDA_VERSION >= 12030
      Res = cuStreamEndCapture(NativeStream, &NativeGraph);
      assert(Res == CUDA_SUCCESS);
#else
      // cuStreamEndCapture returns a new graph, if we overwrite
      // "NativeGraph" it won't be picked up by the UR runtime, as it's
      // a passed-by-value pointer
      CUgraph RecordedGraph;
      Res = cuStreamEndCapture(NativeStream, &RecordedGraph);
      assert(Res == CUDA_SUCCESS);

      // Add graph to native graph as a child node
      // Need to return a node object for the node to be created,
      // can't be nullptr.
      CUgraphNode Node;
      Res = cuGraphAddChildGraphNode(&Node, NativeGraph, nullptr, 0,
                                     RecordedGraph);
      assert(Res == CUDA_SUCCESS);
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
