// RUN: %{build} -g -o %t.out -lcuda
// RUN: %{run} %t.out
// REQUIRES: cuda

#include "../graph_common.hpp"
#include <cuda.h>
#include <sycl/backend.hpp>
#include <sycl/interop_handle.hpp>

int main() {
  queue Queue;

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
      if (IH.has_graph()) {
        // Newly created stream for this node
        auto NativeStream = IH.get_native_queue<backend::ext_oneapi_cuda>();
        // Graph already created with cuGraphCreate
        CUgraph NativeGraph = IH.get_native_graph<backend::ext_oneapi_cuda>();

        // Start stream capture
        auto Res =
            cuStreamBeginCapture(NativeStream, CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert(Res == CUDA_SUCCESS);

        // Add memcopy node
        Res = cuMemcpyAsync((CUdeviceptr)PtrY, (CUdeviceptr)PtrX,
                            Size * sizeof(int), NativeStream);
        assert(Res == CUDA_SUCCESS);

        // cuStreamEndCapture returns a new graph, if we overwrite
        // "NativeGraph" it won't be picked up by the UR runtime, as it's
        // a passed-by-value pointer
        CUgraph RecordedGraph;
        Res = cuStreamEndCapture(NativeStream, &RecordedGraph);

        // Add graph to native graph as a child node
        // Need to return a node object for the node to be created,
        // can't be nullptr.
        CUgraphNode Node;
        cuGraphAddChildGraphNode(&Node, NativeGraph, nullptr, 0, RecordedGraph);
        assert(Res == CUDA_SUCCESS);
      } else {
        assert(false && "Native Handle should have a graph");
      }
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
    assert(check_value(Ref, HostData[i],
                       std::string("HostData at index ") + std::to_string(i)));
  }

  free(PtrX, Queue);
  free(PtrY, Queue);

  return 0;
}
