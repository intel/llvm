// RUN: %{build} -o %t.out %cuda_options
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %cuda_options %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: target-nvidia, cuda_dev_kit

#include <cuda.h>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  queue Queue;

  const size_t Size = 128;
  buffer<int, 1> BufX{Size};
  BufX.set_write_back(false);
  buffer<int, 1> BufY{Size};
  BufY.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue, {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    Graph.begin_recording(Queue);

    Queue.submit([&](handler &CGH) {
      auto AccX = BufX.get_access(CGH);
      auto AccY = BufY.get_access(CGH);
      CGH.single_task([=]() {
        for (size_t i = 0; i < Size; i++) {
          AccX[i] = i;
          AccY[i] = 0;
        }
      });
    });

    Queue.submit([&](handler &CGH) {
      auto AccX = BufX.get_access(CGH);
      auto AccY = BufY.get_access(CGH);

      CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
        if (!IH.ext_codeplay_has_graph()) {
          assert(false && "Native Handle should have a graph");
        }
        // Newly created stream for this node
        auto NativeStream = IH.get_native_queue<backend::ext_oneapi_cuda>();
        // Graph already created with cuGraphCreate
        CUgraph NativeGraph =
            IH.ext_codeplay_get_native_graph<backend::ext_oneapi_cuda>();

        auto PtrX = IH.get_native_mem<backend::ext_oneapi_cuda>(AccX);
        auto PtrY = IH.get_native_mem<backend::ext_oneapi_cuda>(AccY);

        // Start stream capture
        // After CUDA 12.3 we can use cuStreamBeginCaptureToGraph to capture
        // the stream directly in the native graph, rather than needing to
        // instantiate the stream capture as a new graph.
#if CUDA_VERSION >= 12030
        auto Res = cuStreamBeginCaptureToGraph(NativeStream, NativeGraph,
                                               nullptr, nullptr, 0,
                                               CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert(Res == CUDA_SUCCESS);
#else
        auto Res =
            cuStreamBeginCapture(NativeStream, CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert(Res == CUDA_SUCCESS);
#endif

        // Add memcopy node
        Res = cuMemcpyAsync(PtrY, PtrX, Size * sizeof(int), NativeStream);
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
      auto AccY = BufY.get_access(CGH);
      CGH.parallel_for(range<1>{Size}, [=](id<1> it) { AccY[it] *= 2; });
    });

    Graph.end_recording();

    auto ExecGraph = Graph.finalize();
    Queue.ext_oneapi_graph(ExecGraph).wait();
  }

  auto HostAcc = BufY.get_host_access();
  for (size_t i = 0; i < Size; i++) {
    const int Ref = i * 2;
    assert(Ref == HostAcc[i]);
  }

  return 0;
}
