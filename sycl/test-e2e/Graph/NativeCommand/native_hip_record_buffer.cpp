// FIXME: the rocm include path and link path are highly platform dependent,
// we should set this with some variable instead.
// RUN: %{build} -Wno-error=deprecated-pragma -o %t.out -I%rocm_path/include -L%rocm_path/lib -lamdhip64
// RUN: %{run} %t.out
// REQUIRES: target-amd

#include "../graph_common.hpp"
#include <sycl/backend.hpp>
#include <sycl/interop_handle.hpp>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

int main() {
  queue Queue;

  int *PtrX = malloc_device<int>(Size, Queue);
  buffer<int, 1> BufX{Size};
  BufX.set_write_back(false);
  buffer<int, 1> BufY{Size};
  BufY.set_write_back(false);

  {

    exp_ext::command_graph Graph{
        Queue, {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    Graph.begin_recording(Queue);

    Queue.submit([&](handler &CGH) {
      auto AccX = BufX.get_access();
      auto AccY = BufY.get_access();
      CGH.single_task([=]() {
        for (size_t i = 0; i < Size; i++) {
          AccX[i] = i;
          AccY[i] = 0;
        }
      });
    });

    Queue.submit([&](handler &CGH) {
      auto AccX = BufX.get_access();
      auto AccY = BufY.get_access();

      CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
        if (!IH.ext_oneapi_has_graph()) {
          assert(false && "Native Handle should have a graph");
        }
        // Newly created stream for this node
        auto NativeStream = IH.get_native_queue<backend::ext_oneapi_hip>();
        // Graph already created with hipGraphCreate
        HIPGraph NativeGraph =
            IH.ext_oneapi_get_native_graph<backend::ext_oneapi_hip>();

        auto PtrX = IH.get_native_mem<backend::ext_oneapi_hip>(AccX);
        auto PtrY = IH.get_native_mem<backend::ext_oneapi_hip>(AccY);

        // Start stream capture
        // After HIP 6.2 we can use hipStreamBeginCaptureToGraph to capture
        // the stream directly in the native graph, rather than needing to
        // instantiate the stream capture as a new graph.
#if HIP_VERSION
        auto Res =
            hipStreamBeginCapture(NativeStream, NativeGraph, nullptr, nullptr,
                                  0, hipStreamCaptureModeGlobal);
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

#if HIP_VERSION
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
        hipGraphAddChildGraphNode(&Node, NativeGraph, nullptr, 0,
                                  RecordedGraph);
        assert(Res == hipSuccess);
#endif
      });
    });

    Queue.submit([&](handler &CGH) {
      auto AccY = BufY.get_access();
      CGH.parallel_for(range<1>{Size}, [=](id<1> it) { AccY[it] *= 2; });
    });

    Graph.end_recording();

    auto ExecGraph = Graph.finalize();
    Queue.ext_oneapi_graph(ExecGraph).wait();
  }

  auto HostAcc = BufY.get_host_access();
  for (size_t i = 0; i < Size; i++) {
    const int Ref = i * 2;
    assert(check_value(Ref, HostAcc[i],
                       std::string("HostAcc at index ") + std::to_string(i)));
  }

  return 0;
}
