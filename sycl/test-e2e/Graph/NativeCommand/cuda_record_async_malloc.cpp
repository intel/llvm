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
  // Test is only expected to pass after CUDA 12.9
  // See SYCL-Graph design document on CUDA native-command support
  int CudaDriverVersion = 0;
  cuDriverGetVersion(&CudaDriverVersion);
  if (CudaDriverVersion < 12090) {
    return 0;
  }

  queue Queue;

  const size_t Size = 128;
  int32_t *PtrX = malloc_device<int32_t>(Size, Queue);

  exp_ext::command_graph Graph{Queue};

  Graph.begin_recording(Queue);

  const int32_t Pattern = 42;
  Queue.submit([&](handler &CGH) {
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
      auto Res = cuStreamBeginCaptureToGraph(NativeStream, NativeGraph, nullptr,
                                             nullptr, 0,
                                             CU_STREAM_CAPTURE_MODE_GLOBAL);
      assert(Res == CUDA_SUCCESS);

      // Add asynchronous malloc node
      CUdeviceptr PtrAsync;
      Res = cuMemAllocAsync(&PtrAsync, Size * sizeof(int32_t), NativeStream);
      assert(Res == CUDA_SUCCESS);

      // Fill async allocation
      Res = cuMemsetD32Async(PtrAsync, Pattern, Size, NativeStream);
      assert(Res == CUDA_SUCCESS);

      // Add memcopy node to USM allocation
      Res = cuMemcpyAsync((CUdeviceptr)PtrX, PtrAsync, Size * sizeof(int32_t),
                          NativeStream);
      assert(Res == CUDA_SUCCESS);

      Res = cuMemFreeAsync(PtrAsync, NativeStream);
      assert(Res == CUDA_SUCCESS);

      Res = cuStreamEndCapture(NativeStream, &NativeGraph);
      assert(Res == CUDA_SUCCESS);
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int32_t> HostData(Size);
  Queue.copy(PtrX, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(Pattern == HostData[i]);
  }

  free(PtrX, Queue);

  return 0;
}
