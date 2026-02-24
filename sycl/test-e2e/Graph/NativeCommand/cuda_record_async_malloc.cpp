// RUN: %{build} -o %t.out %cuda_options
// RUN: %{run} %t.out
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
      // Graph already created with cuGraphCreate
      CUgraph NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_cuda>();

      // Start stream capture
      // After CUDA 12.3 we can use cuStreamBeginCaptureToGraph to capture
      // the stream directly in the native graph, rather than needing to
      // instantiate the stream capture as a new graph.
#if CUDA_VERSION >= 12030
      // Newly created stream for this node
      auto NativeStream = IH.get_native_queue<backend::ext_oneapi_cuda>();

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
#else
      // Use explicit graph building API to add alloc/free nodes when
      // cuGraphAddMemFreeNode isn't available
      auto Device = IH.get_native_device<backend::ext_oneapi_cuda>();
      CUDA_MEM_ALLOC_NODE_PARAMS AllocParams{};
      AllocParams.bytesize = Size * sizeof(int32_t);
      AllocParams.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
      AllocParams.poolProps.location.id = Device;
      AllocParams.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      CUgraphNode AllocNode;
      auto Res = cuGraphAddMemAllocNode(&AllocNode, NativeGraph, nullptr, 0,
                                        &AllocParams);
      assert(Res == CUDA_SUCCESS);

      CUdeviceptr PtrAsync = AllocParams.dptr;
      CUDA_MEMSET_NODE_PARAMS MemsetParams{};
      MemsetParams.dst = PtrAsync;
      MemsetParams.elementSize = sizeof(int32_t);
      MemsetParams.height = Size;
      MemsetParams.pitch = sizeof(int32_t);
      MemsetParams.value = Pattern;
      MemsetParams.width = 1;
      CUgraphNode MemsetNode;
      CUcontext Context = IH.get_native_context<backend::ext_oneapi_cuda>();
      Res = cuGraphAddMemsetNode(&MemsetNode, NativeGraph, &AllocNode, 1,
                                 &MemsetParams, Context);
      assert(Res == CUDA_SUCCESS);

      CUDA_MEMCPY3D MemcpyParams{};
      std::memset(&MemcpyParams, 0, sizeof(CUDA_MEMCPY3D));
      MemcpyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      MemcpyParams.srcDevice = PtrAsync;
      MemcpyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      MemcpyParams.dstDevice = (CUdeviceptr)PtrX;
      MemcpyParams.WidthInBytes = Size * sizeof(int32_t);
      MemcpyParams.Height = 1;
      MemcpyParams.Depth = 1;
      CUgraphNode MemcpyNode;
      Res = cuGraphAddMemcpyNode(&MemcpyNode, NativeGraph, &MemsetNode, 1,
                                 &MemcpyParams, Context);
      assert(Res == CUDA_SUCCESS);

      CUgraphNode FreeNode;
      Res = cuGraphAddMemFreeNode(&FreeNode, NativeGraph, &MemcpyNode, 1,
                                  PtrAsync);
      assert(Res == CUDA_SUCCESS);
#endif
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
