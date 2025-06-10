// RUN: %{build} -o %t.out %cuda_options
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %cuda_options %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: target-nvidia, cuda_dev_kit
// REQUIRES: aspect-usm_shared_allocations

// Test that when a host-task splits a graph into multiple backend UR
// command-buffers that we use the correct command-buffer for
// the native commands.

#include <cuda.h>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/usm.hpp>

#include <sycl/properties/queue_properties.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  queue Queue{{sycl::property::queue::in_order{}}};

  const size_t Size = 128;
  int *PtrX = malloc_device<int>(Size, Queue);
  int *PtrY = malloc_device<int>(Size, Queue);
  int *PtrZ = malloc_device<int>(Size, Queue);
  int *PtrS = malloc_shared<int>(Size, Queue);

  exp_ext::command_graph Graph{Queue};

  Graph.begin_recording(Queue);

  const int ModValue = 42;
  Queue.submit([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrX[i] = i;
        PtrY[i] = 0;
        PtrZ[i] = 0;
        PtrS[i] = ModValue;
      }
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
      if (!IH.ext_codeplay_has_graph()) {
        assert(false && "Native Handle should have a graph");
      }
      CUgraph NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_cuda>();

      CUDA_MEMCPY3D Params;
      std::memset(&Params, 0, sizeof(CUDA_MEMCPY3D));
      Params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      Params.srcDevice = (CUdeviceptr)PtrX;
      Params.srcHost = nullptr;
      Params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      Params.dstDevice = (CUdeviceptr)PtrY;
      Params.dstHost = nullptr;
      Params.WidthInBytes = Size * sizeof(int);
      Params.Height = 1;
      Params.Depth = 1;

      CUgraphNode Node;
      CUcontext Context = IH.get_native_context<backend::ext_oneapi_cuda>();
      auto Res = cuGraphAddMemcpyNode(&Node, NativeGraph, nullptr, 0, &Params,
                                      Context);
      assert(Res == CUDA_SUCCESS);
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.host_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrS[i] = ModValue * 2;
      }
    });
  });

  Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{Size}, [=](id<1> it) { PtrX[it] = PtrS[it]; });
  });

  Queue.submit([&](handler &CGH) {
    CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
      if (!IH.ext_codeplay_has_graph()) {
        assert(false && "Native Handle should have a graph");
      }
      CUgraph NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_cuda>();

      CUDA_MEMCPY3D Params;
      std::memset(&Params, 0, sizeof(CUDA_MEMCPY3D));
      Params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      Params.srcDevice = (CUdeviceptr)PtrX;
      Params.srcHost = nullptr;
      Params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      Params.dstDevice = (CUdeviceptr)PtrZ;
      Params.dstHost = nullptr;
      Params.WidthInBytes = Size * sizeof(int);
      Params.Height = 1;
      Params.Depth = 1;

      CUgraphNode Node;
      CUcontext Context = IH.get_native_context<backend::ext_oneapi_cuda>();
      auto Res = cuGraphAddMemcpyNode(&Node, NativeGraph, nullptr, 0, &Params,
                                      Context);
      assert(Res == CUDA_SUCCESS);
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecGraph);

  std::vector<int> HostDataY(Size);
  std::vector<int> HostDataZ(Size);
  Queue.copy(PtrY, HostDataY.data(), Size);
  Queue.copy(PtrZ, HostDataZ.data(), Size);
  Queue.wait();
  for (size_t i = 0; i < Size; i++) {
    assert(i == HostDataY[i]);
    assert(ModValue * 2 == HostDataZ[i]);
  }

  free(PtrX, Queue);
  free(PtrY, Queue);
  free(PtrZ, Queue);
  free(PtrS, Queue);

  return 0;
}
