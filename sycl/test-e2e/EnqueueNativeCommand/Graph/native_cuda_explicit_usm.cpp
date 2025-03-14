// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out %cuda_options
// RUN: %{run} %t.out
// REQUIRES: cuda, cuda_dev_kit

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
      CUgraph NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_cuda>();

      CUDA_MEMCPY3D Params;
      std::memset(&Params, 0, sizeof(CUDA_MEMCPY3D));
      Params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      Params.srcDevice = (CUdeviceptr)PtrX;
      Params.srcHost = nullptr;
      Params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      Params.dstDevice = (CUdeviceptr)PtrY, Params.dstHost = nullptr;
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
