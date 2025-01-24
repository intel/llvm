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
        CUgraph NativeGraph = IH.get_native_graph<backend::ext_oneapi_cuda>();
        CUgraphNode Node;
        // TODO figure this out
        /*
        CUDA_MEMCPY3D &Params
   181   std::memset(&Params, 0, sizeof(CUDA_MEMCPY3D));
   182
   183   Params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
   184   Params.srcDevice = SrcType == 
   185                          ? *static_cast<const CUdeviceptr *>(SrcPtr)
   186                          : 0;
   187   Params.srcHost = cType == CU_MEMORYTYPE_HOST ? SrcPtr : nullptr;
   188   Params.dstMemoryType = DstType;
   189   Params.dstDevice =
   190       DstType == CU_MEMORYTYPE_DEVICE ? *static_cast<CUdeviceptr *>(DstPtr) : 0;
   191   Params.dstHost = DstType == CU_MEMORYTYPE_HOST ? DstPtr : nullptr;
   192   Params.WidthInBytes = Size;
   193   Params.Height = 1;
   194   Params.Depth = 1;
        auto Res = cuGraphAddMemcpyNode1D(&Node, NativeGraph, nullptr, 0,
                                          ((CUdeviceptr)PtrY, (CUdeviceptr)PtrX,
                                          Size * sizeof(int), &Params, Context);
        assert(Res == CUDA_SUCCESS);
        */
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
