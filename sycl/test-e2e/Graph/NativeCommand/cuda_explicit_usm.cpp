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

  auto NodeA = Graph.add([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrX[i] = i;
        PtrY[i] = 0;
      }
    });
  });

  auto NodeB = Graph.add(
      [&](handler &CGH) {
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
          auto Res = cuGraphAddMemcpyNode(&Node, NativeGraph, nullptr, 0,
                                          &Params, Context);
          assert(Res == CUDA_SUCCESS);
        });
      },
      exp_ext::property::node::depends_on(NodeA));

  Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{Size}, [=](id<1> it) { PtrY[it] *= 2; });
      },
      exp_ext::property::node::depends_on(NodeB));

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
