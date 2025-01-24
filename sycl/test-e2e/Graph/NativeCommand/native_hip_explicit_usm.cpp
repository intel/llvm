// FIXME: the rocm include path and link path are highly platform dependent,
// we should set this with some variable instead.
// RUN: %{build} -Wno-error=deprecated-pragma -o %t.out -I%rocm_path/include -L%rocm_path/lib -lamdhip64
// RUN: %{run} %t.out
// REQUIRES: target-amd

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include "../graph_common.hpp"
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
      if (!IH.ext_oneapi_has_graph()) {
        assert(false && "Native Handle should have a graph");
      }
      // Graph already created with hipGraphCreate
      HIPGraph NativeGraph =
          IH.ext_oneapi_get_native_graph<backend::ext_oneapi_hip>();

      HIPGraphNode Node;
      auto Res = hipGraphAddMemcpyNode1D(&Node, NativeGraph, nullptr, 0,
                                         PtrY, PtrX, Size * sizeof(int), hipMemcpyDefault));

      assert(Res == hipSuccess);
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
