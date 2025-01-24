// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// REQUIRES: level_zero, level_zero_dev_kit

#include <level_zero/ze_api.h>
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
      ze_command_list_handle_t NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();

      auto Res = zeCommandListAppendMemoryCopy(
          NativeGraph, PtrY, PtrX, Size * sizeof(int), nullptr, 0, nullptr);
      assert(Res == ZE_RESULT_SUCCESS);
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
