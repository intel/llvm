// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} %level_zero_options -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: level_zero, level_zero_dev_kit

// Tests that the optimization to use the L0 Copy Engine for memory commands
// does synchronizes correctly with the native commands.
//
// REQUIRES: aspect-usm_host_allocations

// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17847

#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/usm.hpp>

#include <sycl/properties/queue_properties.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  // Initialize Level Zero driver is required if this test is linked
  // statically with Level Zero loader, the driver will not be init otherwise.
  ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  if (result != ZE_RESULT_SUCCESS) {
    std::cout << "zeInit failed\n";
    return 1;
  }

  queue Queue{{sycl::property::queue::in_order{}}};

  const size_t Size = 128;
  int *PtrD1 = malloc_device<int>(Size, Queue);
  int *PtrD2 = malloc_device<int>(Size, Queue);
  int *PtrH1 = malloc_host<int>(Size, Queue);
  int *PtrH2 = malloc_host<int>(Size, Queue);

  exp_ext::command_graph Graph{Queue};

  Graph.begin_recording(Queue);

  Queue.submit([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        PtrD1[i] = i;
        PtrD2[i] = 0;
      }
    });
  });

  Queue.memset(PtrH1, 0, Size * sizeof(int));
  Queue.memset(PtrH2, 0, Size * sizeof(int));

  Queue.submit([&](handler &CGH) {
    CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
      if (!IH.ext_codeplay_has_graph()) {
        assert(false && "Native Handle should have a graph");
      }
      ze_command_list_handle_t NativeGraph =
          IH.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();

      auto Res = zeCommandListAppendMemoryCopy(
          NativeGraph, PtrH1, PtrD1, Size * sizeof(int), nullptr, 0, nullptr);
      assert(Res == ZE_RESULT_SUCCESS);

      int Pattern = 42;
      Res = zeCommandListAppendMemoryFill(NativeGraph, PtrD2, &Pattern,
                                          sizeof(int), Size * sizeof(int),
                                          nullptr, 0, nullptr);
      assert(Res == ZE_RESULT_SUCCESS);
    });
  });

  Queue.copy(PtrD2, PtrH2, Size);

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue.ext_oneapi_graph(ExecGraph).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(i == PtrH1[i]);
    assert(42 == PtrH2[i]);
  }

  free(PtrD1, Queue);
  free(PtrD2, Queue);

  free(PtrH1, Queue);
  free(PtrH2, Queue);

  return 0;
}
