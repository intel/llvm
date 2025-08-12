// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} %level_zero_options -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: level_zero, level_zero_dev_kit

// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17847

#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>
#include <sycl/usm.hpp>

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
          ze_command_list_handle_t NativeGraph =
              IH.ext_codeplay_get_native_graph<
                  backend::ext_oneapi_level_zero>();

          auto Res = zeCommandListAppendMemoryCopy(
              NativeGraph, PtrY, PtrX, Size * sizeof(int), nullptr, 0, nullptr);
          assert(Res == ZE_RESULT_SUCCESS);
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
