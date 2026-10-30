// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// REQUIRES-INTEL-DRIVER: lin: 40105

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=%{leak_check_not} %}

// Tests cooperative kernels recorded with native recording. Each work-item
// writes In, synchronizes across the whole root group, then reads the value
// written by the matching work-item of the previous work-group. The root-group
// barrier comes either from use_root_sync or from a hand-written atomic barrier
// in a kernel with a concurrent work-group forward progress guarantee.

#include "../../graph_common.hpp"

#include <sycl/atomic_ref.hpp>
#include <sycl/builtins.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/ext/oneapi/experimental/kernel_queue_info.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_barrier.hpp>

template <bool UseCustomRootGroupBarrier> struct RotateKernel {
  int *In;
  int *Out;
  const int *Seed;
  uint32_t *Counter;
  size_t NumGroups;

  void operator()(nd_item<1> Item) const {
    const size_t I = Item.get_global_linear_id();
    const size_t N = Item.get_global_range(0);
    const size_t WGSize = Item.get_local_range(0);
    // Delay the writes of odd work-groups so that the even work-groups reading
    // them see stale data unless the barrier really synchronizes the device.
    if (Item.get_group(0) % 2 == 1) {
      volatile float X = 1.0f;
      for (int K = 0; K < 100; ++K)
        X = X + sycl::sin(X);
    }
    In[I] = static_cast<int>(I) + *Seed;
    if constexpr (UseCustomRootGroupBarrier) {
      group_barrier(Item.get_group(), memory_scope::device);
      if (Item.get_local_linear_id() == 0) {
        atomic_ref<uint32_t, memory_order::acq_rel, memory_scope::device,
                   access::address_space::global_space>
            Arrived{*Counter};
        Arrived.fetch_add(1);
        while (Arrived.load(memory_order::acquire) < NumGroups) {
        }
      }
      group_barrier(Item.get_group(), memory_scope::device);
    } else {
      group_barrier(Item.ext_oneapi_get_root_group());
    }
    Out[I] = In[(I + N - WGSize) % N];
  }

  auto get(exp_ext::properties_tag) const {
    if constexpr (UseCustomRootGroupBarrier)
      return exp_ext::properties{exp_ext::work_group_progress<
          exp_ext::forward_progress_guarantee::concurrent,
          exp_ext::execution_scope::root_group>};
    else
      return exp_ext::properties{exp_ext::use_root_sync};
  }
};

template <bool UseCustomRootGroupBarrier> class RotateKernelName;

template <bool UseCustomRootGroupBarrier> void run() {
  using KernelName = RotateKernelName<UseCustomRootGroupBarrier>;
  queue Queue{property::queue::in_order{}};

  constexpr size_t WGSize = 64;
  auto Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());
  const size_t MaxGroups =
      Bundle.get_kernel<KernelName>()
          .template ext_oneapi_get_info<
              exp_ext::info::kernel_queue_specific::max_num_work_groups>(
              Queue, range<1>{WGSize}, 0);
  assert(MaxGroups >= 2);
  const size_t NumGroups = std::min<size_t>(MaxGroups, 256);
  const size_t N = NumGroups * WGSize;

  int *In = malloc_device<int>(N, Queue);
  int *Out = malloc_device<int>(N, Queue);
  int *Seed = malloc_device<int>(1, Queue);
  uint32_t *Counter = malloc_device<uint32_t>(1, Queue);

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  Graph.begin_recording(Queue);
  Queue.memset(Counter, 0, sizeof(uint32_t));
  Queue.parallel_for<KernelName>(nd_range<1>{N, WGSize},
                                 RotateKernel<UseCustomRootGroupBarrier>{
                                     In, Out, Seed, Counter, NumGroups});
  Graph.end_recording(Queue);

  auto ExecGraph = Graph.finalize();

  std::vector<int> HostOut(N);
  for (int Replay = 0; Replay < 3; ++Replay) {
    const int SeedValue = Replay * 100000;
    Queue.copy(&SeedValue, Seed, 1);
    Queue.memset(Out, 0, N * sizeof(int));
    Queue.ext_oneapi_graph(ExecGraph);
    Queue.copy(Out, HostOut.data(), N).wait();

    for (size_t i = 0; i < N; i++) {
      const int Expected = static_cast<int>((i + N - WGSize) % N) + SeedValue;
      assert(check_value(i, Expected, HostOut[i], "HostOut"));
    }
  }

  free(In, Queue);
  free(Out, Queue);
  free(Seed, Queue);
  free(Counter, Queue);
}

void test(bool UseCustomRootGroupBarrier) {
  if (UseCustomRootGroupBarrier)
    run<true>();
  else
    run<false>();
}

int main() {
  test(/*UseCustomRootGroupBarrier=*/false);
  test(/*UseCustomRootGroupBarrier=*/true);
  return 0;
}
