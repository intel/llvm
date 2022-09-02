// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// Sub-groups are not suported on Host
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DUSE_DEPRECATED_LOCAL_ACC %s -o %t.out
// Sub-groups are not suported on Host
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_GenericCastToPtrExplicit_ToLocal,
// __spirv_SubgroupInvocationId, __spirv_GenericCastToPtrExplicit_ToGlobal,
// __spirv_SubgroupBlockReadINTEL, __assert_fail,
// __spirv_SubgroupBlockWriteINTEL on AMD
// error message `Barrier is not supported on the host device yet.` on Nvidia.
// XFAIL: hip_amd || hip_nvidia
// UNSUPPORTED: ze_debug-1,ze_debug4

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {
  sycl::queue queue;
  printf("Device Name = %s\n",
         queue.get_device().get_info<sycl::info::device::name>().c_str());

  // Initialize some host memory
  constexpr int N = 64;
  int host_mem[N];
  for (int i = 0; i < N; ++i) {
    host_mem[i] = i * 100;
  }

  // Use the device to transform each value
  {
    sycl::buffer<int, 1> buf(host_mem, N);
    queue.submit([&](sycl::handler &cgh) {
      auto global = buf.get_access<sycl::access::mode::read_write,
                                   sycl::access::target::device>(cgh);
#ifdef USE_DEPRECATED_LOCAL_ACC
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local(N, cgh);
#else
      sycl::local_accessor<int, 1> local(N, cgh);
#endif

      cgh.parallel_for<class test>(
          sycl::nd_range<1>(N, 32), [=](sycl::nd_item<1> it) {
            sycl::ext::oneapi::sub_group sg = it.get_sub_group();
            if (!it.get_local_id(0)) {
              int end = it.get_global_id(0) + it.get_local_range()[0];
              for (int i = it.get_global_id(0); i < end; i++) {
                local[i] = i;
              }
            }
            it.barrier();

            int i = (it.get_global_id(0) / sg.get_max_local_range()[0]) *
                    sg.get_max_local_range()[0];
            // Global address space
            auto x = sg.load(&global[i]);
            auto x_cv = sg.load<const volatile int>(&global[i]);

            // Local address space
            auto y = sg.load(&local[i]);
            auto y_cv = sg.load<const volatile int>(&local[i]);

            // Store result only if same for non-cv and cv
            if (x == x_cv && y == y_cv)
              sg.store(&global[i], x + y);
          });
    });
  }

  // Print results and tidy up
  for (int i = 0; i < N; ++i) {
    if (i * 101 != host_mem[i]) {
      printf("Unexpected result %04d vs %04d\n", i * 101, host_mem[i]);
      return 1;
    }
  }
  printf("Success!\n");
  return 0;
}
