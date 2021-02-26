// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// Sub-groups are not suported on Host
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// Execution on CPU and FPGA takes 100000 times longer
// RUNx: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUNx: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  cl::sycl::queue queue;
  printf("Device Name = %s\n",
         queue.get_device().get_info<cl::sycl::info::device::name>().c_str());

  // Initialize some host memory
  constexpr int N = 64;
  int host_mem[N];
  for (int i = 0; i < N; ++i) {
    host_mem[i] = i * 100;
  }

  // Use the device to transform each value
  {
    cl::sycl::buffer<int, 1> buf(host_mem, N);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto global =
          buf.get_access<cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>(cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local(N, cgh);

      cgh.parallel_for<class test>(
          cl::sycl::nd_range<1>(N, 32), [=](cl::sycl::nd_item<1> it) {
            int v[N] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
            cl::sycl::ONEAPI::sub_group sg = it.get_sub_group();
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

            // Local address space
            auto y = sg.load(&local[i]);

            // CHECK: Sub-group load() is not supported for private pointers.
            auto z = sg.load(v + i);

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
