// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// RUN: %{build} -DUSE_DEPRECATED_LOCAL_ACC -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: ze_debug

#include "helper.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  sycl::queue queue;
  printf("Device Name = %s\n",
         queue.get_device().get_info<sycl::info::device::name>().c_str());

  // Initialize some host memory
  constexpr int N = 64;
  sycl::vec<int, 2> host_mem[N];
  for (int i = 0; i < N; ++i) {
    host_mem[i].s0() = i;
    host_mem[i].s1() = 0;
  }

  // Use the device to transform each value
  {
    sycl::buffer<sycl::vec<int, 2>, 1> buf(host_mem, N);
    queue.submit([&](sycl::handler &cgh) {
      auto global = buf.get_access<sycl::access::mode::read_write,
                                   sycl::access::target::device>(cgh);
#ifdef DUSE_DEPRECATED_LOCAL_ACC
      sycl::accessor<sycl::vec<int, 2>, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local(N, cgh);
#else
      sycl::local_accessor<sycl::vec<int, 2>, 1> local(N, cgh);
#endif
      cgh.parallel_for<class test>(
          sycl::nd_range<1>(N, 32), [=](sycl::nd_item<1> it) {
            sycl::sub_group sg = it.get_sub_group();
            if (!it.get_local_id(0)) {
              int end = it.get_global_id(0) + it.get_local_range()[0];
              for (int i = it.get_global_id(0); i < end; i++) {
                local[i].s0() = 0;
                local[i].s1() = i;
              }
            }
            it.barrier();

            int i = (it.get_global_id(0) / sg.get_local_range()[0]) *
                    sg.get_local_range()[0];
            // Global address space
            auto x = sg.load(&global[i]);
            auto x_cv1 = sg.load<const volatile sycl::int2>(&global[i]);
            auto x_cv2 = sg.load(
                sycl::global_ptr<const volatile sycl::int2>(&global[i]));

            // Local address space
            auto y = sg.load(&local[i]);
            auto y_cv1 = sg.load<const volatile sycl::int2>(&local[i]);
            auto y_cv2 =
                sg.load(sycl::local_ptr<const volatile sycl::int2>(&local[i]));

            // Store result only if same for non-cv and cv
            if (utils<int, 2>::cmp_vec(x, x_cv1) &&
                utils<int, 2>::cmp_vec(x, x_cv2) &&
                utils<int, 2>::cmp_vec(y, y_cv1) &&
                utils<int, 2>::cmp_vec(y, y_cv2))
              sg.store(&global[i], x + y);
          });
    });
  }

  // Print results and tidy up
  for (int i = 0; i < N; ++i) {
    if (i != host_mem[i].s0() || i != host_mem[i].s1()) {
      printf("Unexpected result [%02d,%02d] vs [%02d,%02d]\n", i, i,
             host_mem[i].s0(), host_mem[i].s1());
      return 1;
    }
  }
  printf("Success!\n");
  return 0;
}
