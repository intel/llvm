// REQUIRES: sg-32,aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/usm.hpp>

constexpr size_t problem_size = 32;

class kernel_name;

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  sycl::device Device = q.get_device();
  int Failed = 0;

  if (!isInlineASMSupported(Device)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  syclex::architecture CurrentDeviceArch =
      Device.get_info<syclex::info::device::architecture>();
  // This check is carried out because the test is not supported on BMG and
  // subsequent devices.
  if (CurrentDeviceArch >= syclex::architecture::intel_gpu_bmg_g21) {
    std::cout << "This test is not supported on BMG and later. Skipping..."
              << std::endl;
    return 0;
  }

  auto ctx = q.get_context();
  int *a =
      (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  int *b =
      (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  int *c =
      (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  for (int i = 0; i < problem_size; i++) {
    b[i] = -10;
    a[i] = i;
    c[i] = i;
  }

  q.parallel_for<kernel_name>(
       sycl::range<1>(problem_size),
       [=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(32)]] {
         int i = idx[0];
         // The use of if_architecture_is_ge is a precaution in case the test is
         // compiled with the -fsycl-targets flag.
         syclex::if_architecture_is_ge<syclex::architecture::intel_gpu_bmg_g21>(
             []() {})
             .otherwise([&]() {
#if defined(__SYCL_DEVICE_ONLY__)
               asm volatile(R"a(
    {
        .decl V52 v_type=G type=d num_elts=16 align=GRF
        .decl V53 v_type=G type=d num_elts=16 align=GRF
        .decl V54 v_type=G type=d num_elts=16 align=GRF
        .decl V55 v_type=G type=d num_elts=16 align=GRF
        .decl V56 v_type=G type=d num_elts=16 align=GRF
        .decl V57 v_type=G type=d num_elts=16 align=GRF
        svm_gather.4.1 (M1, 16) %2.0 V54.0
        svm_gather.4.1 (M1, 16) %3.0 V55.0
        svm_gather.4.1 (M1, 16) %4.0 V56.0
        svm_gather.4.1 (M1, 16) %5.0 V57.0
        mul (M1, 16) V52(0,0)<1> V54(0,0)<1;1,0> V56(0,0)<1;1,0>
        mul (M1, 16) V53(0,0)<1> V55(0,0)<1;1,0> V57(0,0)<1;1,0>
        svm_scatter.4.1 (M1, 16) %0.0 V52.0
        svm_scatter.4.1 (M1, 16) %1.0 V53.0
    }
    )a" ::"rw"(&b[i]),
                            "rw"(&b[i] + 16), "rw"(&a[i]), "rw"(&a[i] + 16),
                            "rw"(&c[i]), "rw"(&c[i] + 16));
#else
               b[i] = a[i] * c[i];
#endif
             });
       })
      .wait();

  for (int i = 0; i < problem_size; i++) {
    if (b[i] != a[i] * c[i]) {
      std::cerr << "error in a[" << i << "]=" << b[i] << "!=" << a[i] * c[i]
                << std::endl;
      ++Failed;
    }
  }

  sycl::free(a, ctx);
  sycl::free(b, ctx);
  sycl::free(c, ctx);

  return Failed;
}
