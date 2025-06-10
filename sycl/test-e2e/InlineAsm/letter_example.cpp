// REQUIRES: sg-16,aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t problem_size = 16;

class kernel_name;

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
  int *a = (int *)malloc_shared(sizeof(int) * problem_size, Device, ctx);

  for (int i = 0; i < problem_size; i++) {
    a[i] = i;
  }

  q.parallel_for<kernel_name>(
       sycl::range<1>(problem_size),
       [=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(16)]] {
         // The use of if_architecture_is_ge is a precaution in case the test is
         // compiled with the -fsycl-targets flag.
         syclex::if_architecture_is_ge<syclex::architecture::intel_gpu_bmg_g21>(
             []() {})
             .otherwise([&]() {
#if defined(__SYCL_DEVICE_ONLY__)
               int i = idx[0];
               asm volatile(
                   "{\n.decl V52 v_type=G type=d num_elts=16 align=GRF\n"
                   "svm_gather.4.1 (M1, 16) %0.0 V52.0\n"
                   "add(M1, 16) V52(0, 0)<1> V52(0, 0)<1; 1, 0> 0x1:w\n"
                   "svm_scatter.4.1 (M1, 16) %0.0 V52.0\n}"
                   :
                   : "rw"(&a[i]));
#else
               a[idx[0]]++;
#endif
             });
       })
      .wait();

  for (int i = 0; i < problem_size; i++) {
    if (a[i] != (i + 1)) {
      std::cerr << "error in a[" << i << "]=" << a[i] << "!=" << (i + 1)
                << std::endl;
      ++Failed;
    }
  }

  sycl::free(a, ctx);
  return Failed;
}
