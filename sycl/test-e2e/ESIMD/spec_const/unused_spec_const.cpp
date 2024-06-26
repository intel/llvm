// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that if specialization constant is set but not used, then
// 1) The kernel still works properly;
// 2) The kernel parameter holding the buffer for specialization constants is
//    optimized away.
//    TODO: the second part of the check should be added to this test when
//    DAE (Dead Arguments Elimination) optimization is enabled for ESIMD.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

constexpr sycl::specialization_id<int> Spec;

int main() {
  sycl::queue Q;
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  int *A = sycl::malloc_shared<int>(1, Q);
  *A = 123;
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_specialization_constant<Spec>(42);
     CGH.single_task([=](sycl::kernel_handler KH) SYCL_ESIMD_KERNEL {
       // This use of get_specializaiton_constant is commented out.
       // Thus the device code doen't have info about it and cannot allocate
       // memory/buffer for it on device side.
       // The test checks that the unused kernel parameter is either
       // eliminated by DAE optimization OR nullptr is passed for that argument.

       // KH.get_specialization_constant<Spec>();
       sycl::ext::intel::esimd::simd<int, 1> V(A);
       V += 2;
       V.copy_to(A);
     });
   }).wait();

  int AVal = *A;
  sycl::free(A, Q);

  if (AVal != 125) {
    std::cout << "Failed: " << AVal << " != 125" << std::endl;
    return 1;
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
