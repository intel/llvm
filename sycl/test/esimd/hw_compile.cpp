// Basic ESIMD test which checks that ESIMD invocation syntax can get compiled.
// RUN: %clangxx -fsycl -fsycl-device-only -c %s -o %t.bc

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

int main(void) {
  constexpr unsigned Size = 4;
  int A[Size] = {1, 2, 3, 4};
  int B[Size] = {1, 2, 3, 4};
  int C[Size];

  {
    cl::sycl::range<1> UnitRange{1};
    cl::sycl::buffer<int, 1> bufA(A, UnitRange);
    cl::sycl::buffer<int, 1> bufB(B, UnitRange);
    cl::sycl::buffer<int, 1> bufC(C, UnitRange);

    cl::sycl::queue().submit([&](cl::sycl::handler &cgh) {
      auto accA = bufA.get_access<cl::sycl::access::mode::read>(cgh);
      auto accB = bufB.get_access<cl::sycl::access::mode::read>(cgh);
      auto accC = bufC.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          UnitRange * UnitRange, [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
            int off = i.get(0) * Size;
            // those operations below would normally be
            // represented as a single vector operation
            // through ESIMD vector
            cl::sycl::ext::intel::esimd::simd<int, Size> A(accA,
                                                           off * sizeof(int));
            cl::sycl::ext::intel::esimd::simd<int, Size> B(accB,
                                                           off * sizeof(int));
            cl::sycl::ext::intel::esimd::simd<int, Size> C = A + B;
            C.copy_to(accC, off * sizeof(int));
          });
    });
  }

  return 0;
}
