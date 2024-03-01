// Test1 - check that kernel can call a SYCL_EXTERNAL function defined in a
// different object file.
// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DSOURCE1 %s -c -o %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DSOURCE2 %s -c -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t1.o %t2.o -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <iostream>
#include <sycl/sycl.hpp>

#ifdef SOURCE1
int bar(int b);

SYCL_EXTERNAL
int foo(int a) { return bar(a); }

__attribute((noinline)) int bar(int b) {
#ifdef __SYCL_DEVICE_ONLY__
  return 1;
#else
  return 0;
#endif
}
#endif // SOURCE1

#ifdef SOURCE2
SYCL_EXTERNAL
int foo(int A);

int main(void) {
  constexpr unsigned Size = 4;
  int A[Size] = {1, 2, 3, 4};
  int B[Size] = {1, 2, 3, 4};
  int C[Size];

  {
    sycl::range<1> range{Size};
    sycl::buffer<int, 1> bufA(A, range);
    sycl::buffer<int, 1> bufB(B, range);
    sycl::buffer<int, 1> bufC(C, range);

    sycl::queue().submit([&](sycl::handler &cgh) {
      auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
      auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
      auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          range, [=](sycl::id<1> ID) { accC[ID] = foo(accA[ID]); });
    });
  }

  for (unsigned I = 0; I < Size; ++I) {
    int Ref = foo(A[I]);
    if (C[I] != 1) {
      std::cout << "fail: [" << I << "] == " << C[I] << ", expected " << 1
                << "\n";
      return 1;
    }
  }
  std::cout << "pass\n";
  return 0;
}
#endif // SOURCE2
