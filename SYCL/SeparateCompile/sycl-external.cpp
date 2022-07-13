// Test1 - check that kernel can call a SYCL_EXTERNAL function defined in a
// different object file.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DSOURCE1 -c %s -o %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DSOURCE2 -c %s -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %t1.o %t2.o -o %t.exe
// RUN: %CPU_RUN_PLACEHOLDER %t.exe
// RUN: %GPU_RUN_PLACEHOLDER %t.exe
// RUN: %ACC_RUN_PLACEHOLDER %t.exe
//
// Test2 - check that kernel can call a SYCL_EXTERNAL function defined in a
// static library.
// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %t2.o -foffload-static-lib=%t.a -o %t.exe
// RUN: %CPU_RUN_PLACEHOLDER %t.exe
// RUN: %GPU_RUN_PLACEHOLDER %t.exe
// RUN: %ACC_RUN_PLACEHOLDER %t.exe
//
// Linking issues with HIP AMD
// XFAIL: hip_amd

#include <iostream>
#include <sycl/sycl.hpp>

#ifdef SOURCE1
int bar(int b);

SYCL_EXTERNAL
int foo(int a, int b) { return a + bar(b); }

int bar(int b) { return b + 5; }
#endif // SOURCE1

#ifdef SOURCE2
SYCL_EXTERNAL
int foo(int A, int B);

int main(void) {
  constexpr unsigned Size = 4;
  int A[Size] = {1, 2, 3, 4};
  int B[Size] = {1, 2, 3, 4};
  int C[Size];

  {
    cl::sycl::range<1> range{Size};
    cl::sycl::buffer<int, 1> bufA(A, range);
    cl::sycl::buffer<int, 1> bufB(B, range);
    cl::sycl::buffer<int, 1> bufC(C, range);

    cl::sycl::queue().submit([&](cl::sycl::handler &cgh) {
      auto accA = bufA.get_access<cl::sycl::access::mode::read>(cgh);
      auto accB = bufB.get_access<cl::sycl::access::mode::read>(cgh);
      auto accC = bufC.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(range, [=](cl::sycl::id<1> ID) {
        accC[ID] = foo(accA[ID], accB[ID]);
      });
    });
  }

  for (unsigned I = 0; I < Size; ++I) {
    int Ref = foo(A[I], B[I]);
    if (C[I] != Ref) {
      std::cout << "fail: [" << I << "] == " << C[I] << ", expected " << Ref
                << "\n";
      return 1;
    }
  }
  std::cout << "pass\n";
  return 0;
}
#endif // SOURCE2
