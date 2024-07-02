// Test1 - check that kernel can call a SYCL_EXTERNAL function defined in a
// different object file.
// RUN: %{build} -DSOURCE1 -c -o %t1.o
// RUN: %{build} -DSOURCE2 -c -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %t1.o %t2.o -o %t.exe
// RUN: %{run} %t.exe
//
// Test2 - check that kernel can call a SYCL_EXTERNAL function defined in a
// static library.
// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %t2.o %t.a -o %t.exe
// RUN: %{run} %t.exe

#include <iostream>
#include <sycl/detail/core.hpp>

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
    sycl::range<1> range{Size};
    sycl::buffer<int, 1> bufA(A, range);
    sycl::buffer<int, 1> bufB(B, range);
    sycl::buffer<int, 1> bufC(C, range);

    sycl::queue().submit([&](sycl::handler &cgh) {
      auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
      auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
      auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Test>(
          range, [=](sycl::id<1> ID) { accC[ID] = foo(accA[ID], accB[ID]); });
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
