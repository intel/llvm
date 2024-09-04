// Check that, when linking with a static library, SYCL_EXTERNAL device code
// is preserved despite optimizations.
// RUN: %{build} -O3 -DSOURCE1 -c -o %t1.o
// RUN: %{build} -O3 -DSOURCE2 -c -o %t2.o
// RUN: %{build} -O3 -DSOURCE3 -c -o %t3.o
// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t1.o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -O3 %t3.o %t.a -o %t1.exe
// RUN: %{run} %t1.exe

// Check the repacked case as it can behave differently.
// RUN: echo create %t_repacked.a > %t.txt
// RUN: echo addlib %t.a >> %t.txt
// RUN: echo save >> %t.txt
// RUN: cat %t.txt | llvm-ar -M
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -O3 %t3.o %t_repacked.a -o %t2.exe
// RUN: %{run} %t2.exe

#include <iostream>
#include <sycl/detail/core.hpp>

#ifdef SOURCE1
int local_f2(int b);

SYCL_EXTERNAL
int external_f1(int a, int b) { return a + local_f2(b); }

int local_f2(int b) { return b + 5; }
#endif // SOURCE1

#ifdef SOURCE2
SYCL_EXTERNAL
int external_f1(int A, int B);

void hostf(unsigned Size, sycl::buffer<int, 1> &bufA,
           sycl::buffer<int, 1> &bufB, sycl::buffer<int, 1> &bufC) {
  sycl::range<1> range{Size};
  sycl::queue().submit([&](sycl::handler &cgh) {
    auto accA = bufA.get_access<sycl::access::mode::read>(cgh);
    auto accB = bufB.get_access<sycl::access::mode::read>(cgh);
    auto accC = bufC.get_access<sycl::access::mode::write>(cgh);

    cgh.parallel_for<class Test>(range, [=](sycl::id<1> ID) {
      accC[ID] = external_f1(accA[ID], accB[ID]);
    });
  });
}
#endif

#ifdef SOURCE3
extern void hostf(unsigned Size, sycl::buffer<int, 1> &bufA,
                  sycl::buffer<int, 1> &bufB, sycl::buffer<int, 1> &c);
int ref(int a, int b) { return a + b + 5; }

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
    hostf(Size, bufA, bufB, bufC);
  }
  for (unsigned I = 0; I < Size; ++I) {
    int Ref = ref(A[I], B[I]);
    if (C[I] != Ref) {
      std::cout << "fail: [" << I << "] == " << C[I] << ", expected " << Ref
                << "\n";
      return 1;
    }
  }
  std::cout << "pass\n";
  return 0;
}
#endif // SOURCE3
