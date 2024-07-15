// RUN: %{build} -o %t.out
// RUN: %t.out

#include <stdlib.h>
#include <sycl/detail/core.hpp>
#include <vector>

using namespace sycl;
using namespace std;

// This test checks the ext_oneapi_prod extension. This extension introduces
// only one new function, queue::ext_oneapi_prod(), which serves as a hint to
// the compiler to flush the queue. Since it is simply a hint, we cannot really
// test what the backend is doing but we can at least make sure that code
// involving this function compiles and runs successfully in various contexts.
int main() {
  sycl::queue q;

  // Test on an empty queue multiple times.
  q.ext_oneapi_prod();
  q.ext_oneapi_prod();

  // Test on a queue after we've submitted a kernel in various contexts.
  q.single_task([]() {});
  q.ext_oneapi_prod();

  q.parallel_for(range<1>{}, [=](auto &idx) {});
  q.ext_oneapi_prod();
  q.wait();

  // Test that the result of an in-progress addition kernel is not affected by
  // calling ext_oneapi_prod.
  srand(time(0));
  constexpr int N = 16;
  int A[N];
  int B[N];
  int add[N];
  int mult[N];
  for (int i = 0; i < N; ++i) {
    A[i] = rand();
    B[i] = rand();
  }
  {
    buffer<int> bufA{A, N};
    buffer<int> bufB{B, N};
    buffer<int> bufadd{add, N};

    q.submit([&](handler &cgh) {
      accessor accA{bufA, cgh};
      accessor accB{bufB, cgh};
      accessor accadd{bufadd, cgh};
      cgh.parallel_for(N, [=](id<1> i) { accadd[i] = accA[i] + accB[i]; });
    });
    q.ext_oneapi_prod();
  }
  for (int i = 0; i < N; ++i) {
    assert(add[i] == (A[i] + B[i]));
  }

  // Test that the result of an in-progress addition and multiplication kernel
  // is not affected by calling ext_oneapi_prod
  {
    buffer<int> bufA{A, N};
    buffer<int> bufB{B, N};
    buffer<int> bufadd{add, N};
    buffer<int> bufmult{mult, N};
    q.submit([&](handler &cgh) {
      accessor accA{bufA, cgh};
      accessor accB{bufB, cgh};
      accessor accadd{bufadd, cgh};
      cgh.parallel_for(N, [=](id<1> i) { accadd[i] = accA[i] + accB[i]; });
    });

    q.submit([&](handler &cgh) {
      accessor accA{bufA, cgh};
      accessor accB{bufB, cgh};
      accessor accmult{bufmult, cgh};
      cgh.parallel_for(N, [=](id<1> i) { accmult[i] = accA[i] * accB[i]; });
    });
    q.ext_oneapi_prod();
  }
  for (int i = 0; i < N; ++i) {
    assert(add[i] == (A[i] + B[i]) && mult[i] == (A[i] * B[i]));
  }
}
