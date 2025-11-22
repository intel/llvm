// RUN: %{build} -fprofile-instr-generate -fcoverage-mapping -o %t.out
// RUN: %{run} LLVM_PROFILE_FILE=%t.profraw %t.out
// RUN: %{run-aux} llvm-profdata merge %t.profraw -o %t.profdata
// RUN: %{run-aux} llvm-cov show -instr-profile=%t.profdata %t.out -name="main" | FileCheck %s

#include <sycl/usm.hpp>

int main() {
  sycl::queue q;
  int *values = sycl::malloc_shared<int>(10, q);
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
       if (idx[0] < 8)
         values[idx] = 42;
       else
         values[idx] = 7;
     });
   }).wait();
  for (int i = 0; i < 10; i++)
    assert(values[i] == (i < 8 ? 42 : 7));
  sycl::free(values, q);
  return 0;
}

// REQUIRES: target-spir
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287
// UNSUPPORTED: windows
// UNSUPPORTED-INTENDED: On Windows, compiler-rt requires /MT but the flag
//                       cannot be used with SYCL.

// CHECK: main:
// CHECK:     8|      1|int main() {
// CHECK:     9|      1|  sycl::queue q;
// CHECK:    10|      1|  int *values = sycl::malloc_shared<int>(10, q);
// CHECK:    11|      1|  q.submit([&](sycl::handler &h) {
// CHECK:    12|      1|     h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
// CHECK:    13|      1|       if (idx[0] < 8)
// CHECK:    14|      1|         values[idx] = 42;
// CHECK:    15|      1|       else
// CHECK:    16|      1|         values[idx] = 7;
// CHECK:    17|      1|     });
// CHECK:    18|      1|   }).wait();
// CHECK:    19|     11|  for (int i = 0; i < 10; i++)
// CHECK:    20|     10|    assert(values[i] == (i < 8 ? 42 : 7));
// CHECK:    21|      1|  sycl::free(values, q);
// CHECK:    22|      1|  return 0;
// CHECK:    23|      1|}
// CHECK: device_code_coverage.cpp:_ZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_:
// CHECK:    11|      1|  q.submit([&](sycl::handler &h) {
// CHECK:    12|      1|     h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
// CHECK:    13|      1|       if (idx[0] < 8)
// CHECK:    14|      1|         values[idx] = 42;
// CHECK:    15|      1|       else
// CHECK:    16|      1|         values[idx] = 7;
// CHECK:    17|      1|     });
// CHECK:    18|      1|   }).wait();
// CHECK: device_code_coverage.cpp:_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_:
// CHECK:    12|     10|     h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
// CHECK:    13|     10|       if (idx[0] < 8)
// CHECK:    14|      8|         values[idx] = 42;
// CHECK:    15|      2|       else
// CHECK:    16|      2|         values[idx] = 7;
// CHECK:    17|     10|     });
