// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -S -emit-llvm  -o - %s | FileCheck %s

// Checks that the subhandler is correctly emitted in the module
#include <CL/sycl.hpp>

#include <cstdlib>
#include <iostream>
const size_t N = 10;

template <typename T> class init_a;
using namespace sycl;

template <typename T> void gen_test(queue myQueue) {
    buffer<float, 1> a(range<1>{N});
    const T test = rand() % 10;

    myQueue.submit([&](handler &cgh) {
      auto A = a.get_access<access::mode::write>(cgh);
      cgh.parallel_for<init_a<T>>(range<1>{N},
                                  [=](id<1> index) { A[index] = test; });
    }).wait();
}


void test() {
  queue q;
  gen_test<int>(q);
//CHECK:  define weak void @_Z6init_aIiEsubhandler(ptr %0, ptr %1) #2 {
//CHECK-NEXT:entry:
//CHECK-NEXT:  %2 = getelementptr %0, ptr %0, i64 0
//CHECK-NEXT:  %3 = load ptr, ptr %2, align 8
//CHECK-NEXT:  %4 = getelementptr %0, ptr %0, i64 3
//CHECK-NEXT:  %5 = load ptr, ptr %4, align 8
//CHECK-NEXT:  %6 = getelementptr %0, ptr %0, i64 4
//CHECK-NEXT:  %7 = load ptr, ptr %6, align 8
//CHECK-NEXT:  %8 = load i32, ptr %7, align 4
//CHECK-NEXT:  call void @_Z6init_aIiE_ncpu(ptr %3, ptr %5, i32 %8, ptr %1)
//CHECK-NEXT:  ret void
//CHECK-NEXT:}
  gen_test<float>(q);
//CHECK:  define weak void @_Z6init_aIfEsubhandler(ptr %0, ptr %1) #2 {
//CHECK-NEXT:entry:
//CHECK-NEXT:  %2 = getelementptr %0, ptr %0, i64 0
//CHECK-NEXT:  %3 = load ptr, ptr %2, align 8
//CHECK-NEXT:  %4 = getelementptr %0, ptr %0, i64 3
//CHECK-NEXT:  %5 = load ptr, ptr %4, align 8
//CHECK-NEXT:  %6 = getelementptr %0, ptr %0, i64 4
//CHECK-NEXT:  %7 = load ptr, ptr %6, align 8
//CHECK-NEXT:  %8 = load float, ptr %7, align 4
//CHECK-NEXT:  call void @_Z6init_aIfE_ncpu(ptr %3, ptr %5, float %8, ptr %1)
//CHECK-NEXT:  ret void
//CHECK-NEXT:} 
}
