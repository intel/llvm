// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -S -emit-llvm  -o - %s | FileCheck %s

// Checks that the subhandler is correctly emitted in the module
#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iostream>
const size_t N = 10;

template <typename T> class init_a;
class Test1;
using namespace sycl;

template <typename T> void gen_test(queue myQueue) {
  buffer<float, 1> a(range<1>{N});
  const T test = rand() % 10;

  myQueue
      .submit([&](handler &cgh) {
        auto A = a.get_access<access::mode::write>(cgh);
        cgh.parallel_for<init_a<T>>(range<1>{N},
                                    [=](id<1> index) { A[index] = test; });
      })
      .wait();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void launch(const Func &kernelFunc) {
  kernelFunc();
}

void test() {
  queue q;
  gen_test<int>(q);
  //CHECK:  define weak void @_ZTS6init_aIiE_NativeCPUKernelsubhandler(ptr %0, ptr %1) #2 {
  //CHECK:       %{{.*}} = getelementptr %0, ptr %0, i64 0
  //CHECK-NEXT:  %[[ARG1:.*]] = load ptr, ptr %2, align 8
  //CHECK-NEXT:  %{{.*}} = getelementptr %0, ptr %0, i64 3
  //CHECK-NEXT:  %[[ARG2:.*]] = load ptr, ptr %4, align 8
  //CHECK-NEXT:  %{{.*}} = getelementptr %0, ptr %0, i64 4
  //CHECK-NEXT:  %{{.*}} = load ptr, ptr %6, align 8
  //CHECK-NEXT:  %[[ARG3:.*]] = load i32, ptr %7, align 4
  //CHECK-NEXT:  call void @_ZTS6init_aIiE_NativeCPUKernel_NativeCPUKernel(ptr %[[ARG1]], ptr %[[ARG2]], i32 %[[ARG3]], ptr %1)
  //CHECK-NEXT:  ret void
  //CHECK-NEXT:}
  gen_test<float>(q);
  //CHECK:  define weak void @_ZTS6init_aIfE_NativeCPUKernelsubhandler(ptr %0, ptr %1) #2 {
  //CHECK:       %{{.*}} = getelementptr %0, ptr %0, i64 0
  //CHECK-NEXT:  %[[ARGF1:.*]] = load ptr, ptr %2, align 8
  //CHECK-NEXT:  %{{.*}} = getelementptr %0, ptr %0, i64 3
  //CHECK-NEXT:  %[[ARGF2:.*]] = load ptr, ptr %4, align 8
  //CHECK-NEXT:  %{{.*}} = getelementptr %0, ptr %0, i64 4
  //CHECK-NEXT:  %{{.*}} = load ptr, ptr %6, align 8
  //CHECK-NEXT:  %[[ARGF3:.*]] = load float, ptr %7, align 4
  //CHECK-NEXT:  call void @_ZTS6init_aIfE_NativeCPUKernel_NativeCPUKernel(ptr %[[ARGF1]], ptr %[[ARGF2]], float %[[ARGF3]], ptr %1)
  //CHECK-NEXT:  ret void
  //CHECK-NEXT:}

  // Check that subhandler is emitted correctly for kernels with no
  // args:deviceQueue.submit([&](sycl::handler &h) {
  sycl::accessor<int, 1, sycl::access::mode::write> acc;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<Test1>(range<1>(1), [=](sycl::id<1> id) {
      acc[id[0]]; // all kernel arguments are removed
    });
  });
  //CHECK:define weak void @_ZTS5Test1_NativeCPUKernelsubhandler(ptr %0, ptr %1) #2 {
  //CHECK:       call void @_ZTS5Test1_NativeCPUKernel_NativeCPUKernel(ptr %1)
  //CHECK-NEXT:  ret void
  //CHECK-NEXT:}

  launch<class TestKernel>([]() {});
  //CHECK:define weak void @_ZTSZ4testvE10TestKernel_NativeCPUKernelsubhandler(ptr %0, ptr %1) #2 {
  //CHECK:       call void @_ZTSZ4testvE10TestKernel_NativeCPUKernel_NativeCPUKernel(ptr %1)
  //CHECK-NEXT:  ret void
  //CHECK-NEXT:}
}
