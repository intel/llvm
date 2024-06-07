// REQUIRES: linux
//
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so

// RUN: %{build} -DFOO_FIRST -L%T -o %t.out -l%basename_t -Wl,-rpath=%T
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-FIRST,CHECK --implicit-check-not=piProgramBuild

// RUN: %{build} -L%T -o %t.out -l%basename_t -Wl,-rpath=%T
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-LAST,CHECK --implicit-check-not=piProgramBuild

#include <sycl/detail/core.hpp>

#include <iostream>

#ifdef BUILD_LIB
int foo() {
  sycl::queue q;
  sycl::buffer<int, 1> b(1);
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(b, cgh);
     using AccTy = decltype(acc);
     struct Kernel {
       void operator()() const { acc[0] = 1; }
       AccTy acc;
     } k = {acc};
     cgh.single_task(k);
   }).wait();
  auto val = sycl::host_accessor(b)[0];
  std::cout << "Foo: " << val << std::endl;
  return val;
}
#else
int foo();
void run() {
  sycl::queue q;
  sycl::buffer<int, 1> b(1);
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(b, cgh);
     using AccTy = decltype(acc);
     // Use different name to avoid ODR-violation.
     struct Kernel2 {
       void operator()() const { acc[0] = 2; }
       AccTy acc;
     } k = {acc};
     cgh.single_task(k);
   }).wait();
  auto val = sycl::host_accessor(b)[0];
  std::cout << "Main: " << val << std::endl;
  assert(val == 2);
}
int main() {
#ifdef FOO_FIRST
  // CHECK-FIRST: piProgramBuild
  // CHECK-FIRST: Foo: 1
  // CHECK-FIRST: Foo: 1
  assert(foo() == 1);
  assert(foo() == 1);
#endif
  // CHECK: piProgramBuild
  // CHECK: Main: 2
  // CHECK: Main: 2
  run();
  run();
#ifndef FOO_FIRST
  // CHECK-LAST: piProgramBuild
  // CHECK-LAST: Foo: 1
  // CHECK-LAST: Foo: 1
  assert(foo() == 1);
  assert(foo() == 1);
#endif
  return 0;
}
#endif
