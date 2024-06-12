// REQUIRES: linux
//
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so

// DEFINE: %{compile} = %{build} -DFNAME=%basename_t -o %t.out -ldl -Wl,-rpath=%T

// RUN: %{compile} -DRUN_FIRST
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-FIRST,CHECK --implicit-check-not=piProgramBuild

// RUN: %{compile} -DRUN_MIDDLE_BEFORE
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-MIDDLE-BEFORE,CHECK --implicit-check-not=piProgramBuild

// RUN: %{compile} -DRUN_MIDDLE_AFTER
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-MIDDLE-AFTER,CHECK --implicit-check-not=piProgramBuild

// clang-format off
// This causes SEG. FAULT.
// RUNx: %{compile} -DRUN_LAST
// RUNx: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-LAST,CHECK --implicit-check-not=piProgramBuild
// clang-format on

#include <sycl/detail/core.hpp>

#include <dlfcn.h>
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
void run() {
  sycl::queue q;
  sycl::buffer<int, 1> b(1);
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc(b, cgh);
     using AccTy = decltype(acc);
     // Same Kernel name as in the shared library.
     struct Kernel {
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
#ifdef RUN_FIRST
  // CHECK-FIRST: piProgramBuild
  // CHECK-FIRST: Main: 2
  // CHECK-FIRST: Main: 2
  run();
  run();
#endif

#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)
#define SO_FNAME "lib" STRINGIFY(FNAME) ".so"

  void *handle = dlopen(SO_FNAME, RTLD_LAZY);
  int (*func)();
  *(void **)(&func) = dlsym(handle, "_Z3foov");

#ifdef RUN_MIDDLE_BEFORE
  // CHECK-MIDDLE-BEFORE: piProgramBuild
  // CHECK-MIDDLE-BEFORE: Main: 2
  // CHECK-MIDDLE-BEFORE: Main: 2
  run();
  run();
#endif

  // CHECK: piProgramBuild
  // CHECK: Foo: 1
  // CHECK: Foo: 1
  assert(func() == 1);
  assert(func() == 1);

#ifdef RUN_MIDDLE_AFTER
  // CHECK-MIDDLE-AFTER: piProgramBuild
  // CHECK-MIDDLE-AFTER: Main: 2
  // CHECK-MIDDLE-AFTER: Main: 2
  run();
  run();
#endif

  dlclose(handle);

#ifdef RUN_LAST
  // CHECK-LAST: piProgramBuild
  // CHECK-LAST: Main: 2
  // CHECK-LAST: Main: 2
  run();
  run();
#endif

  return 0;
}
#endif
