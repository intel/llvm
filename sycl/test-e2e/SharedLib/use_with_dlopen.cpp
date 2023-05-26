// REQUIRES: linux
//
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so

// RUN: %{build} -DRUN_FIRST -o %t.out -ldl -Wl,-rpath=%T
// RUN: %{run} %t.out

// RUN: %{build} -DRUN_MIDDLE_BEFORE -o %t.out -ldl -Wl,-rpath=%T
// RUN: %{run} %t.out

// RUN: %{build} -DRUN_MIDDLE_AFTER -o %t.out -ldl -Wl,-rpath=%T
// RUN: %{run} %t.out

// This causes SEG. FAULT.
// RUNx: %{build} -DRUN_LAST -o %t.out -ldl -Wl,-rpath=%T
// RUNx: %{run} %t.out

#include <sycl/sycl.hpp>

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
  run();
#endif
  void *handle = dlopen("libuse_with_dlopen.cpp.so", RTLD_LAZY);
  int (*func)();
  *(void **)(&func) = dlsym(handle, "_Z3foov");
#ifdef RUN_MIDDLE_BEFORE
  run();
#endif
  assert(func() == 1);
#ifdef RUN_MIDDLE_AFTER
  run();
#endif
  dlclose(handle);
#ifdef RUN_LAST
  run();
#endif
  return 0;
}
#endif
