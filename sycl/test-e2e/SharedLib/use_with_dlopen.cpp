// REQUIRES: linux
//
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so

// DEFINE: %{compile} = %{build} -DFNAME=%basename_t -ldl -Wl,-rpath=%T

// RUN: %{compile} -o %t1.out -DRUN_FIRST
// RUN: %{run} %t1.out

// RUN: %{compile} -o %t2.out -DRUN_MIDDLE_BEFORE
// RUN: %{run} %t2.out

// RUN: %{compile} -o %t3.out -DRUN_MIDDLE_AFTER
// RUN: %{run} %t3.out

// RUN: %{compile} -o %t4.out -DRUN_LAST
// RUN: %{run} %t4.out

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
  run();
#endif

#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)
#define SO_FNAME "lib" STRINGIFY(FNAME) ".so"

  void *handle = dlopen(SO_FNAME, RTLD_LAZY);
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
