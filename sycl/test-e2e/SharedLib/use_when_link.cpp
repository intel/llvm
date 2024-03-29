// REQUIRES: linux
//
// RUN: %{build} -DBUILD_LIB -fPIC -shared -o %T/lib%basename_t.so

// RUN: %{build} -DFOO_FIRST -L%T -o %t.out -l%basename_t -Wl,-rpath=%T
// RUN: %{run} %t.out

// RUN: %{build} -L%T -o %t.out -l%basename_t -Wl,-rpath=%T
// RUN: %{run} %t.out

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
  assert(foo() == 1);
#endif
  run();
#ifndef FOO_FIRST
  assert(foo() == 1);
#endif
  return 0;
}
#endif
