// End-to-End test for testing device image compression when we have two
// translation units, one compressed and one not compressed.
// REQUIRES: zstd

// DEFINE: %{fPIC_flag} =  %if windows %{%} %else %{-fPIC%}
// RUN: %{build} --offload-compress -DENABLE_KERNEL1 -shared %{fPIC_flag} -o %t_kernel1.so
// RUN: %{build} -DENABLE_KERNEL2 -shared %{fPIC_flag} -o %t_kernel2.so

// RUN: %{build} %t_kernel1.so %t_kernel2.so -Wl,-rpath=%T -o %t_compress.out
// RUN: %{run} %t_compress.out
#if defined(ENABLE_KERNEL1) || defined(ENABLE_KERNEL2)
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
using namespace sycl;
#endif

#ifdef ENABLE_KERNEL1
void kernel1() {
  int data = -1;
  {
    buffer<int> b(&data, range(1));
    queue q;
    q.submit([&](sycl::handler &cgh) {
      auto acc = accessor(b, cgh);
      cgh.single_task([=] { acc[0] = abs(acc[0]); });
    });
  }
  assert(data == 1);
}
#endif

#ifdef ENABLE_KERNEL2
void kernel2() {
  int data = -2;
  {
    buffer<int> b(&data, range(1));
    queue q;
    q.submit([&](sycl::handler &cgh) {
      auto acc = accessor(b, cgh);
      cgh.single_task([=] { acc[0] = abs(acc[0]); });
    });
  }
  assert(data == 2);
}
#endif

#if not defined(ENABLE_KERNEL1) && not defined(ENABLE_KERNEL2)
void kernel1();
void kernel2();

int main() {
  kernel1();
  kernel2();

  return 0;
}
#endif
