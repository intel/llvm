// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename T> class foo;

template <typename T> void kernel_func(T val) {
  queue testQueue;

  T data = val;
  buffer<T, 1> buf(&data, range<1>(1));

  testQueue.submit([&](handler &cgh) {
    auto GlobAcc = buf.template get_access<access::mode::atomic>(cgh);
    cgh.single_task<class foo<T>>([=]() {
      auto a = GlobAcc[0];
      T var = a.load();
    });
  });
}

int main() {
  kernel_func<float>(5.5);
  kernel_func<int>(42);
  return 0;
}
