// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <sycl/atomic.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

template <typename T> class foo;

template <typename T> void kernel_func(T val) {
  queue testQueue;

  T data = val;
  buffer<T, 1> buf(&data, range<1>(1));

  testQueue.submit([&](handler &cgh) {
    auto GlobAcc = buf.template get_access<access_mode::read_write>(cgh);
    cgh.single_task<class foo<T>>([=]() {
      sycl::atomic_ref<T, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          a(GlobAcc[0]);
      T var = a.load();
    });
  });
}

int main() {
  kernel_func<float>(5.5);
  kernel_func<int>(42);
  return 0;
}
