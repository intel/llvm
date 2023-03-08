// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr static int size = 1;
  queue testQueue;

  try {

    testQueue.submit([&](handler &cgh) {
      auto local_acc = local_accessor<int, 1>({size}, cgh);
      cgh.single_task<class kernel>([=]() { (void)local_acc; });
    });
    assert(0);
  } catch (sycl::exception) {
  }

  try {
    testQueue.submit([&](sycl::handler &cgh) {
      auto local_acc = local_accessor<int, 1>({size}, cgh);
      cgh.parallel_for<class parallel_kernel>(
          sycl::range<1>{size}, [=](sycl::id<1> ID) { (void)local_acc; });
    });
    assert(0);
  } catch (sycl::exception) {
  }

  return 0;
}
