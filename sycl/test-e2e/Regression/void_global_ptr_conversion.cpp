// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// Regression test to check that conversion to/from global_ptr<void...> works as
// expected.

// CHECK: 1
// CHECK: 2
// CHECK: 3
// CHECK: 4
// CHECK: 5

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::access;

int main() {
  sycl::queue queue;
  constexpr int bufLen = 5;

  int *buf = sycl::malloc_host<int>(bufLen, queue);

  queue.submit([&](sycl::handler &cgh) {
    auto kernel = [=](sycl::id<1> id) {
      buf[id] = 1;
      sycl::global_ptr<int, decorated::yes> IntermediatePtr =
          sycl::address_space_cast<address_space::global_space, decorated::yes,
                                   int>(buf);
      sycl::global_ptr<void, decorated::yes> voidIntermediatePtr{
          IntermediatePtr};
      sycl::global_ptr<int, decorated::yes> Ptr =
          static_cast<sycl::global_ptr<int, decorated::yes>>(
              voidIntermediatePtr);
      Ptr[id] += static_cast<int>(id);
    };
    cgh.parallel_for(sycl::range<1>(bufLen), kernel);
  });
  queue.wait();

  for (int i = 0; i < bufLen; ++i) {
    std::cout << buf[i] << std::endl;
  }

  sycl::free(buf, queue);
  return 0;
}
