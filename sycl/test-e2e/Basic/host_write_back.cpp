// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

static constexpr int N = 32;

void testAccessor() {
  std::vector<int> vec(N, 1);
  {
    buffer<int, 1> buf(static_cast<const int *>(vec.data()), range<1>{N});
    queue q;
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access_mode::read_write>(cgh);
      cgh.parallel_for<class Kernel>({N}, [=](id<1> i) { acc[i] += 5; });
    });
  }
  assert(vec[0] == 1);
}

void testHostAcessor() {
  std::vector<int> vec(N, 1);
  {
    buffer<int, 1> buf(static_cast<const int *>(vec.data()), range<1>{N});
    auto acc = buf.get_host_access();
    acc[0] += 5;
  }
  assert(vec[0] == 1);
}

int main() {
  testAccessor();
  testHostAcessor();
}
