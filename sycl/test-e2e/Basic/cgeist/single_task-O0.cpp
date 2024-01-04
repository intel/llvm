// RUN: %{build} -O0 -o %t.O0.out
// RUN: %{run} %t.O0.out
// REQUIRES: linux
// UNSUPPORTED: hip || cuda

// COM: FIXME: Fails with -O0
// XFAIL: *

#include <sycl/sycl.hpp>
using namespace sycl;

void single_task(std::array<int, 1> &A) {
  auto q = queue{};
  device d = q.get_device();
  std::cout << "Using " << d.get_info<info::device::name>() << "\n";

  {
    auto buf = buffer<int, 1>{A.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class kernel_single_task>([=]() {
        A[0] = 1;
      });
    });
  }
}

int main() {
  std::array<int, 1> A = {0};
  single_task(A);
  assert(A[0] == 1);
  std::cout << "Test passed" << std::endl;
}
