// RUN: %{build} -O0 -o %t.O0.out
// RUN: %{run} %t.O0.out
// RUN: %{build} -O1 -o %t.O1.out
// RUN: %{run} %t.O1.out
// RUN: %{build} -O2 -o %t.O2.out
// RUN: %{run} %t.O2.out
// RUN: %{build} -O3 -o %t.O3.out
// RUN: %{run} %t.O3.out
// RUN: %{build} -Ofast -o %t.Ofast.out
// RUN: %{run} %t.Ofast.out
// REQUIRES: linux
// UNSUPPORTED: hip || cuda

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
