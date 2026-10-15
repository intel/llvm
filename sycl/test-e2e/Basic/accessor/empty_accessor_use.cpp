// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that 3D accessors with 0 elements are allowed to be captured in a
// kernel.

#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Q;

  buffer<int, 3> Buf{range<3>{1, 1, 1}};
  assert(Buf.size() == 1);
  Q.submit([&](handler &CGH) {
    accessor Acc{Buf, CGH, range<3>{0, 0, 0}, read_write};
    assert(Acc.empty());
    CGH.single_task([=] {
      if (!Acc.empty())
        Acc[id<3>{0, 0, 0}] = 42;
    });
  });
  return 0;
}
