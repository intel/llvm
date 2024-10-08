// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <cassert>

namespace s = sycl;

int main() {
  // max
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxF1F1>(
            [=]() { AccR[0] = s::max(float{0.5f}, float{2.3f}); });
      });
    }
    assert(r == 2.3f);
  }

  return 0;
}
