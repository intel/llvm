// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

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
