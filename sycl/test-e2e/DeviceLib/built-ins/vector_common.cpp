// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <cassert>

namespace s = sycl;

int main() {
  // max
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxF2F2>([=]() {
          AccR[0] = s::max(s::float2{0.5f, 3.4f}, s::float2{2.3f, 0.4f});
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    assert(r1 == 2.3f);
    assert(r2 == 3.4f);
  }

  // max
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class maxF2F1>([=]() {
          AccR[0] = s::max(s::float2{0.5f, 3.4f}, float{3.0f});
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    assert(r1 == 3.0f);
    assert(r2 == 3.4f);
  }

  return 0;
}
