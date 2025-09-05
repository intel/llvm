// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <cassert>

namespace s = sycl;

int main() {
  // dot
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class dotF1F1>(
            [=]() { AccR[0] = s::dot(float{0.5}, float{1.6}); });
      });
    }
    assert(r == 0.8f);
  }

  // distance
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class distanceF1>(
            [=]() { AccR[0] = s::distance(float{1.f}, float{3.f}); });
      });
    }
    assert(r == 2.f);
  }

  // length
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class lengthF1>(
            [=]() { AccR[0] = s::length(float{1.f}); });
      });
    }
    assert(r == 1.f);
  }

  // normalize
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class normalizeF1>(
            [=]() { AccR[0] = s::normalize(float{2.f}); });
      });
    }
    assert(r == 1.f);
  }

  // fast_distance
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_distanceF1>(
            [=]() { AccR[0] = s::fast_distance(float{1.f}, float{3.f}); });
      });
    }
    assert(r == 2.f);
  }

  // fast_length
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_lengthF1>(
            [=]() { AccR[0] = s::fast_length(float{2.f}); });
      });
    }
    assert(r == 2.f);
  }

  // fast_normalize
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_normalizeF1>(
            [=]() { AccR[0] = s::fast_normalize(float{2.f}); });
      });
    }

    assert(r == 1.f);
  }

  return 0;
}
