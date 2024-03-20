// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <array>
#include <cassert>
#include <cmath>

namespace s = sycl;

int main() {
  // fmin
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fminF2F2>([=]() {
          AccR[0] = s::fmin(s::float2{0.5f, 3.4f}, s::float2{2.3f, 0.4f});
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    assert(r1 == 0.5f);
    assert(r2 == 0.4f);
  }

  // fabs
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fabsF2>([=]() {
          AccR[0] = s::fabs(s::float2{-1.0f, 2.0f});
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // floor
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class floorF2>([=]() {
          AccR[0] = s::floor(s::float2{1.4f, 2.8f});
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    assert(r1 == 1.0f);
    assert(r2 == 2.0f);
  }

  // ceil
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ceilF2>([=]() {
          AccR[0] = s::ceil(s::float2{1.4f, 2.8f});
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    assert(r1 == 2);
    assert(r2 == 3);
  }

  // fract with global memory
  {
    s::float2 r{0, 0};
    s::float2 i{0, 0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::float2, 1> BufI(&i, s::range<1>(1));

      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF2GF2>([=]() {
          s::global_ptr<s::float2> Iptr(AccI);
          AccR[0] = s::fract(s::float2{1.5f, 2.5f}, Iptr);
        });
      });
    }

    float r1 = r.x();
    float r2 = r.y();
    float i1 = i.x();
    float i2 = i.y();

    assert(r1 == 0.5f);
    assert(r2 == 0.5f);
    assert(i1 == 1.0f);
    assert(i2 == 2.0f);
  }

  // fract with private memory
  {
    s::float2 r{0, 0};
    s::float2 i{0, 0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::float2, 1> BufI(&i, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF2PF2>([=]() {
          s::float2 temp(0.0);
          s::private_ptr<s::float2> Iptr(&temp);
          AccR[0] = s::fract(s::float2{1.5f, 2.5f}, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }

    float r1 = r.x();
    float r2 = r.y();
    float i1 = i.x();
    float i2 = i.y();

    assert(r1 == 0.5f);
    assert(r2 == 0.5f);
    assert(i1 == 1.0f);
    assert(i2 == 2.0f);
  }

  // lgamma with private memory
  {
    s::float2 r{0, 0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF2>([=]() {
          AccR[0] = s::lgamma(s::float2{10.f, -2.4f});
        });
      });
    }

    float r1 = r.x();
    float r2 = r.y();

    assert(r1 > 12.8017f && r1 < 12.8019f); // ~12.8018
    assert(r2 > 0.1024f && r2 < 0.1026f);   // ~0.102583
  }

  // lgamma_r with private memory
  {
    s::float2 r{0, 0};
    s::int2 i{0, 0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::int2, 1> BufI(&i, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF2PF2>([=]() {
          s::int2 temp(0.0);
          s::private_ptr<s::int2> Iptr(&temp);
          AccR[0] = s::lgamma_r(s::float2{10.f, -2.4f}, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }

    float r1 = r.x();
    float r2 = r.y();
    int i1 = i.x();
    int i2 = i.y();

    assert(r1 > 12.8017f && r1 < 12.8019f); // ~12.8018
    assert(r2 > 0.1024f && r2 < 0.1026f);   // ~0.102583
    assert(i1 == 1);                        // tgamma of 10 is ~362880.0
    assert(i2 == -1); // tgamma of -2.4 is ~-1.1080299470333461
  }

  return 0;
}
