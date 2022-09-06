// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <array>
#include <cassert>
#include <cmath>

namespace s = sycl;

int main() {

  // acospi
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class acospiF1>(
            [=]() { AccR[0] = s::acospi(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.333f && r < 0.334f); // ~0.33333333333333337
  }

  // asinpi
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class asinpiF1>(
            [=]() { AccR[0] = s::asinpi(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.166f && r < 0.167f); // ~0.16666666666666669
  }

  // atan2
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atan2F1F1>(
            [=]() { AccR[0] = s::atan2(s::cl_float{0.5}, s::cl_float{0.5}); });
      });
    }
    assert(r > 0.785f && r < 0.786f); // ~0.7853981633974483
  }

  // atanpi
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atanpiF1>(
            [=]() { AccR[0] = s::atanpi(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.147f && r < 0.148f); // ~0.14758361765043326
  }

  // atan2pi
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atan2piF1F1>([=]() {
          AccR[0] = s::atan2pi(s::cl_float{0.5}, s::cl_float{0.5});
        });
      });
    }
    assert(r > 0.249f && r < 0.251f); // ~0.25
  }

  // copysign
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class copysignF1F1>([=]() {
          AccR[0] = s::copysign(s::cl_float{1}, s::cl_float{-0.5});
        });
      });
    }
    assert(r == -1.f);
  }

  // fdim
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fdimF1F1>(
            [=]() { AccR[0] = s::fdim(s::cl_float{1.6}, s::cl_float{0.6}); });
      });
    }
    assert(r == 1.0f);
  }

  // fma
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fmaF1F1F1>([=]() {
          AccR[0] =
              s::fma(s::cl_float{0.5}, s::cl_float{10.0}, s::cl_float{3.0});
        });
      });
    }
    assert(r == 8.0f);
  }

  // fract with global memory
  {
    s::cl_float r{0};
    s::cl_float i{999};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_float, 1> BufI(&i, s::range<1>(1),
                                     {s::property::buffer::use_host_ptr()});
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF1GF1>([=]() {
          s::global_ptr<s::cl_float> Iptr(AccI);
          AccR[0] = s::fract(s::cl_float{1.5}, Iptr);
        });
      });
    }
    assert(r == 0.5f);
    assert(i == 1.0f);
  }

  // fract with private memory
  {
    s::cl_float r{0};
    s::cl_float i{999};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_float, 1> BufI(&i, s::range<1>(1),
                                     {s::property::buffer::use_host_ptr()});
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class fractF1PF1>([=]() {
          s::cl_float temp(0.0);
          s::private_ptr<s::cl_float> Iptr(&temp);
          AccR[0] = s::fract(s::cl_float{1.5f}, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }
    assert(r == 0.5f);
    assert(i == 1.0f);
  }

  // lgamma_r with private memory
  {
    s::cl_float r{0};
    s::cl_int i{999};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_int, 1> BufI(&i, s::range<1>(1),
                                   {s::property::buffer::use_host_ptr()});
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF1PI1>([=]() {
          s::cl_int temp(0.0);
          s::private_ptr<s::cl_int> Iptr(&temp);
          AccR[0] = s::lgamma_r(s::cl_float{10.f}, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }
    assert(r > 12.8017f && r < 12.8019f); // ~12.8018
    assert(i == 1);                       // tgamma of 10 is ~362880.0
  }

  // lgamma_r with private memory
  {
    s::cl_float r{0};
    s::cl_int i{999};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::buffer<s::cl_int, 1> BufI(&i, s::range<1>(1),
                                   {s::property::buffer::use_host_ptr()});
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        auto AccI = BufI.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF1PI1_neg>([=]() {
          s::cl_int temp(0.0);
          s::private_ptr<s::cl_int> Iptr(&temp);
          AccR[0] = s::lgamma_r(s::cl_float{-2.4f}, Iptr);
          AccI[0] = *Iptr;
        });
      });
    }
    assert(r > 0.1024f && r < 0.1026f); // ~0.102583
    assert(i == -1); // tgamma of -2.4 is ~-1.1080299470333461
  }

  return 0;
}
