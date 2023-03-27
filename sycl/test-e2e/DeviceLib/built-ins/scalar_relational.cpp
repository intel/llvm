// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>

namespace s = sycl;

int main() {
  s::queue myQueue;

  // isequal-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequalF1F1>(
            [=]() { AccR[0] = s::isequal(float{10.5f}, float{10.5f}); });
      });
    }
    assert(r == 1);
  }

  // isnotequal-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotequalF1F1>(
            [=]() { AccR[0] = s::isnotequal(float{0.4f}, float{0.5f}); });
      });
    }
    assert(r == 1);
  }

  // isgreater-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterF1F1>(
            [=]() { AccR[0] = s::isgreater(float{0.6f}, float{0.5f}); });
      });
    }
    assert(r == 1);
  }

  // isgreaterequal-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterequalF1F1>(
            [=]() { AccR[0] = s::isgreaterequal(float{0.5f}, float{0.5f}); });
      });
    }
    assert(r == 1);
  }

  // isless-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessF1F1>(
            [=]() { AccR[0] = s::isless(float{0.4f}, float{0.5f}); });
      });
    }
    assert(r == 1);
  }

  // islessequal-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessequalF1F1>(
            [=]() { AccR[0] = s::islessequal(float{0.5f}, float{0.5f}); });
      });
    }
    assert(r == 1);
  }

  // islessgreater-float
  {
    int r{1};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessgreaterF1F1>(
            [=]() { AccR[0] = s::islessgreater(float{0.5f}, float{0.5f}); });
      });
    }
    assert(r == 0);
  }

  // isfinite-float
  {
    int r{1};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isfiniteF1>(
            [=]() { AccR[0] = s::isfinite(float{NAN}); });
      });
    }
    assert(r == 0);
  }

  // isinf-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isinfF1>(
            [=]() { AccR[0] = s::isinf(float{INFINITY}); });
      });
    }
    assert(r == 1);
  }

  // isnan-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnanF1>(
            [=]() { AccR[0] = s::isnan(float{NAN}); });
      });
    }
    assert(r == 1);
  }

  // isnormal-float
  {
    int r{1};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnormalF1>(
            [=]() { AccR[0] = s::isnormal(float{INFINITY}); });
      });
    }
    assert(r == 0);
  }

  // isnormal-double
  if (myQueue.get_device().has(sycl::aspect::fp64)) {
    int r{1};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnormalD1>(
            [=]() { AccR[0] = s::isnormal(double{INFINITY}); });
      });
    }
    assert(r == 0);
  }

  // isordered-float
  {
    int r{1};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isorderedF1F1>(
            [=]() { AccR[0] = s::isordered(float{4.0f}, float{NAN}); });
      });
    }
    assert(r == 0);
  }

  // isunordered-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isunorderedF1F1>(
            [=]() { AccR[0] = s::isunordered(float{4.0f}, float{NAN}); });
      });
    }
    assert(r == 1);
  }

  // signbit-float
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class signbitF1>(
            [=]() { AccR[0] = s::signbit(float{-12.0f}); });
      });
    }
    assert(r == 1);
  }

  // any-integer
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyF1positive>(
            [=]() { AccR[0] = s::any(int{12}); });
      });
    }
    assert(r == 0);
  }
  // any-integer
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyF1zero>([=]() { AccR[0] = s::any(int{0}); });
      });
    }
    assert(r == 0);
  }

  // any-integer
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyF1negative>(
            [=]() { AccR[0] = s::any(int{-12}); });
      });
    }
    assert(r == 1);
  }

  // all-integer
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allF1positive>(
            [=]() { AccR[0] = s::all(int{12}); });
      });
    }
    assert(r == 0);
  }

  // all-integer
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allF1zero>([=]() { AccR[0] = s::all(int{0}); });
      });
    }
    assert(r == 0);
  }

  // all-integer
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allF1negative>(
            [=]() { AccR[0] = s::all(int{-12}); });
      });
    }
    assert(r == 1);
  }

  // bitselect-float
  {
    float r{0.0f};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class bitselectF1F1F1>([=]() {
          AccR[0] = s::bitselect(float{112.112}, float{34.34}, float{3.3});
        });
      });
    }
    assert(r <= 80.5478 && r >= 80.5476); // r = 80.5477
  }

  // select-float,int
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1positive>([=]() {
          AccR[0] = s::select(float{34.34}, float{123.123}, int{1});
        });
      });
    }
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  // select-float,int
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1zero>([=]() {
          AccR[0] = s::select(float{34.34}, float{123.123}, int{0});
        });
      });
    }
    assert(r <= 34.35 && r >= 34.33); // r = 34.34
  }

  // select-float,int
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1I1negative>([=]() {
          AccR[0] = s::select(float{34.34}, float{123.123}, int{-1});
        });
      });
    }
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  // select-float,bool
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1B1true>(
            [=]() { AccR[0] = s::select(34.34f, 123.123f, true); });
      });
    }
    assert(r <= 123.124 && r >= 123.122); // r = 123.123
  }

  // select-float,bool
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF1F1B1false>(
            [=]() { AccR[0] = s::select(34.34f, 123.123f, false); });
      });
    }
    assert(r <= 34.35 && r >= 34.33); // r = 34.34
  }

  return 0;
}
