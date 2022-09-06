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
  // acos
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class acosF1>(
            [=]() { AccR[0] = s::acos(s::cl_float{0.5}); });
      });
    }
    assert(r > 1.047f && r < 1.048f); // ~1.0471975511965979
  }

  // acosh
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class acoshF1>(
            [=]() { AccR[0] = s::acosh(s::cl_float{2.4}); });
      });
    }
    assert(r > 1.522f && r < 1.523f); // ~1.5220793674636532
  }

  // asin
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class asinF1>(
            [=]() { AccR[0] = s::asin(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.523f && r < 0.524f); // ~0.5235987755982989
  }

  // asinh
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class asinhF1>(
            [=]() { AccR[0] = s::asinh(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.481f && r < 0.482f); // ~0.48121182505960347
  }

  // atan
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atanF1>(
            [=]() { AccR[0] = s::atan(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.463f && r < 0.464f); // ~0.4636476090008061
  }

  // atanh
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class atanhF1>(
            [=]() { AccR[0] = s::atanh(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.549f && r < 0.550f); // ~0.5493061443340549
  }

  // cbrt
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class cbrtF1>(
            [=]() { AccR[0] = s::cbrt(s::cl_float{27.0}); });
      });
    }
    assert(r == 3.f);
  }

  // ceil
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class ceilF1>(
            [=]() { AccR[0] = s::ceil(s::cl_float{0.5}); });
      });
    }
    assert(r == 1.f);
  }

  // cos
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class cosF1>(
            [=]() { AccR[0] = s::cos(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.877f && r < 0.878f); // ~0.8775825618903728
  }

  // cosh
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class coshF1>(
            [=]() { AccR[0] = s::cosh(s::cl_float{0.5}); });
      });
    }
    assert(r > 1.127f && r < 1.128f); // ~1.1276259652063807
  }

  // cospi
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class cospiF1>(
            [=]() { AccR[0] = s::cospi(s::cl_float{0.1}); });
      });
    }
    assert(r > 0.951f && r < 0.952f); // ~0.9510565162951535
  }

  // erfc
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class erfcF1>(
            [=]() { AccR[0] = s::erfc(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.479f && r < 0.480f); // ~0.4795001221869535
  }

  // erf
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class erfF1>(
            [=]() { AccR[0] = s::erf(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.520f && r < 0.521f); // ~0.5204998778130465
  }

  // exp
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class expF1>(
            [=]() { AccR[0] = s::exp(s::cl_float{0.5}); });
      });
    }
    assert(r > 1.648f && r < 1.649f); // ~1.6487212707001282
  }

  // exp2
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class exp2F1>(
            [=]() { AccR[0] = s::exp2(s::cl_float{8.0}); });
      });
    }
    assert(r == 256.0f);
  }

  // exp10
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class exp10F1>(
            [=]() { AccR[0] = s::exp10(s::cl_float{2}); });
      });
    }
    assert(r == 100.0f);
  }

  // expm1
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class expm1F1>(
            [=]() { AccR[0] = s::expm1(s::cl_float{0.5}); });
      });
    }
    assert(r > 0.648f && r < 0.649f); // ~0.6487212707001282
  }

  // fabs
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fabsF1>(
            [=]() { AccR[0] = s::fabs(s::cl_float{-0.5}); });
      });
    }
    assert(r == 0.5f);
  }

  // floor
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class floorF1>(
            [=]() { AccR[0] = s::floor(s::cl_float{0.5}); });
      });
    }
    assert(r == 0.f);
  }

  // fmax
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fmaxF1F1>(
            [=]() { AccR[0] = s::fmax(s::cl_float{0.5}, s::cl_float{0.8}); });
      });
    }
    assert(r == 0.8f);
  }

  // fmin
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fminF1F1>(
            [=]() { AccR[0] = s::fmin(s::cl_float{0.5}, s::cl_float{0.8}); });
      });
    }
    assert(r == 0.5f);
  }

  // fmod
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fmodF1F1>(
            [=]() { AccR[0] = s::fmod(s::cl_float{5.1}, s::cl_float{3.0}); });
      });
    }
    assert(r == 2.1f);
  }

  // lgamma with private memory
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgammaF1>(
            [=]() { AccR[0] = s::lgamma(s::cl_float{10.f}); });
      });
    }
    assert(r > 12.8017f && r < 12.8019f); // ~12.8018
  }

  // lgamma with private memory
  {
    s::cl_float r{0};
    {
      s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgammaF1_neg>(
            [=]() { AccR[0] = s::lgamma(s::cl_float{-2.4f}); });
      });
    }
    assert(r > 0.1024f && r < 0.1026f); // ~0.102583
  }

  return 0;
}
