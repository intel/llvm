// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <cassert>
#include <cmath>

namespace s = sycl;

template <typename T> bool isEqualTo(T x, T y, T epsilon = 0.005) {
  return std::fabs(x - y) <= epsilon;
}

int main() {
  s::queue myQueue;

  // dot
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class dotF2F2>([=]() {
          AccR[0] = s::dot(
              s::float2{
                  1.f,
                  2.f,
              },
              s::float2{4.f, 6.f});
        });
      });
    }
    assert(r == 16.f);
  }

  // cross (float)
  {
    s::float4 r{0};
    {
      s::buffer<s::float4, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class crossF4>([=]() {
          AccR[0] = s::cross(
              s::float4{
                  2.f,
                  3.f,
                  4.f,
                  0.f,
              },
              s::float4{
                  5.f,
                  6.f,
                  7.f,
                  0.f,
              });
        });
      });
    }

    float r1 = r.x();
    float r2 = r.y();
    float r3 = r.z();
    float r4 = r.w();

    assert(r1 == -3.f);
    assert(r2 == 6.f);
    assert(r3 == -3.f);
    assert(r4 == 0.0f);
  }

  // cross (double)
  if (myQueue.get_device().has(sycl::aspect::fp64)) {
    s::double4 r{0};
    {
      s::buffer<s::double4, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class crossD4>([=]() {
          AccR[0] = s::cross(
              s::double4{
                  2.5,
                  3.0,
                  4.0,
                  0.0,
              },
              s::double4{
                  5.2,
                  6.0,
                  7.0,
                  0.0,
              });
        });
      });
    }

    double r1 = r.x();
    double r2 = r.y();
    double r3 = r.z();
    double r4 = r.w();

    assert(isEqualTo<double>(r1, -3.0));
    assert(isEqualTo<double>(r2, 3.3));
    assert(isEqualTo<double>(r3, -0.6));
    assert(isEqualTo<double>(r4, 0.0));
  }

  // distance
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class distanceF2>([=]() {
          AccR[0] = s::distance(
              s::float2{
                  1.f,
                  2.f,
              },
              s::float2{
                  3.f,
                  4.f,
              });
        });
      });
    }
    assert(isEqualTo<float>(r, 2.82843f));
  }

  // length
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class lengthF2>([=]() {
          AccR[0] = s::length(s::float2{
              1.f,
              2.f,
          });
        });
      });
    }
    assert(isEqualTo<float>(r, 2.23607f));
  }

  // normalize
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class normalizeF2>([=]() {
          AccR[0] = s::normalize(s::float2{
              1.f,
              2.f,
          });
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();

    assert(isEqualTo<float>(r1, 0.447214f));
    assert(isEqualTo<float>(r2, 0.894427f));
  }

  // fast_distance
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_distanceF2>([=]() {
          AccR[0] = s::fast_distance(
              s::float2{
                  1.f,
                  2.f,
              },
              s::float2{
                  3.f,
                  4.f,
              });
        });
      });
    }
    assert(isEqualTo<float>(r, 2.82843f));
  }

  // fast_length
  {
    float r{0};
    {
      s::buffer<float, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_lengthF2>([=]() {
          AccR[0] = s::fast_length(s::float2{
              1.f,
              2.f,
          });
        });
      });
    }
    assert(isEqualTo<float>(r, 2.23607f));
  }

  // fast_normalize
  {
    s::float2 r{0};
    {
      s::buffer<s::float2, 1> BufR(&r, s::range<1>(1));
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class fast_normalizeF2>([=]() {
          AccR[0] = s::fast_normalize(s::float2{
              1.f,
              2.f,
          });
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();

    assert(isEqualTo<float>(r1, 0.447144));
    assert(isEqualTo<float>(r2, 0.894287));
  }

  return 0;
}
