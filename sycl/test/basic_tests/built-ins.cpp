// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// CUDA does not support printf.
// UNSUPPORTED: cuda
#include <CL/sycl.hpp>

#include <cassert>

namespace s = cl::sycl;

// According to OpenCL C spec, the format string must be in constant address
// space
#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

static const CONSTANT char format[] = "Hello, World! %d %f\n";

int main() {
  s::queue q{};

  // Test printf
  q.submit([&](s::handler &CGH) {
     CGH.single_task<class printf>([=]() {
       s::ONEAPI::experimental::printf(format, 123, 1.23);
       // CHECK: {{(Hello, World! 123 1.23)?}}
     });
   }).wait();

  s::ONEAPI::experimental::printf(format, 321, 3.21);
  // CHECK: {{(Hello, World! 123 1.23)?}}

  // Test common
  {
    s::buffer<s::cl_float, 1> BufMin(s::range<1>(1));
    s::buffer<s::cl_float2, 1> BufMax(s::range<1>(1));
    q.submit([&](s::handler &cgh) {
      auto AccMin = BufMin.get_access<s::access::mode::write>(cgh);
      auto AccMax = BufMax.get_access<s::access::mode::write>(cgh);
      cgh.single_task<class common>([=]() {
        AccMax[0] = s::max(s::cl_float2{0.5f, 2.5}, s::cl_float2{2.3f, 2.3});
        AccMin[0] = s::min(s::cl_float{0.5f}, s::cl_float{2.3f});
      });
    });

    auto AccMin = BufMin.template get_access<s::access::mode::read>();
    auto AccMax = BufMax.template get_access<s::access::mode::read>();

    assert(AccMin[0] == 0.5);
    assert(AccMax[0].x() == 2.3f && AccMax[0].y() == 2.5f);
    assert(s::min(0.5f, 2.3f) == 0.5);
    auto Res = s::max(s::int4{5, 2, 1, 5}, s::int4{3, 3, 4, 2});
    assert(Res.x() == 5 && Res.y() == 3 && Res.z() == 4 && Res.w() == 5);
  }

  return 0;
}
