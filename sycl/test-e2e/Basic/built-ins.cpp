// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// RUN: %{build} -D__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ -o %t_var.out
// RUN: %{run} %t_var.out | FileCheck %s

// Hits an assertion with AMD:
// XFAIL: hip_amd

#include <sycl/sycl.hpp>

#include <cassert>

namespace s = sycl;

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

#ifdef __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__
  if (!q.get_device().has(sycl::aspect::fp64)) {
    std::cout << "Test with __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ defined is "
                 "skipped because the device did not have fp64."
              << std::endl;
    return 0;
  }
#endif

  // Test printf
  q.submit([&](s::handler &CGH) {
     CGH.single_task<class printf>([=]() {
       s::ext::oneapi::experimental::printf(format, 123, 1.23f);
       // CHECK: {{(Hello, World! 123 1.23)?}}
     });
   }).wait();

  s::ext::oneapi::experimental::printf(format, 321, 3.21f);
  // CHECK: {{(Hello, World! 123 1.23)?}}

  // Test common
  {
    s::buffer<float, 1> BufMin(s::range<1>(1));
    s::buffer<s::float2, 1> BufMax(s::range<1>(1));
    q.submit([&](s::handler &cgh) {
      auto AccMin = BufMin.get_access<s::access::mode::write>(cgh);
      auto AccMax = BufMax.get_access<s::access::mode::write>(cgh);
      cgh.single_task<class common>([=]() {
        AccMax[0] = s::max(s::float2{0.5f, 2.5f}, s::float2{2.3f, 2.3f});
        AccMin[0] = s::min(float{0.5f}, float{2.3f});
      });
    });

    sycl::host_accessor AccMin(BufMin, sycl::read_only);
    sycl::host_accessor AccMax(BufMax, sycl::read_only);

    assert(AccMin[0] == 0.5);
    assert(AccMax[0].x() == 2.3f && AccMax[0].y() == 2.5f);
    assert(s::min(0.5f, 2.3f) == 0.5);
    auto Res = s::max(s::int4{5, 2, 1, 5}, s::int4{3, 3, 4, 2});
    assert(Res.x() == 5 && Res.y() == 3 && Res.z() == 4 && Res.w() == 5);
  }

  return 0;
}
