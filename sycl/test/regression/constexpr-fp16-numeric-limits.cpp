// RUN: %clangxx -fsycl-device-only -fsyntax-only -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// expected-no-diagnostics
#include <CL/sycl.hpp>

int main() {
  constexpr cl::sycl::half L1 = std::numeric_limits<cl::sycl::half>::min();
  constexpr cl::sycl::half L2 = std::numeric_limits<cl::sycl::half>::max();
  constexpr cl::sycl::half L3 = std::numeric_limits<cl::sycl::half>::lowest();
  constexpr cl::sycl::half L4 = std::numeric_limits<cl::sycl::half>::epsilon();
  constexpr cl::sycl::half L5 =
      std::numeric_limits<cl::sycl::half>::round_error();
  constexpr cl::sycl::half L6 = std::numeric_limits<cl::sycl::half>::infinity();
  constexpr cl::sycl::half L6n =
      -std::numeric_limits<cl::sycl::half>::infinity();
  constexpr cl::sycl::half L7 =
      std::numeric_limits<cl::sycl::half>::quiet_NaN();
  constexpr cl::sycl::half L8 =
      std::numeric_limits<cl::sycl::half>::signaling_NaN();
  constexpr cl::sycl::half L9 =
      std::numeric_limits<cl::sycl::half>::denorm_min();

  return 0;
}
