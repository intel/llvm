// RUN: %clangxx -fsycl-device-only -fsyntax-only -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// expected-no-diagnostics
#include <sycl/sycl.hpp>

int main() {
  constexpr sycl::half L1 = std::numeric_limits<sycl::half>::min();
  constexpr sycl::half L2 = std::numeric_limits<sycl::half>::max();
  constexpr sycl::half L3 = std::numeric_limits<sycl::half>::lowest();
  constexpr sycl::half L4 = std::numeric_limits<sycl::half>::epsilon();
  constexpr sycl::half L5 = std::numeric_limits<sycl::half>::round_error();
  constexpr sycl::half L6 = std::numeric_limits<sycl::half>::infinity();
  constexpr sycl::half L6n = -std::numeric_limits<sycl::half>::infinity();
  constexpr sycl::half L7 = std::numeric_limits<sycl::half>::quiet_NaN();
  constexpr sycl::half L8 = std::numeric_limits<sycl::half>::signaling_NaN();
  constexpr sycl::half L9 = std::numeric_limits<sycl::half>::denorm_min();

  return 0;
}
