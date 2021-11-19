// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning

#include <string>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  static_assert(is_device_copyable_v<int>);
  std::vector<int> iv(5, 1);
  buffer b1(iv.data(), range<1>(5));

  static_assert(!is_device_copyable_v<std::string>);
  std::vector<std::string> sv{"hello", "sycl", "world"};
  buffer b2(sv.data(), range<1>(3));
  //expected-error@CL/sycl/buffer.hpp:* {{"'std::string' is not a device copyable type"}}

  return 0;
}
