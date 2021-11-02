// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning

#include <sycl/sycl.hpp>
#include <string>

using namespace cl::sycl;

int main() {
  static_assert(is_device_copyable_v<int>);
  std::vector<int> iv(5, 1);
  buffer b1(iv.data(), range<1>(5));

  static_assert(!is_device_copyable_v<std::string>);
  std::vector<std::string> sv{"hello", "sycl", "world"};
  buffer b2(sv.data(), range<1>(3));
  //expected-error@CL/sycl/buffer.hpp:* {{"The underlying data type of a buffer 'T' must be device copyable"}}

  static_assert(is_device_copyable<sycl::vec<int, 4>>::value);
  vec<int, 4> iVec;
  buffer b3(&iVec, range<1>(1));
  return 0;
}
