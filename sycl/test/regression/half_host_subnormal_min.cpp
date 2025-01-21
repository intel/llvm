// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Checks that sycl::half on host can correctly cast its minimum subnormal value
// to and from a floating point value.

#include <sycl/sycl.hpp>

int main() {
  sycl::half SubnormalMin =
      sycl::bit_cast<sycl::half>((uint16_t)0b0000000000000001u);
  sycl::half ConvertedSubnormalMin =
      static_cast<sycl::half>(static_cast<float>(SubnormalMin));

  if (SubnormalMin != ConvertedSubnormalMin) {
    std::cout << "Failed! (0x" << std::hex
              << sycl::bit_cast<uint16_t>(SubnormalMin) << " != 0x"
              << sycl::bit_cast<uint16_t>(ConvertedSubnormalMin) << ")"
              << std::endl;
    return 1;
  }

  std::cout << "Passed!" << std::endl;
  return 0;
}
