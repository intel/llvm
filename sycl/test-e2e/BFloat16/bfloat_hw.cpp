// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// "Hello world" bfloat16 test which checks conversion algorithms on host.

#include <sycl/detail/core.hpp>

#include <cstdint>
#include <type_traits>

template <size_t Size>
using get_uint_type_of_size = typename std::conditional_t<
    Size == 1, uint8_t,
    std::conditional_t<
        Size == 2, uint16_t,
        std::conditional_t<Size == 4, uint32_t,
                           std::conditional_t<Size == 8, uint64_t, void>>>>;

using bfloat16 = sycl::ext::oneapi::bfloat16;
using Bfloat16StorageT = get_uint_type_of_size<sizeof(bfloat16)>;

bool test(float Val, Bfloat16StorageT Bits) {
  std::cout << "Value: " << Val << " Bits: " << std::hex << "0x" << Bits
            << std::dec << "...\n";
  bool Passed = true;
  {
    std::cout << "  float -> bfloat16 conversion ...";
    Bfloat16StorageT RawVal = sycl::bit_cast<Bfloat16StorageT>(bfloat16(Val));
    bool Res = (RawVal == Bits);
    Passed &= Res;

    if (Res) {
      std::cout << "passed\n";
    } else {
      std::cout << "failed. " << std::hex << "0x" << RawVal << " != "
                << "0x" << Bits << "(gold)\n"
                << std::dec;
    }
  }
  {
    std::cout << "  bfloat16 -> float conversion ...";
    float NewVal = static_cast<float>(sycl::bit_cast<bfloat16>(Bits));
    bool Res = (NewVal == Val);
    Passed &= Res;

    if (Res) {
      std::cout << "passed\n";
    } else {
      std::cout << "failed. " << Val << "(Gold) != " << NewVal << "\n";
    }
  }
  return Passed;
}

int main() {
  bool passed = true;
  passed &= test(3.140625f, 0x4049);
  std::cout << (passed ? "Test Passed\n" : "Test FAILED\n");
  return (passed ? 0 : 1);
}
