// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// "Hello world" bfloat16 test which checks conversion algorithms on host.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bfloat16.hpp>

#include <cstdint>
#include <type_traits>

using namespace sycl;

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

// Test bfloat16 array conversion to float and vice versa.
bool testArrayConversion() {
  std::cout << "Array conversion test...\n";
  bool Passed = true;

  // On host
  {
    std::cout << "  float[4] -> bfloat16[4] conversion on host...";
    float FloatArray[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    bfloat16 BFloatArray[4];
    sycl::ext::oneapi::detail::FloatVecToBF16Vec<4>(FloatArray, BFloatArray);
    float NewFloatArray[4];
    sycl::ext::oneapi::detail::BF16VecToFloatVec<4>(BFloatArray, NewFloatArray);
    bool Res = true;
    for (int i = 0; i < 4; ++i) {
      Res &= (FloatArray[i] == NewFloatArray[i]);
    }
    Passed &= Res;

    if (Res) {
      std::cout << "passed\n";
    } else {
      std::cout << "failed\n";
    }
  }

  // On device
  {
    queue Q;
    buffer<float, 1> FloatBuffer{range<1>(4)};
    buffer<float, 1> FloatBuffer2{range<1>(4)};
    buffer<Bfloat16StorageT, 1> BFloatBuffer{range<1>(4)};

    // Convert float array to bfloat16 array
    Q.submit([&](handler &CGH) {
      auto FloatArray = FloatBuffer.get_access<access::mode::write>(CGH);
      auto BFloatArray = BFloatBuffer.get_access<access::mode::write>(CGH);
      FloatArray[0] = 1.0f;
      FloatArray[1] = 2.0f;
      FloatArray[2] = 3.0f;
      FloatArray[3] = 4.0f;
      CGH.single_task<class ArrayConversion>([=]() {
        sycl::ext::oneapi::detail::FloatVecToBF16Vec<4>(FloatArray.get_multi_ptr<access::decorated::no>().get(),
                                                        BFloatArray.get_multi_ptr<access::decorated::no>().get());
      });
    }).wait();

    // Convert bfloat16 array back to float array
    Q.submit([&](handler &CGH) {
      auto BFloatArray = BFloatBuffer.get_access<access::mode::read>(CGH);
      auto FloatArray = FloatBuffer2.get_access<access::mode::write>(CGH);
      CGH.single_task<class ArrayConversion2>([=]() {
        sycl::ext::oneapi::detail::BF16VecToFloatVec<4>(
            BFloatArray.get_multi_ptr<access::decorated::no>().get(),
            FloatArray.get_multi_ptr<access::decorated::no>().get());
      });
    }).wait();

    // Compare the results
    bool Res = true;
    auto FloatArray = FloatBuffer.get_access<access::mode::read>();
    auto FloatArray2 = FloatBuffer2.get_access<access::mode::read>();
    for (int i = 0; i < 4; ++i) {
      Res &= (FloatArray[i] == FloatArray2[i]);
    }
    Passed &= Res;

    if (Res) {
      std::cout << "  float[4] -> bfloat16[4] -> float[4] conversion on device...passed\n";
    } else {
      std::cout << "  float[4] -> bfloat16[4] -> float[4] conversion on device...failed\n";
    }
  }
  return Passed;
}

int main() {
  bool passed = true;
  passed &= test(3.140625f, 0x4049);
  passed &= testArrayConversion();
  std::cout << (passed ? "Test Passed\n" : "Test FAILED\n");
  return (passed ? 0 : 1);
}
