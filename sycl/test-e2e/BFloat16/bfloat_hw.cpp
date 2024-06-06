// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: accelerator
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
    std::cout << "  float[4] -> bfloat16[4] conversion on device..."<<std::flush;
    queue Q;
    int err = 0;
    buffer<int> err_buf(&err, 1);

    // Convert float array to bfloat16 array
    Q.submit([&](handler &CGH) {
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, CGH);
      CGH.single_task([=]() {
        float FloatArray[4] = {1.0f, -1.0f, 0.0f, 2.0f};
        bfloat16 BF16Array[4];
        sycl::ext::oneapi::detail::FloatVecToBF16Vec<4>(FloatArray, BF16Array);
        for (int i = 0; i < 4; i++) {
          if (FloatArray[i] != (float)BF16Array[i]) {
            ERR[0] = 1;
          }
        }
      });
    }).wait();

    if (err)
      std::cout <<"failed\n";
    else
      std::cout <<"passed\n";
    
    std::cout << "bfloat16[4] -> float[4] conversion on device..."<<std::flush;

    // Convert bfloat16 array back to float array
    Q.submit([&](handler &CGH) {
      accessor<int, 1, access::mode::write, target::device> ERR(err_buf, CGH);
      CGH.single_task([=]() {
        bfloat16 BF16Array[3] = {1.0f, 0.0f, -1.0f};
        float FloatArray[3];
        sycl::ext::oneapi::detail::BF16VecToFloatVec<4>(BF16Array, FloatArray);
        for (int i = 0; i < 3; i++) {
          if (FloatArray[i] != (float)BF16Array[i]) {
            ERR[0] = 1;
          }
        }
      });
    }).wait();

    if (err)
      std::cout <<"failed\n";
    else
      std::cout <<"passed\n";

    Passed &= !err;      
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
