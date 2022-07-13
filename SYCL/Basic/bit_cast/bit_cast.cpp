// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <iostream>
#include <math.h>
#include <type_traits>

constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename To, typename From> class BitCastKernel;

template <typename To, typename From> To doBitCast(const From &ValueToConvert) {
  std::vector<To> Vec(1);
  {
    sycl::buffer<To, 1> Buf(Vec.data(), 1);
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &cgh) {
      auto acc = Buf.template get_access<sycl_write>(cgh);
      cgh.single_task<class BitCastKernel<To, From>>([=]() {
        acc[0] = sycl::bit_cast<To>(ValueToConvert);
      });
    });
  }
  return Vec[0];
}

template <typename To, typename From> int test(const From &Value) {
  auto ValueConvertedTwoTimes = doBitCast<From>(doBitCast<To>(Value));
  bool isOriginalValueEqualsToConvertedTwoTimes = false;
  if (std::is_integral<From>::value) {
    isOriginalValueEqualsToConvertedTwoTimes = Value == ValueConvertedTwoTimes;
  } else if ((std::is_floating_point<From>::value) ||
             std::is_same<From, cl::sycl::half>::value) {
    static const float Epsilon = 0.0000001f;
    isOriginalValueEqualsToConvertedTwoTimes =
        fabs(Value - ValueConvertedTwoTimes) < Epsilon;
  } else {
    std::cerr << "Type " << typeid(From).name()
              << " neither integral nor floating point nor cl::sycl::half\n";
    return 1;
  }
  if (!isOriginalValueEqualsToConvertedTwoTimes) {
    std::cerr << "FAIL: Original value which is " << Value
              << " != value converted two times which is "
              << ValueConvertedTwoTimes << "\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

int main() {
  int ReturnCode = 0;

  std::cout << "cl::sycl::half to unsigned short ...\n";
  ReturnCode += test<unsigned short>(cl::sycl::half(1.0f));

  std::cout << "unsigned short to cl::sycl::half ...\n";
  ReturnCode += test<cl::sycl::half>(static_cast<unsigned short>(16384));

  std::cout << "cl::sycl::half to short ...\n";
  ReturnCode += test<short>(cl::sycl::half(1.0f));

  std::cout << "short to cl::sycl::half ...\n";
  ReturnCode += test<cl::sycl::half>(static_cast<short>(16384));

  std::cout << "int to float ...\n";
  ReturnCode += test<float>(static_cast<int>(2));

  std::cout << "float to int ...\n";
  ReturnCode += test<int>(static_cast<float>(-2.4f));

  std::cout << "unsigned int to float ...\n";
  ReturnCode += test<float>(static_cast<unsigned int>(6));

  std::cout << "float to unsigned int ...\n";
  ReturnCode += test<unsigned int>(static_cast<float>(-2.4f));

  return ReturnCode;
}
