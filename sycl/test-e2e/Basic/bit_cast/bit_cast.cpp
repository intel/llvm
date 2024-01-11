// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
// RUN: %if windows %{  %clangxx -fsycl -fsycl-host-compiler=cl -fsycl-host-compiler-options='/std:c++17 /Zc:__cplusplus'  -o %t2.out  %s  %}
// RUN: %if windows %{  %{run} %t2.out  %}

#include <sycl/sycl.hpp>

#include <iostream>
#include <math.h>
#include <type_traits>

constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <typename To, typename From> class BitCastKernel;

template <typename To, typename From>
To doBitCast(sycl::queue Queue, const From &ValueToConvert) {
  std::vector<To> Vec(1);
  {
    sycl::buffer<To, 1> Buf(Vec.data(), 1);
    Queue.submit([&](sycl::handler &cgh) {
      auto acc = Buf.template get_access<sycl_write>(cgh);
      cgh.single_task<class BitCastKernel<To, From>>([=]() {
        acc[0] = sycl::bit_cast<To>(ValueToConvert);
      });
    });
  }
  return Vec[0];
}

template <typename To, typename From>
int test(sycl::queue Queue, const From &Value) {
  auto ValueConvertedTwoTimes =
      doBitCast<From>(Queue, doBitCast<To>(Queue, Value));
  bool isOriginalValueEqualsToConvertedTwoTimes = false;
  if (std::is_integral<From>::value) {
    isOriginalValueEqualsToConvertedTwoTimes = Value == ValueConvertedTwoTimes;
  } else if ((std::is_floating_point<From>::value) ||
             std::is_same<From, sycl::half>::value) {
    static const float Epsilon = 0.0000001f;
    isOriginalValueEqualsToConvertedTwoTimes =
        fabs(Value - ValueConvertedTwoTimes) < Epsilon;
  } else {
    std::cerr << "Type " << typeid(From).name()
              << " neither integral nor floating point nor sycl::half\n";
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
  sycl::queue Queue;
  int ReturnCode = 0;

  if (Queue.get_device().has(sycl::aspect::fp16)) {
    std::cout << "sycl::half to unsigned short ...\n";
    ReturnCode += test<unsigned short>(Queue, sycl::half(1.0f));

    std::cout << "unsigned short to sycl::half ...\n";
    ReturnCode += test<sycl::half>(Queue, static_cast<unsigned short>(16384));

    std::cout << "sycl::half to short ...\n";
    ReturnCode += test<short>(Queue, sycl::half(1.0f));

    std::cout << "short to sycl::half ...\n";
    ReturnCode += test<sycl::half>(Queue, static_cast<short>(16384));
  }

  std::cout << "int to float ...\n";
  ReturnCode += test<float>(Queue, static_cast<int>(2));

  std::cout << "float to int ...\n";
  ReturnCode += test<int>(Queue, static_cast<float>(-2.4f));

  std::cout << "unsigned int to float ...\n";
  ReturnCode += test<float>(Queue, static_cast<unsigned int>(6));

  std::cout << "float to unsigned int ...\n";
  ReturnCode += test<unsigned int>(Queue, static_cast<float>(-2.4f));

  return ReturnCode;
}
