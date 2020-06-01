// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <iostream>
#include <math.h>
#include <type_traits>

constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename To, typename From>
class BitCastKernel;

template <typename To, typename From>
To doBitCast(const From &ValueToConvert) {
  std::vector<To> Vec(1);
  {
    sycl::buffer<To, 1> Buf(Vec.data(), 1);
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &cgh) {
      auto acc = Buf.template get_access<sycl_write>(cgh);
      cgh.single_task<class BitCastKernel<To, From>>([=]() {
        acc[0] = sycl::detail::bit_cast<To>(ValueToConvert);
      });
    });
  }
  return Vec[0];
}

template <typename To, typename From>
int test(const From &ValueToConvert, const To &Expected) {
  auto Actual = doBitCast<To>(ValueToConvert);
  bool isActualEqualsToExpected = false;
  if (std::is_integral<To>::value) {
    isActualEqualsToExpected = Actual == Expected;
  } else if ((std::is_floating_point<To>::value) || std::is_same<To, cl::sycl::half>::value) {
    static const float Epsilon = 0.0000001f;
    isActualEqualsToExpected = fabs(Actual - Expected) < Epsilon;
  } else {
    std::cerr << "Type " << typeid(To).name() << " neither integral nor floating point\n";
    return 1;
  }
  if (!isActualEqualsToExpected) {
    std::cerr << "FAIL: Actual which is " << Actual << " != expected which is " << Expected << "\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

int main() {
  int ReturnCode = 0;

  std::cout << "cl::sycl::half to unsigned short ...\n";
  ReturnCode += test(cl::sycl::half(1.0f), static_cast<unsigned short>(15360));

  std::cout << "unsigned short to cl::sycl::half ...\n";
  ReturnCode += test(static_cast<unsigned short>(16384), cl::sycl::half(2.0f));

  std::cout << "cl::sycl::half to short ...\n";
  ReturnCode += test(cl::sycl::half(1.0f), static_cast<short>(15360));

  std::cout << "short to cl::sycl::half ...\n";
  ReturnCode += test(static_cast<short>(16384), cl::sycl::half(2.0f));

  std::cout << "int to float ...\n";
  ReturnCode += test(static_cast<int>(2), static_cast<float>(2.8026e-45f));

  std::cout << "float to int ...\n";
  ReturnCode += test(static_cast<float>(-2.4f), static_cast<int>(-1072064102));

  std::cout << "unsigned int to float ...\n";
  ReturnCode += test(static_cast<unsigned int>(6), static_cast<float>(8.40779e-45f));

  std::cout << "float to unsigned int ...\n";
  ReturnCode += test(static_cast<float>(-2.4f), static_cast<unsigned int>(3222903194));

  return ReturnCode;
}
