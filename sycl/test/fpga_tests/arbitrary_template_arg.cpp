// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

// This test performs basic check of SYCL FPGA arbitrary template argument list.

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_utils.hpp>
#include <iostream>

enum class enumClass { first, second };

struct testIntID;
template <int _N> struct testInt : std::integral_constant<int, _N> {
  using type_id = testIntID;
};

struct testEnumID;
template <enumClass _N>
struct testEnum : std::integral_constant<enumClass, _N> {
  using type_id = testEnumID;
};

template <int ExpectedIntValue, enumClass ExpectedEnumValue, class... _Params>
void func() {
  static_assert(sycl::INTEL::_GetValue<testInt<0>, _Params...>::value ==
                ExpectedIntValue);
  static_assert(
      sycl::INTEL::_GetValue<testEnum<enumClass::first>, _Params...>::value ==
      ExpectedEnumValue);
}

int main() {
  func<0, enumClass::first>();
  func<1, enumClass::first, testInt<1>>();
  func<0, enumClass::second, testEnum<enumClass::second>>();
  func<1, enumClass::second, testInt<1>, testEnum<enumClass::second>>();
  func<1, enumClass::second, testEnum<enumClass::second>, testInt<1>>();
}
