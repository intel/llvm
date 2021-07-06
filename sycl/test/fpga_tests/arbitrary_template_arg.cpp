// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// This test performs basic check of SYCL FPGA arbitrary template argument list.

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_utils.hpp>
#include <iostream>

enum class enum_class { first, second };

struct test_int_id;
template <int _N> struct test_int : std::integral_constant<int, _N> {
  using type_id = test_int_id;
};

struct test_enum_class_id;
template <enum_class _N>
struct test_enum_class : std::integral_constant<enum_class, _N> {
  using type_id = test_enum_class_id;
};

template <class... _Params> void func() {
  static constexpr auto test_int_value =
      sycl::INTEL::_GetValue<test_int<0>, _Params...>::value;
  static constexpr auto test_enum_class_value =
      sycl::INTEL::_GetValue<test_enum_class<enum_class::first>,
                             _Params...>::value;

  std::cout << test_int_value << " ";
  if (test_enum_class_value == enum_class::first) {
    std::cout << "first" << std::endl;
  } else {
    std::cout << "second" << std::endl;
  }
}

int main() {
  // CHECK: 0 first
  func();
  // CHECK: 1 first
  func<test_int<1>>();
  // CHECK: 0 second
  func<test_enum_class<enum_class::second>>();
  // CHECK: 1 second
  func<test_int<1>, test_enum_class<enum_class::second>>();
  // CHECK: 1 second
  func<test_enum_class<enum_class::second>, test_int<1>>();
}
