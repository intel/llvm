// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

int main() {
  static_assert(is_property_key<work_group_size_key>::value);
  static_assert(is_property_key<work_group_size_hint_key>::value);
  static_assert(is_property_key<sub_group_size_key>::value);

  static_assert(is_property_value<decltype(work_group_size<1>)>::value);
  static_assert(is_property_value<decltype(work_group_size<2, 2>)>::value);
  static_assert(is_property_value<decltype(work_group_size<3, 3, 3>)>::value);
  static_assert(is_property_value<decltype(work_group_size_hint<4>)>::value);
  static_assert(is_property_value<decltype(work_group_size_hint<5, 5>)>::value);
  static_assert(
      is_property_value<decltype(work_group_size_hint<6, 6, 6>)>::value);
  static_assert(is_property_value<decltype(sub_group_size<7>)>::value);

  static_assert(
      std::is_same_v<work_group_size_key, decltype(work_group_size<8>)::key_t>);
  static_assert(std::is_same_v<work_group_size_key,
                               decltype(work_group_size<9, 9>)::key_t>);
  static_assert(std::is_same_v<work_group_size_key,
                               decltype(work_group_size<10, 10, 10>)::key_t>);
  static_assert(std::is_same_v<work_group_size_hint_key,
                               decltype(work_group_size_hint<11>)::key_t>);
  static_assert(std::is_same_v<work_group_size_hint_key,
                               decltype(work_group_size_hint<12, 12>)::key_t>);
  static_assert(
      std::is_same_v<work_group_size_hint_key,
                     decltype(work_group_size_hint<13, 13, 13>)::key_t>);
  static_assert(
      std::is_same_v<sub_group_size_key, decltype(sub_group_size<14>)::key_t>);

  static_assert(work_group_size<15>[0] == 15);
  static_assert(work_group_size<16, 17>[0] == 16);
  static_assert(work_group_size<16, 17>[1] == 17);
  static_assert(work_group_size<18, 19, 20>[0] == 18);
  static_assert(work_group_size<18, 19, 20>[1] == 19);
  static_assert(work_group_size<18, 19, 20>[2] == 20);
  static_assert(work_group_size_hint<21>[0] == 21);
  static_assert(work_group_size_hint<22, 23>[0] == 22);
  static_assert(work_group_size_hint<22, 23>[1] == 23);
  static_assert(work_group_size_hint<24, 25, 26>[0] == 24);
  static_assert(work_group_size_hint<24, 25, 26>[1] == 25);
  static_assert(work_group_size_hint<24, 25, 26>[2] == 26);
  static_assert(sub_group_size<27>.value == 27);

  static_assert(std::is_same_v<decltype(sub_group_size<28>)::value_t,
                               std::integral_constant<uint32_t, 28>>);

  return 0;
}
