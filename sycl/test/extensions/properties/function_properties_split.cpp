// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/kernel_properties/function_properties.hpp>

#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

int main() {
  static_assert(is_property_value<decltype(nd_range_kernel<1>)>::value);
  static_assert(is_property_value<decltype(single_task_kernel)>::value);
  static_assert(is_property_value<decltype(work_group_size<2, 3>)>::value);
  static_assert(
      is_property_value<decltype(work_group_size_hint<4, 5, 6>)>::value);
  static_assert(is_property_value<decltype(sub_group_size<7>)>::value);
  static_assert(is_property_value<decltype(max_work_group_size<8, 9>)>::value);
  static_assert(
      is_property_value<decltype(max_linear_work_group_size<10>)>::value);

  static_assert(work_group_size<2, 3>[0] == 2);
  static_assert(work_group_size<2, 3>[1] == 3);
  static_assert(work_group_size_hint<4, 5, 6>[2] == 6);
  static_assert(sub_group_size<7>.value == 7);
  static_assert(max_work_group_size<8, 9>[1] == 9);
  static_assert(max_linear_work_group_size<10>.value == 10);

  static_assert(std::is_same_v<decltype(work_group_size<2, 3>)::key_t,
                               work_group_size_key>);
  static_assert(std::is_same_v<decltype(work_group_size_hint<4>)::key_t,
                               work_group_size_hint_key>);
  static_assert(
      std::is_same_v<decltype(sub_group_size<7>)::key_t, sub_group_size_key>);
  static_assert(std::is_same_v<decltype(max_work_group_size<8>)::key_t,
                               max_work_group_size_key>);
  static_assert(std::is_same_v<decltype(max_linear_work_group_size<10>)::key_t,
                               max_linear_work_group_size_key>);
  return 0;
}
