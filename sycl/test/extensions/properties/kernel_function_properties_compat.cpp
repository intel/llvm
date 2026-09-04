// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/kernel_properties/function_properties.hpp>

#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

int main() {
  static_assert(is_property_value<decltype(nd_range_kernel<1>)>::value);
  static_assert(is_property_value<decltype(single_task_kernel)>::value);
  static_assert(is_property_value<decltype(work_group_size<2, 3>)>::value);
  static_assert(std::is_same_v<decltype(max_linear_work_group_size<10>)::key_t,
                               max_linear_work_group_size_key>);
  static_assert(work_group_size<2, 3>[0] == 2);
  static_assert(work_group_size<2, 3>[1] == 3);

  return 0;
}