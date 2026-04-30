// RUN: %clangxx -fsycl -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
#include <sycl/ext/oneapi/kernel_properties.hpp>
#include <sycl/ext/oneapi/properties.hpp>

#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

int main() {
  static_assert(is_property_value<decltype(single_task_kernel)>::value);
  static_assert(is_property_value<decltype(work_group_size<2, 3>)>::value);
  static_assert(std::is_same_v<decltype(properties{work_group_size<1>}),
                               decltype(properties{work_group_size<1>})>);
  return 0;
}