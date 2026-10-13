// RUN: %clangxx -fsycl -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

// clang-format off
#include <sycl/ext/oneapi/kernel_properties.hpp>
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
// clang-format on

namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void SingleTaskKernel();

int main() {
  constexpr auto Props = syclexp::properties{
      syclexp::work_group_size<1>, syclexp::device_has<sycl::aspect::cpu>};

  static_assert(
      syclexp::is_property_value<decltype(syclexp::single_task_kernel)>::value);
  static_assert(
      syclexp::is_property_value<decltype(syclexp::nd_range_kernel<1>)>::value);
  static_assert(
      decltype(Props)::template has_property<syclexp::work_group_size_key>());
  static_assert(
      decltype(Props)::template has_property<syclexp::device_has_key>());
  static_assert(
      Props.template get_property<syclexp::work_group_size_key>()[0] == 1);
  return 0;
}