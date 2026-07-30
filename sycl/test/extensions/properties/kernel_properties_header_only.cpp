// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/kernel_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void NDRangeKernel();

struct KernelFunctor {
  auto get(syclexp::properties_tag) const {
    return syclexp::properties{syclexp::work_group_size<4>,
                               syclexp::device_has<sycl::aspect::cpu>};
  }
};

int main() {
  constexpr auto Props = syclexp::properties{
      syclexp::work_group_size<2, 3>, syclexp::device_has<sycl::aspect::cpu>};

  static_assert(
      syclexp::is_property_value<decltype(syclexp::single_task_kernel)>::value);
  static_assert(
      syclexp::is_property_value<decltype(syclexp::nd_range_kernel<1>)>::value);
  static_assert(syclexp::is_property_value<
                decltype(syclexp::device_has<sycl::aspect::cpu>)>::value);
  static_assert(
      decltype(Props)::template has_property<syclexp::work_group_size_key>());
  static_assert(
      decltype(Props)::template has_property<syclexp::device_has_key>());
  static_assert(
      Props.template get_property<syclexp::work_group_size_key>()[0] == 2);
  static_assert(
      Props.template get_property<syclexp::work_group_size_key>()[1] == 3);

  (void)KernelFunctor{}.get(syclexp::properties_tag{});
  return 0;
}