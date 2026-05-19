// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/kernel_properties/properties.hpp>

#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

int main() {
  constexpr auto Props =
      properties{work_group_size<2, 3>, device_has<sycl::aspect::cpu>};

  static_assert(is_property_value<decltype(single_task_kernel)>::value);
  static_assert(decltype(Props)::has_property<work_group_size_key>());
  static_assert(decltype(Props)::has_property<device_has_key>());
  static_assert(Props.get_property<work_group_size_key>()[0] == 2);
  static_assert(Props.get_property<work_group_size_key>()[1] == 3);

  return 0;
}