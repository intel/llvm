// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::experimental;

int main() {
  // Check that oneapi::experimental::is_property_key is correctly specialized
  static_assert(sycl::ext::oneapi::experimental::is_property_key<gpu_cache_config_key>::value);

  // Check that oneapi::experimental::is_property_value is correctly specialized
  static_assert(
      sycl::ext::oneapi::experimental::is_property_value<decltype(gpu_cache_config{large_slm})>::value);
  static_assert(sycl::ext::oneapi::experimental::is_property_value<decltype(gpu_cache_config{large_data})>::value);

  // Check that property lists will accept the new properties
  sycl::ext::oneapi::experimental::properties Props1(gpu_cache_config{large_slm});
  using PS = decltype(Props1);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<PS>);
  static_assert(PS::has_property<gpu_cache_config>());
  assert(Props1.get_property<gpu_cache_config>() == large_slm);

  sycl::ext::oneapi::experimental::properties Props2(gpu_cache_config{large_data});
  using PS = decltype(Props2);
  static_assert(sycl::ext::oneapi::experimental::is_property_list_v<PS>);
  static_assert(PS::has_property<gpu_cache_config>());
  assert(Props2.get_property<gpu_cache_config>() == large_data);
}
