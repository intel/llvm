// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl::ext;

int main() {
  // Check that oneapi::experimental::is_property_key is correctly specialized
  static_assert(oneapi::experimental::is_property_key<
                intel::experimental::gpu_cache_config_key>::value);

  // Check that oneapi::experimental::is_property_value is correctly specialized
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::gpu_cache_config<
                   intel::experimental::gpu_cache_config_enum::large_slm>)>::
          value);
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::gpu_cache_config<
                   intel::experimental::gpu_cache_config_enum::large_data>)>::
          value);

  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::gpu_cache_config_large_slm)>::value);
  static_assert(
      oneapi::experimental::is_property_value<
          decltype(intel::experimental::gpu_cache_config_large_data)>::value);

  // Check that property lists will accept the new properties
  using PS = decltype(oneapi::experimental::properties(
      intel::experimental::gpu_cache_config_large_slm));
  static_assert(oneapi::experimental::is_property_list_v<PS>);
  static_assert(PS::has_property<intel::experimental::gpu_cache_config_key>());
  static_assert(PS::get_property<intel::experimental::gpu_cache_config_key>() ==
                intel::experimental::gpu_cache_config_large_slm);
}
