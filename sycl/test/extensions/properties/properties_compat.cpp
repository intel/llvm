// RUN: %clangxx -fsycl -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/properties/properties.hpp>

#include <cstdint>
#include "mock_compile_time_properties.hpp"

#include <type_traits>

using namespace sycl::ext::oneapi::experimental;

int main() {
  constexpr auto Props = properties{bar, baz<1>, foo{42}};

  static_assert(is_property_list_v<std::remove_cv_t<decltype(Props)>>);
  static_assert(decltype(Props)::has_property<bar_key>());
  static_assert(decltype(Props)::has_property<baz_key>());
  static_assert(decltype(Props)::has_property<foo_key>());
  static_assert(std::is_same_v<decltype(Props.get_property<baz_key>()),
                               baz_key::value_t<1>>);
  static_assert(Props.get_property<foo_key>().value == 42);

  return 0;
}