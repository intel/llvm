// RUN: %clangxx -fsycl -fsyntax-only %s -Xclang -verify -Xclang -verify-ignore-unexpected=note

#include <sycl/ext/oneapi/properties/properties.hpp>

using namespace sycl::ext::oneapi::experimental::new_properties;

using mock_property_sort_key_t = int;

template <typename property_key_t> constexpr auto generate_property_key_name() {
#if defined(__clang__) || defined(__GNUC__)
  return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
  return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
}

template <typename property_t, typename property_key_t = property_t>
struct named_property_base
    : public detail::property_base<property_t, property_key_t> {
  static constexpr std::string_view property_name =
      generate_property_key_name<property_key_t>();
};

template <int N> struct property : named_property_base<property<N>> {};

void test() {
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Property keys must be unique}}
  std::ignore = properties{property<1>{}, property<1>{}};

  constexpr properties pl{property<1>{}, property<2>{}};
  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Property keys must be unique}}
  std::ignore = properties{pl, property<1>{}};
}


