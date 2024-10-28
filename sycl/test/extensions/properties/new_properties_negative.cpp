// RUN: %clangxx -fsycl -fsyntax-only %s -Xclang -verify -Xclang -verify-ignore-unexpected=note

#include <sycl/ext/oneapi/properties/new_properties.hpp>

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

template <int N>
struct property_with_key
    : named_property_base<property_with_key<N>, struct prop_key_t> {};

namespace library_a {
struct prop : detail::property_base<prop> {
  // Wrong, violates the extension specification! Property name must include
  // library namespace to avoid collisions with other libraries!
  static constexpr std::string_view property_name{"prop"};
};
}
namespace library_b {
struct prop : detail::property_base<prop> {
  // Wrong, violates the extension specification! Property name must include
  // library namespace to avoid collisions with other libraries!
  static constexpr std::string_view property_name{"prop"};
};
}

void test() {
  // expected-error@sycl/ext/oneapi/properties/new_properties.hpp:* {{static assertion failed due to requirement '!std::is_same_v<property<1>, property<1>>': Duplicate property!}}
  std::ignore = properties{property<1>{}, property<1>{}};

  constexpr properties pl{property<1>{}, property<2>{}};
  // expected-error@sycl/ext/oneapi/properties/new_properties.hpp:* {{static assertion failed due to requirement '!std::is_same_v<property<2>, property<2>>': Duplicate property!}}
  std::ignore = pl + properties{property<2>{}};

  // Unfortunately, C++ front end doesn't use qualified name for "prop" below...
  // expected-error@sycl/ext/oneapi/properties/new_properties.hpp:* {{static assertion failed due to requirement 'prop::property_name != prop::property_name': Property name collision between different property keys!}}
  std::ignore = properties{library_a::prop{}, library_b::prop{}};

  // expected-error@sycl/ext/oneapi/properties/new_properties.hpp:* {{static assertion failed due to requirement '!std::is_same_v<prop_key_t, prop_key_t>': Duplicate property!}}
  std::ignore = properties{property_with_key<1>{}, property_with_key<2>{}};
}


