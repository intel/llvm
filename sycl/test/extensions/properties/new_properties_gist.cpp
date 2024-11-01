// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/ext/oneapi/properties/new_properties.hpp>

#include <cassert>

using namespace sycl::ext::oneapi::experimental::new_properties;

namespace feature_one {
struct foo : detail::property_base<foo> {
  static constexpr std::string_view property_name{"feature_one::foo"};
};
} // namespace feature_one

namespace feature_two {
template <int N>
struct foo
    : detail::property_base<foo<N>, detail::property_key_value_template<foo>> {
  static constexpr std::string_view property_name{"feature_two::foo"};

  static constexpr auto value = N;

  static constexpr const char *ir_attribute_name = "sycl-foo";
  static constexpr int ir_attribute_value = N;
};

struct bar : detail::property_base<bar> {
  static constexpr std::string_view property_name{"feature_two::bar"};
  int value;

  bar(int x) : value(x) {};
};

template <typename T>
struct xyz
    : detail::property_base<xyz<T>, detail::property_key_type_template<xyz>> {
  static constexpr std::string_view property_name{"feature_two::xyz"};

  using type = T;

  template <typename OtherT>
  friend constexpr bool operator==(const xyz &, const feature_two::xyz<OtherT> &) {
    return std::is_same_v<T, OtherT>;
  }
};
} // namespace feature_two

void foo() {
  properties pl1{feature_one::foo{}, feature_two::foo<42>{},
                 feature_two::bar{43}};

  static_assert(pl1.has_property<feature_one::foo>());
  static_assert(pl1.has_property<feature_two::foo>());
  static_assert(pl1.has_property<feature_two::bar>());

  // compile-time properties have "static" get_feature:
  constexpr auto f1_foo = pl1.get_property<feature_one::foo>();
  constexpr auto f2_foo = pl1.get_property<feature_two::foo>();
  // run-time property:
  auto f2_bar = pl1.get_property<feature_two::bar>();

  static_assert(f2_foo.value == 42);
  assert(f2_bar.value == 43);

  static_assert(!pl1.has_property<feature_two::xyz>());

  auto pl2 = pl1 + feature_two::xyz<float>{};
  static_assert(pl2.has_property<feature_two::xyz>());
  constexpr auto f2_xyz = pl2.get_property<feature_two::xyz>();
  static_assert(std::is_same_v<decltype(f2_xyz)::type, float>);
  static_assert(f2_xyz == feature_two::xyz<float>{});
  static_assert(!(f2_xyz == feature_two::xyz<int>{}));
}
