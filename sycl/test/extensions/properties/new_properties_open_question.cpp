// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/ext/oneapi/properties/properties.hpp>

using namespace sycl::ext::oneapi::experimental::new_properties;

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

struct prop_key {};

template <typename PropList> auto foo(PropList props = {}) {
  static_assert(props.template has_property<prop_key>());
  static_assert(props.template get_property<prop_key>().value == 42);
}

namespace approach_one {
// Don't provide `inline constexpr` property variables at all, they don't
// provide enough benefits now that `property_value` thing was eliminated.

template <int N> struct prop : named_property_base<prop<N>, prop_key> {
  static constexpr int value = N;
};
struct other_prop : named_property_base<other_prop> {};

void bar() {
  foo(properties{prop<42>{}});
  foo(properties{prop<42>{}, other_prop{}});
}
} // namespace approach_one

namespace approach_two {
// Keep providing `inline constexpr` "shortcuts".
template <int N>
struct prop_property : named_property_base<prop_property<N>, prop_key> {
  static constexpr int value = N;
};
struct other_prop_property : named_property_base<other_prop_property> {};

template <int N> inline constexpr prop_property<N> prop{};
inline constexpr other_prop_property other_prop{};

void bar() {
  // Still need `properties` here.
  foo(properties{prop<42>});
  foo(properties{prop<42>, other_prop});
}
} // namespace approach_two
namespace approach_three {
// Use operator+ to create `properties` property list from individual
// properties.
template <int N>
struct prop_property : named_property_base<prop_property<N>, prop_key> {
  static constexpr int value = N;
};
struct other_prop_property : named_property_base<other_prop_property> {};

template <int N> inline constexpr prop_property<N> prop{};
inline constexpr other_prop_property other_prop{};

void bar() {
  // Unary `+` has a bit of hacky feeling...
  foo(+prop<42>);
  // More than one property in a property list looks very natural.
  // Alternatively, that can be `operator|` but it has no unary version.
  foo(prop<42> + other_prop);
}
} // namespace approach_three

namespace approach_four {
// "Duck-typing" - make individual properties behave almost like a property
// list.

// This will be part of the implementation's `detail::property_base` but I'm
// keeping it in the test for now until the future direction is chosen.
template <typename property_t, typename property_key_t = property_t>
struct adjusted_property_base
    : public named_property_base<property_t, property_key_t> {
  template <typename T> static constexpr bool has_property() {
    return std::is_same_v<T, property_key_t>;
  }

  // Technically it should be two version static/non-static depending on
  // `std::is_empty_v<property_t>`, skipped here for brevity.
  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, property_key_t>>>
  static constexpr property_t get_property() {
    return property_t{};
  }
};

template <int N>
struct prop_property : adjusted_property_base<prop_property<N>, prop_key> {
  static constexpr int value = N;
};
struct other_prop_property : adjusted_property_base<other_prop_property> {};

template <int N> inline constexpr prop_property<N> prop{};
inline constexpr other_prop_property other_prop{};

void bar() {
  // "Duck-typing" here.
  foo(prop<42>);

  // Now nothing prevents us from using `operator|` here that someone might
  // found more natural. Not doing for simplicity for now.
  foo(prop<42> + other_prop);
}

// Problem:
template <typename T, typename PropListTy> struct fake_device_global;
template <typename T, typename... property_tys>
struct
    [[__sycl_detail__::global_variable_allowed,
      __sycl_detail__::add_ir_attributes_global_variable(
          // This attribute needs pack expansion and can't be made working with
          // "duck-typing".
          property_tys::ir_attribute_name...,
          property_tys::
              ir_attribute_value...)]] fake_device_global<T,
                                                          properties<detail::properties_type_list<
                                                              property_tys...>>> {
  T value;
};

// Needs this:
fake_device_global<int, decltype(properties{prop<42>})> fg_int1{43};

#if 0
// This fails. Note that "decltype" might be implicit through CTAD.
fake_device_global<int, decltype(prop<42>)> fg_int{43};
#endif
} // namespace approach_four

namespace approach_five {
// Same as approach three but make shortcuts property lists instead of
// individual properties.
template <int N>
struct prop_property : named_property_base<prop_property<N>, prop_key> {
  static constexpr int value = N;
};
struct other_prop_property : named_property_base<other_prop_property> {};

template <int N> inline constexpr auto prop = properties{prop_property<N>{}};
inline constexpr auto other_prop = properties{other_prop_property{}};

void bar() {
  foo(prop<42>);
  foo(prop<42> + other_prop);
}

// Problem:
struct rt_prop_property : named_property_base<rt_prop_property> {
  int x;
};

// What should we do in the shortcut? On the other hand, maybe it's a more
// generic problem and such shortcuts are impossible/meaningless for run-time
// properties anyway.

// Another side of the problem:
template <int N> inline constexpr auto expected_behavior = prop_property<N>{};
static_assert(expected_behavior<42>.value == 42);

#if 0
// This fails, `properties` don't expose individual property's interfaces.
// Approach six - modify `properties` to use "public" inheritance from a
// property + expose its ctors.
static_assert(prop<42>.value == 42);
#endif
}
