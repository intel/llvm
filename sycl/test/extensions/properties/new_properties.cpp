// RUN: %clangxx -fsycl -fsyntax-only %s

#define __SYCL_EBO
#include <sycl/ext/oneapi/properties/new_properties.hpp>
#include <sycl/detail/type_traits.hpp>

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

template <typename property_t,
          typename property_key_t =
              detail::property_key_non_template<property_t>>
struct named_property_base
    : public detail::property_base<property_t, property_key_t> {
  static constexpr std::string_view property_name =
      generate_property_key_name<property_key_t>();
};

namespace test_sorting {
// Treat each instantiation as a separate property by not providing a common
// key.
template <int N> struct Property : named_property_base<Property<N>> {};
static_assert(
    std::is_same_v<decltype(properties{Property<3>{}, Property<2>{}}),
                   decltype(properties{Property<2>{}, Property<3>{}})>);
} // namespace test_sorting

namespace test {
struct property1 : named_property_base<property1> {};

template <int N>
struct property2
    : named_property_base<property2<N>,
                          detail::property_key_value_template<property2>> {};
struct property3 : named_property_base<property3> {
  property3(int x) : x(x) {}
  int x;
};

void test() {
  property1 p1;
  property2<42> p2;
  property3 p3{11};

  properties pl1{p1, p2, p3};
  properties pl2{p3};

  static_assert(pl1.has_property<property1>());
  static_assert(!pl2.has_property<property1>());

  static_assert(pl1.has_property<property2>());
  static_assert(!pl2.has_property<property2>());

  static_assert(pl1.has_property<property3>());
  static_assert(pl2.has_property<property3>());
}
} // namespace test

namespace bench {
template <int N> struct property : named_property_base<property<N>> {
  static constexpr int value() { return N; }
};

template <int... N> void test(std::integer_sequence<int, N...>) {
  properties pl{property<N>{}...};
  static_assert((pl.template has_property<property<N>>() && ...));
  static_assert(
      ((pl.template get_property<property<N>>().value() == N) && ...));
}
} // namespace bench

namespace test_operator_plus {
template <int N> struct property : named_property_base<property<N>> {};

constexpr properties pl1{property<1>{}, property<2>{}, property<3>{}};
constexpr properties pl2 = pl1 + properties{property<4>{}};
static_assert(!pl1.has_property<property<4>>());
static_assert(pl2.has_property<property<2>>());
static_assert(pl2.has_property<property<4>>());
} // namespace test_operator_plus

namespace test_compile_prop_in_runtime_list {
template <int N>
struct ct_prop
    : named_property_base<ct_prop<N>,
                          detail::property_key_value_template<ct_prop>> {
  static constexpr auto value() { return N; }
};
struct rt_prop : named_property_base<rt_prop> {
  rt_prop(int N) : x(N) {}

  int x;

  constexpr auto value() { return x; }
};
void test() {
  int x = 42;
  properties pl{ct_prop<42>{}, rt_prop{x}};
  constexpr auto p = pl.get_property<ct_prop>();
  static_assert(std::is_same_v<decltype(p), const ct_prop<42>>);
  static_assert(p.value() == 42);
}
} // namespace test_compile_prop_in_runtime_list

namespace test_static_get_property {
  struct ct_prop : named_property_base<ct_prop> {};
  struct rt_prop : named_property_base<rt_prop> {
    int x;
  };
  void test() {
    properties pl{ct_prop{}, rt_prop{}};
    constexpr auto c = decltype(pl)::get_property<ct_prop>();
    auto r = pl.get_property<rt_prop>();
  }
}

namespace properties_validation_example {
struct prop : named_property_base<prop> {};
struct prop2 : named_property_base<prop2> {};

template <typename> struct foo;
template <typename... property_tys>
struct foo<properties<detail::properties_type_list<property_tys...>>> {
  template <typename... key_tys> static constexpr bool bar(key_tys...) {
    return ((sycl::detail::check_type_in_v<typename property_tys::key_t,
                                           key_tys...> &&
             ...));
  }
};
constexpr properties pl{prop{}};
using ty = std::remove_const_t<decltype(pl)>;
static_assert(foo<ty>::bar(detail::key<prop>()));
static_assert(foo<ty>::bar(detail::key<prop>(), detail::key<prop2>()));
static_assert(foo<empty_properties_t>::bar(detail::key<prop>()));
static_assert(!foo<ty>::bar(detail::key<prop2>()));

// static_assert(all_properties_in_v<ty, prop>);
// static_assert(all_properties_in_v<empty_properties_t, prop>);
// static_assert(!all_properties_in_v<ty, prop2>);
} // namespace properties_validation_example

namespace test_combine_op {
struct prop : named_property_base<prop> {};
struct prop2 : named_property_base<prop2> {};
using pl = decltype(properties{prop{}, prop2{}});

static_assert(std::is_same_v<pl, decltype(prop{} + prop2{})>);
static_assert(std::is_same_v<pl, decltype(prop2{} + prop{})>);
static_assert(std::is_same_v<decltype(properties{prop{}}),
                             decltype(empty_properties_t{} + prop{})>);
static_assert(
    std::is_same_v<pl, decltype(empty_properties_t{} + prop{} + prop2{})>);
static_assert(
    std::is_same_v<pl, decltype(empty_properties_t{} + prop2{} + prop{})>);

static_assert(std::is_same_v<pl, decltype(properties{prop{}} + prop2{})>);
static_assert(std::is_same_v<pl, decltype(properties{prop2{}} + prop{})>);

static_assert(std::is_same_v<decltype(properties{prop{}} + prop2{}), pl>);
}

namespace test_inheritance_visibility {
template <int N> struct prop : named_property_base<prop<N>> {
  static constexpr int value = N;
};

template <typename T, typename = void> struct has_value : std::false_type {};
template <typename T>
struct has_value<T, std::void_t<decltype(T::value)>> : std::true_type {};
template <typename T> inline constexpr bool has_value_v = has_value<T>::value;

constexpr properties pl1{prop<1>{}};
constexpr properties pl2{prop<1>{}, prop<2>{}};

static_assert(has_value_v<prop<1>>);
static_assert(!has_value_v<decltype(pl1)>);
static_assert(!has_value_v<decltype(pl2)>);
}

namespace implicit_key {
struct non_template_prop : named_property_base<non_template_prop> {};
template <int N>
struct prop
    : named_property_base<prop<N>, detail::property_key_value_template<prop>> {
  static constexpr int value = N;
};
constexpr properties pl1{non_template_prop{}};
constexpr properties pl2{prop<42>{}};
static_assert(pl1.has_property<non_template_prop>());
static_assert(!pl1.has_property<prop>());
static_assert(!pl2.has_property<non_template_prop>());
static_assert(pl2.has_property<prop>());
static_assert(!empty_properties_t::has_property<non_template_prop>());
static_assert(!empty_properties_t::has_property<prop>());

constexpr auto p1 = pl1.get_property<non_template_prop>();
static_assert(std::is_same_v<decltype(p1), const non_template_prop>);
constexpr auto p2 = pl2.get_property<prop>();
static_assert(std::is_same_v<decltype(p2), const prop<42>>);
} // namespace implicit_key

int main() {
  test::test();
  bench::test(std::make_integer_sequence<int, 45>{});
  // More than that fails with clang, e.g.
  // clang-format off
  // new_properties.cpp:165:10: note: in instantiation of function template specialization 'bench::test<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  //       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  //       52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67>' requested here
  //   165 |   bench::test(std::make_integer_sequence<int, 68>{});
  //       |          ^
  // /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/bits/char_traits.h:367:7: note: constexpr evaluation hit maximum step limit;
  //       possible infinite loop?
  //   367 |       {
  //       |       ^
  // /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/bits/char_traits.h:382:12: note: in call to 'lt(__PRETTY_FUNCTION__[22],
  //       __PRETTY_FUNCTION__[22])'
  //   382 |               if (lt(__s1[__i], __s2[__i]))
  //       |                   ^~~~~~~~~~~~~~~~~~~~~~~~
  // /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/string_view:320:14: note: in call to 'compare(&__PRETTY_FUNCTION__[0],
  //       &__PRETTY_FUNCTION__[0], 71)'
  //   320 |         int __ret = traits_type::compare(this->_M_str, __str._M_str, __rlen);
  //       |                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // /usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/string_view:596:14: note: in call to '__x.compare({71, &__PRETTY_FUNCTION__[0]})'
  //   596 |     { return __x.compare(__y) < 0; }
  //       |              ^~~~~~~~~~~~~~~~
  // /iusers/aeloviko/sycl/sycl/include/sycl/ext/oneapi/properties/properties.hpp:78:13: note: in call to 'operator<<char,
  //       std::char_traits<char>>({71, &__PRETTY_FUNCTION__[0]}, {71, &__PRETTY_FUNCTION__[0]})'
  //    78 |         if (to_sort[j].first < to_sort[i].first)
  //       |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // clang-format on
  // bench::test(std::make_integer_sequence<int, 100>{});

  properties empty_props{};
}
