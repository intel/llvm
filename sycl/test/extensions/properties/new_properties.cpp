// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/ext/oneapi/properties/properties.hpp>

using namespace sycl::ext::oneapi::experimental::new_properties;

using mock_property_sort_key_t = int;

namespace test_sorting {
template <int N> struct Property : detail::property_base<Property<N>> {
  static constexpr mock_property_sort_key_t sort_key = N;
};
static_assert(
    std::is_same_v<decltype(properties{Property<3>{}, Property<2>{}}),
                   decltype(properties{Property<2>{}, Property<3>{}})>);
} // namespace test_sorting

namespace test {
struct property1 : detail::property_base<property1> {
  static constexpr mock_property_sort_key_t sort_key = 1;
};

template <int N>
struct property2 : detail::property_base<property2<N>, struct property2_key> {
  static constexpr mock_property_sort_key_t sort_key = 2;
};

struct property3 : detail::property_base<property3> {
  static constexpr mock_property_sort_key_t sort_key = 3;
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

  static_assert(pl1.has_property<property2_key>());
  static_assert(!pl2.has_property<property2_key>());

  static_assert(pl1.has_property<property3>());
  static_assert(pl2.has_property<property3>());
}
} // namespace test

namespace bench {
template <int N> struct property : detail::property_base<property<N>> {
  static constexpr mock_property_sort_key_t sort_key = 1000 + N;
  static constexpr int value() { return N; }
};

template <int... N> void test(std::integer_sequence<int, N...>) {
  properties pl{property<N>{}...};
  static_assert((pl.template has_property<property<N>>() && ...));
  static_assert(
      ((pl.template get_property<property<N>>().value() == N) && ...));
}
} // namespace bench

int main() {
  test::test();
  bench::test(std::make_integer_sequence<int, 100>{});

  properties empty_props{};
}
