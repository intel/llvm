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

namespace test_group_load_store {
struct naive : detail::property_base<naive> {
  static constexpr mock_property_sort_key_t sort_key = 1;
};
struct full_group : detail::property_base<full_group> {
  static constexpr mock_property_sort_key_t sort_key = 2;
};
constexpr properties pl1{full_group{}};
constexpr properties pl2{pl1, naive{}};
static_assert(pl1.template has_property<full_group>());
static_assert(!pl1.template has_property<naive>());
static_assert(pl2.template has_property<full_group>());
static_assert(pl2.template has_property<naive>());

enum class data_placement { blocked, striped };
template <data_placement placement>
struct data_placement_property
    : detail::property_base<data_placement_property<placement>,
                            struct data_placement_property_key> {
  static constexpr mock_property_sort_key_t sort_key = 3;
  static constexpr bool is_blocked() {
    return placement == data_placement::blocked;
  }
};
inline constexpr data_placement_property<data_placement::blocked> blocked;
inline constexpr data_placement_property<data_placement::striped> striped;

static_assert(properties{naive{}, blocked}
                  .get_property<struct data_placement_property_key>()
                  .is_blocked());
static_assert(!properties{naive{}, striped}
                   .get_property<struct data_placement_property_key>()
                   .is_blocked());
static_assert(
    properties{naive{}, blocked}
        .get_property_or_default_to<struct data_placement_property_key>(blocked)
        .is_blocked());
static_assert(
    !properties{naive{}, data_placement_property<data_placement::striped>{}}
         .get_property_or_default_to<struct data_placement_property_key>(
             blocked)
         .is_blocked());
static_assert(
    properties{naive{}}
        .get_property_or_default_to<struct data_placement_property_key>(blocked)
        .is_blocked());
static_assert(
    !properties{naive{}}
         .get_property_or_default_to<struct data_placement_property_key>(
             striped)
         .is_blocked());

constexpr properties pl3{full_group{}, blocked};
// constexpr properties pl4{pl3, naive{}};
template <typename... other_property_list_tys, typename... other_property_tys>
constexpr auto merge_properties(
    properties<detail::properties_type_list<other_property_list_tys...>>,
    other_property_tys...) {
  return 42;
}
static_assert(merge_properties(pl3, naive{}) == 42);
} // namespace test_group_load_store

namespace test_merge_ctor {
template <int N> struct property : detail::property_base<property<N>> {
  static constexpr int sort_key = N;
};

constexpr properties pl1{property<1>{}, property<2>{}, property<3>{}};
constexpr properties pl2{pl1, property<4>{}};
} // namespace test_merge_ctor

namespace test_compile_prop_in_runtime_list {
template <int N>
struct ct_prop : detail::property_base<ct_prop<N>, struct ct_prop_key> {
  static constexpr int sort_key = 1;

  static constexpr auto value() { return N; }
};
struct rt_prop : detail::property_base<rt_prop> {
  static constexpr int sort_key = 2;
  rt_prop(int N) : x(N) {}

  int x;

  constexpr auto value() { return x; }
};
void test() {
  int x = 42;
  properties pl{ct_prop<42>{}, rt_prop{x}};
  constexpr auto p = pl.get_property<struct ct_prop_key>();
  static_assert(std::is_same_v<decltype(p), const ct_prop<42>>);
  static_assert(p.value() == 42);
}
} // namespace test_compile_prop_in_runtime_list

int main() {
  test::test();
  bench::test(std::make_integer_sequence<int, 100>{});

  properties empty_props{};
}
