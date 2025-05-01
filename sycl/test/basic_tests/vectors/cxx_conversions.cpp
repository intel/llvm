// RUN: %clangxx -fsycl -fsyntax-only %s -fpreview-breaking-changes
// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

template <class From, class To, class = void>
struct is_explicitly_convertible_to_impl : std::false_type {};

template <class From, class To>
struct is_explicitly_convertible_to_impl<
    From, To, std::void_t<decltype(static_cast<To>(std::declval<From>()))>>
    : std::true_type {};

template <class From, class To>
struct is_explicitly_convertible_to
    : is_explicitly_convertible_to_impl<From, To> {};

template <class From, class To>
inline constexpr bool is_explicitly_convertible_to_v =
    is_explicitly_convertible_to<From, To>::value;

using sycl::half;
using sycl::vec;

void f_half_v1(vec<half, 1>);
void f_half_v4(vec<half, 4>);
void f_float_v1(vec<float, 1>);
void f_float_v4(vec<float, 4>);

using sw_half_1 = decltype(std::declval<vec<half, 4>>().swizzle<1>());
using sw_half_2 = decltype(std::declval<vec<half, 4>>().swizzle<1, 2>());

using sw_float_1 = decltype(std::declval<vec<float, 4>>().swizzle<1>());
using sw_float_2 = decltype(std::declval<vec<float, 4>>().swizzle<1, 2>());

using sw_double_1 = decltype(std::declval<vec<double, 4>>().swizzle<1>());
using sw_double_2 = decltype(std::declval<vec<double, 4>>().swizzle<1, 2>());

#if __INTEL_PREVIEW_BREAKING_CHANGES
#define EXCEPT_IN_PREVIEW !
#define PREVIEW_ONLY
#else
#define EXCEPT_IN_PREVIEW
#define PREVIEW_ONLY !
#endif

// clang-format off

//            IN_PREVIEW_ONLY   condition<>
//            EXCEPT_IN_PREVIEW condition<>

static_assert(                  std::is_invocable_v<decltype(f_half_v1), half>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_half_v1), float>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_half_v1), double>);
static_assert(                  std::is_invocable_v<decltype(f_half_v1), sw_half_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_half_v1), sw_float_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_half_v1), sw_double_1>);
static_assert(                  std::is_invocable_v<decltype(f_half_v1), vec<half, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_half_v1), vec<float, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_half_v1), vec<double, 1>>);

static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_float_v1), half>);
static_assert(                  std::is_invocable_v<decltype(f_float_v1), float>);
static_assert(                  std::is_invocable_v<decltype(f_float_v1), double>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_float_v1), sw_half_1>);
static_assert(                  std::is_invocable_v<decltype(f_float_v1), sw_float_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_float_v1), sw_double_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_float_v1), vec<half, 1>>);
static_assert(                  std::is_invocable_v<decltype(f_float_v1), vec<float, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_invocable_v<decltype(f_float_v1), vec<double, 1>>);

static_assert(                 !std::is_invocable_v<decltype(f_half_v4), half>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), float>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), double>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_half_1>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_float_1>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_double_1>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_half_2, sw_half_2>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_float_2, sw_float_2>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_double_2, sw_double_2>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_half_2, sw_float_2>);
static_assert(                 !std::is_invocable_v<decltype(f_half_v4), sw_half_2, sw_double_2>);

static_assert(                 !std::is_invocable_v<decltype(f_float_v4), half>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), float>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), double>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_half_1>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_float_1>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_double_1>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_half_2, sw_half_2>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_float_2, sw_float_2>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_double_2, sw_double_2>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_float_2, sw_float_2>);
static_assert(                 !std::is_invocable_v<decltype(f_float_v4), sw_float_2, sw_double_2>);

static_assert(                  is_explicitly_convertible_to_v<half,          vec<half, 1>>);
static_assert(                  is_explicitly_convertible_to_v<float,         vec<half, 1>>);
static_assert(                  is_explicitly_convertible_to_v<double,        vec<half, 1>>);
static_assert(                  is_explicitly_convertible_to_v<sw_half_1,     vec<half, 1>>);
static_assert(                  is_explicitly_convertible_to_v<sw_float_1,    vec<half, 1>>);
static_assert(                  is_explicitly_convertible_to_v<sw_double_1,   vec<half, 1>>);

static_assert(                  is_explicitly_convertible_to_v<half,          vec<float, 1>>);
static_assert(                  is_explicitly_convertible_to_v<float,         vec<float, 1>>);
static_assert(                  is_explicitly_convertible_to_v<double,        vec<float, 1>>);
static_assert(                  is_explicitly_convertible_to_v<sw_half_1,     vec<float, 1>>);
static_assert(                  is_explicitly_convertible_to_v<sw_float_1,    vec<float, 1>>);
static_assert(                  is_explicitly_convertible_to_v<sw_double_1,   vec<float, 1>>);

static_assert(                  is_explicitly_convertible_to_v<half,          vec<half, 4>>);
static_assert(                  is_explicitly_convertible_to_v<float,         vec<half, 4>>);
static_assert(                  is_explicitly_convertible_to_v<double,        vec<half, 4>>);
static_assert(                  is_explicitly_convertible_to_v<sw_half_1,     vec<half, 4>>);
static_assert(                  is_explicitly_convertible_to_v<sw_float_1,    vec<half, 4>>);
static_assert(                  is_explicitly_convertible_to_v<sw_double_1,   vec<half, 4>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_half_2,    vec<half, 4>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_float_2,   vec<half, 4>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_double_2,  vec<half, 4>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_half_2,    vec<half, 4>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_half_2,    vec<half, 4>>);
static_assert(                  is_explicitly_convertible_to_v<sw_half_2,     vec<half, 2>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_float_2,   vec<half, 2>>);
static_assert(                 !is_explicitly_convertible_to_v<sw_double_2,  vec<half, 2>>);
static_assert(                  is_explicitly_convertible_to_v<sw_half_2,     vec<half, 2>>);
static_assert(                  is_explicitly_convertible_to_v<sw_half_2,     vec<half, 2>>);

static_assert(                  is_explicitly_convertible_to_v<vec<half, 1>,  half>);
#if __SYCL_DEVICE_ONLY__
static_assert(                  is_explicitly_convertible_to_v<vec<half, 1>,  float>);
static_assert(                  is_explicitly_convertible_to_v<vec<half, 1>,  double>);
#else
static_assert(PREVIEW_ONLY      is_explicitly_convertible_to_v<vec<half, 1>,  float>);
static_assert(PREVIEW_ONLY      is_explicitly_convertible_to_v<vec<half, 1>,  double>);
#endif

static_assert(                  is_explicitly_convertible_to_v<vec<float, 1>,  half>);
static_assert(                  is_explicitly_convertible_to_v<vec<float, 1>,  float>);
static_assert(                  is_explicitly_convertible_to_v<vec<float, 1>,  double>);
