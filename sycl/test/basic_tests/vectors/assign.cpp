// RUN: %clangxx -fsycl -fsyntax-only %s -fpreview-breaking-changes
// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

using sycl::half;
using sycl::vec;
using sw_half_1 = decltype(std::declval<vec<half, 4>>().swizzle<1>());
using sw_half_2 = decltype(std::declval<vec<half, 4>>().swizzle<1, 2>());

using sw_float_1 = decltype(std::declval<vec<float, 4>>().swizzle<1>());
using sw_float_2 = decltype(std::declval<vec<float, 4>>().swizzle<1, 2>());

using sw_double_1 = decltype(std::declval<vec<double, 4>>().swizzle<1>());
using sw_double_2 = decltype(std::declval<vec<double, 4>>().swizzle<1, 2>());

#if __INTEL_PREVIEW_BREAKING_CHANGES
#define EXCEPT_IN_PREVIEW !
#else
#define EXCEPT_IN_PREVIEW
#endif

// clang-format off

//            IN_PREVIEW_ONLY   condition<>
//            EXCEPT_IN_PREVIEW condition<>

static_assert(                  std::is_assignable_v<vec<half, 1>, half>);
static_assert(                  std::is_assignable_v<vec<half, 1>, float>);
static_assert(                  std::is_assignable_v<vec<half, 1>, double>);
static_assert(                  std::is_assignable_v<vec<half, 1>, vec<half, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<half, 1>, vec<float, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<half, 1>, vec<double, 1>>);
static_assert(                  std::is_assignable_v<vec<half, 1>, sw_half_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<half, 1>, sw_float_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<half, 1>, sw_double_1>);
static_assert(                 !std::is_assignable_v<vec<half, 1>, sw_half_2>);
static_assert(                 !std::is_assignable_v<vec<half, 1>, sw_float_2>);
static_assert(                 !std::is_assignable_v<vec<half, 1>, sw_double_2>);

static_assert(                  std::is_assignable_v<vec<half, 2>, half>);
static_assert(                  std::is_assignable_v<vec<half, 2>, float>);
static_assert(                  std::is_assignable_v<vec<half, 2>, double>);
static_assert(                  std::is_assignable_v<vec<half, 2>, vec<half, 1>>);
static_assert(                 !std::is_assignable_v<vec<half, 2>, vec<float, 1>>);
static_assert(                 !std::is_assignable_v<vec<half, 2>, vec<double, 1>>);
static_assert(                  std::is_assignable_v<vec<half, 2>, sw_half_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<half, 2>, sw_float_1>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<half, 2>, sw_double_1>);
static_assert(                  std::is_assignable_v<vec<half, 2>, sw_half_2>);
static_assert(                 !std::is_assignable_v<vec<half, 2>, sw_float_2>);
static_assert(                 !std::is_assignable_v<vec<half, 2>, sw_double_2>);

static_assert(                  std::is_assignable_v<vec<float, 1>, half>);
static_assert(                  std::is_assignable_v<vec<float, 1>, float>);
static_assert(                  std::is_assignable_v<vec<float, 1>, double>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<float, 1>, vec<half, 1>>);
static_assert(                  std::is_assignable_v<vec<float, 1>, vec<float, 1>>);
static_assert(                  std::is_assignable_v<vec<float, 1>, vec<double, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<float, 1>, sw_half_1>);
static_assert(                  std::is_assignable_v<vec<float, 1>, sw_float_1>);
static_assert(                  std::is_assignable_v<vec<float, 1>, sw_double_1>);
static_assert(                 !std::is_assignable_v<vec<float, 1>, sw_half_2>);
static_assert(                 !std::is_assignable_v<vec<float, 1>, sw_float_2>);
static_assert(                 !std::is_assignable_v<vec<float, 1>, sw_double_2>);

static_assert(                  std::is_assignable_v<vec<float, 2>, half>);
static_assert(                  std::is_assignable_v<vec<float, 2>, float>);
static_assert(                  std::is_assignable_v<vec<float, 2>, double>);
#if __SYCL_DEVICE_ONLY__
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<float, 2>, vec<half, 1>>);
#else
static_assert(                 !std::is_assignable_v<vec<float, 2>, vec<half, 1>>);
#endif
static_assert(                  std::is_assignable_v<vec<float, 2>, vec<float, 1>>);
static_assert(                  std::is_assignable_v<vec<float, 2>, vec<double, 1>>);
static_assert(EXCEPT_IN_PREVIEW std::is_assignable_v<vec<float, 2>, sw_half_1>);
static_assert(                  std::is_assignable_v<vec<float, 2>, sw_float_1>);
static_assert(                  std::is_assignable_v<vec<float, 2>, sw_double_1>);
static_assert(                 !std::is_assignable_v<vec<float, 2>, sw_half_2>);
static_assert(                  std::is_assignable_v<vec<float, 2>, sw_float_2>);
static_assert(                 !std::is_assignable_v<vec<float, 2>, sw_double_2>);
