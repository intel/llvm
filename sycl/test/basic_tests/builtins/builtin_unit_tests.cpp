// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only %s -Xclang -verify
// REQUIRES: preview-breaking-changes-supported

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::detail;

namespace builtin_same_shape_v_tests {
using swizzle1 = decltype(std::declval<vec<float, 2>>().swizzle<0>());
using swizzle2 = decltype(std::declval<vec<float, 2>>().swizzle<0, 0>());
using swizzle3 = decltype(std::declval<vec<float, 2>>().swizzle<0, 0, 1>());

static_assert(builtin_same_shape_v<float>);
static_assert(builtin_same_shape_v<int, float>);
static_assert(builtin_same_shape_v<marray<int, 2>>);
static_assert(builtin_same_shape_v<marray<int, 2>, marray<float, 2>>);
static_assert(builtin_same_shape_v<vec<int, 2>>);
static_assert(builtin_same_shape_v<vec<int, 2>, vec<float, 2>>);
static_assert(builtin_same_shape_v<vec<int, 2>, swizzle2>);

static_assert(!builtin_same_shape_v<float, marray<float, 1>>);
static_assert(!builtin_same_shape_v<float, vec<float, 1>>);
static_assert(!builtin_same_shape_v<marray<float, 1>, vec<float, 1>>);
static_assert(!builtin_same_shape_v<float, swizzle1>);
static_assert(!builtin_same_shape_v<marray<float, 1>, swizzle1>);
static_assert(!builtin_same_shape_v<swizzle2, swizzle1>);
} // namespace builtin_same_shape_v_tests

namespace builtin_marray_impl_tests {
// Integer functions/relational bitselect only accept fixed-width integer
// element types for vector/swizzle elements. Make sure that our marray->vec
// delegator can handle that.

auto foo(char x) { return x; }
auto foo(signed char x) { return x; }
auto foo(unsigned char x) { return x; }
auto foo(vec<int8_t, 2> x) { return x; }
auto foo(vec<uint8_t, 2> x) { return x; }

auto test() {
  marray<char, 2> x;
  marray<signed char, 2> y;
  marray<unsigned char, 2> z;
  auto TestOne = [](auto x) {
    std::ignore = builtin_marray_impl([](auto x) { return foo(x); }, x);
  };
  TestOne(x);
  TestOne(y);
  TestOne(z);
}
} // namespace builtin_marray_impl_tests

namespace builtin_enable_integer_tests {
using swizzle1 = decltype(std::declval<vec<int8_t, 2>>().swizzle<0>());
using swizzle2 = decltype(std::declval<vec<int8_t, 2>>().swizzle<0, 0>());
template <typename... Ts> void ignore() {}

void test() {
  // clang-format off
  ignore<builtin_enable_integer_t<char>,
         builtin_enable_integer_t<signed char>,
         builtin_enable_integer_t<unsigned char>>();
  // clang-format on

  ignore<builtin_enable_integer_t<vec<int8_t, 2>>,
         builtin_enable_integer_t<vec<uint8_t, 2>>>();

  ignore<builtin_enable_integer_t<char, char>>();
  ignore<builtin_enable_integer_t<vec<int8_t, 2>, vec<int8_t, 2>>>();
  ignore<builtin_enable_integer_t<vec<int8_t, 2>, swizzle2>>();
  ignore<builtin_enable_integer_t<swizzle2, swizzle2>>();

  {
    // Only one of char/signed char maps onto int8_t. The other type isn't a
    // valid vector element type for integer builtins.

    static_assert(std::is_signed_v<char>);

    // clang-format off
    // expected-error-re@*:* {{no type named 'type' in 'sycl::detail::builtin_enable<sycl::detail::default_ret_type, sycl::detail::integer_elem_type, sycl::detail::any_shape, sycl::detail::same_elem_type, sycl::vec<{{.*}}, 2>>'}}
    // expected-note@+1 {{in instantiation of template type alias 'builtin_enable_integer_t' requested here}}
    ignore<builtin_enable_integer_t<vec<signed char, 2>>, builtin_enable_integer_t<vec<char, 2>>>();
    // clang-format on
  }

  // expected-error@*:* {{no type named 'type' in 'sycl::detail::builtin_enable<sycl::detail::default_ret_type, sycl::detail::integer_elem_type, sycl::detail::any_shape, sycl::detail::same_elem_type, char, signed char>'}}
  // expected-note@+1 {{in instantiation of template type alias 'builtin_enable_integer_t' requested here}}
  ignore<builtin_enable_integer_t<char, signed char>>();
}
} // namespace builtin_enable_integer_tests

namespace builtin_enable_bitselect_tests {
// Essentially the same as builtin_enable_integer_t + FP types support.
using swizzle1 = decltype(std::declval<vec<int8_t, 2>>().swizzle<0>());
using swizzle2 = decltype(std::declval<vec<int8_t, 2>>().swizzle<0, 0>());
template <typename... Ts> void ignore() {}

void test() {
  // clang-format off
  ignore<builtin_enable_bitselect_t<char>,
         builtin_enable_bitselect_t<signed char>,
         builtin_enable_bitselect_t<unsigned char>,
         builtin_enable_bitselect_t<float>>();
  // clang-format on

  ignore<builtin_enable_bitselect_t<vec<int8_t, 2>>,
         builtin_enable_bitselect_t<vec<uint8_t, 2>>,
         builtin_enable_bitselect_t<vec<float, 2>>>();

  ignore<builtin_enable_bitselect_t<char, char>>();
  ignore<builtin_enable_bitselect_t<vec<int8_t, 2>, vec<int8_t, 2>>>();
  ignore<builtin_enable_bitselect_t<vec<int8_t, 2>, swizzle2>>();
  ignore<builtin_enable_bitselect_t<swizzle2, swizzle2>>();

  {
    // Only one of char/signed char maps onto int8_t. The other type isn't a
    // valid vector element type for integer builtins.

    static_assert(std::is_signed_v<char>);

    // clang-format off
    // expected-error-re@*:* {{no type named 'type' in 'sycl::detail::builtin_enable<sycl::detail::default_ret_type, sycl::detail::bitselect_elem_type, sycl::detail::any_shape, sycl::detail::same_elem_type, sycl::vec<{{.*}}, 2>>'}}
    // expected-note@+1 {{in instantiation of template type alias 'builtin_enable_bitselect_t' requested here}}
    ignore<builtin_enable_bitselect_t<vec<signed char, 2>>, builtin_enable_bitselect_t<vec<char, 2>>>();
    // clang-format on
  }

  // expected-error@*:* {{no type named 'type' in 'sycl::detail::builtin_enable<sycl::detail::default_ret_type, sycl::detail::bitselect_elem_type, sycl::detail::any_shape, sycl::detail::same_elem_type, char, signed char>'}}
  // expected-note@+1 {{in instantiation of template type alias 'builtin_enable_bitselect_t' requested here}}
  ignore<builtin_enable_bitselect_t<char, signed char>>();
}
} // namespace builtin_enable_bitselect_tests
