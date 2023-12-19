// RUN: %clangxx -fsycl -fsyntax-only %s
//==---------------- type_list.cpp - SYCL type_list test -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <type_traits>

using namespace std;

namespace s = sycl;
namespace d = sycl::detail;

template <template <typename, typename> class Predicate, typename T,
          typename T2, bool Expected = true>
void test_predicate() {
  static_assert(Predicate<T, T2>::value == Expected, "");
}

template <template <typename, typename> class Trait, typename TL, typename T,
          typename ExpectedT, bool Expected = true>
void test_trait() {
  static_assert(is_same<Trait<TL, T>, ExpectedT>::value == Expected, "");
}

int main() {
  // test list of types
  using scalar_float = d::type_list<float>;
  using vector_float = d::type_list<s::vec<float, 1>, s::vec<float, 2>>;
  using float_list = d::tl_append<scalar_float, vector_float>;
  using scalar_double = d::type_list<double>;
  using vector_double = d::type_list<s::vec<double, 1>, s::vec<double, 2>>;
  using double_list = d::tl_append<scalar_double, vector_double>;
  using floating_list = d::tl_append<float_list, double_list>;

  using scalar_int = d::type_list<int>;
  using scalar_long = d::type_list<long>;
  using vector_int = d::type_list<s::vec<int, 1>, s::vec<int, 2>>;
  using vector_long = d::type_list<s::vec<long, 1>, s::vec<long, 2>>;
  using int_list = d::tl_append<scalar_int, vector_int>;
  using long_list = d::tl_append<scalar_long, vector_long>;
  using integer_list = d::tl_append<int_list, long_list>;

  using types = d::tl_append<floating_list, integer_list>;

  static_assert(d::is_contained<float, scalar_float>::value, "");
  static_assert(d::is_contained<s::vec<float, 2>, scalar_float>::value == false,
                "");
  static_assert(d::is_contained<float, vector_float>::value == false, "");
  static_assert(d::is_contained<s::vec<float, 2>, vector_float>::value, "");
  static_assert(d::is_contained<float, types>::value, "");
  static_assert(d::is_contained<s::vec<float, 2>, types>::value, "");
  static_assert(d::is_contained<bool, types>::value == false, "");

  // test list of non-type
  using my_int_list = d::value_list<int, 3, 1, -2>;
  static_assert(my_int_list::head == 3, "");
  static_assert(my_int_list::tail::head == 1, "");
  static_assert(my_int_list::tail::tail::head == -2, "");

  using my_bool_list = d::value_list<bool, false, true, false>;
  static_assert(my_bool_list::head == false, "");
  static_assert(my_bool_list::tail::head == true, "");
  static_assert(my_bool_list::tail::tail::head == false, "");

  static_assert(d::is_contained_value<int, 4, my_int_list>::value == false, "");
  static_assert(d::is_contained_value<int, 1, my_int_list>::value, "");
  static_assert(d::is_contained_value<bool, false, my_bool_list>::value, "");
  static_assert(d::is_contained_value<bool, true, my_bool_list>::value, "");

  test_predicate<d::is_type_size_equal, int8_t, s::opencl::cl_char>();
  test_predicate<d::is_type_size_equal, int16_t, s::opencl::cl_char, false>();
  test_predicate<d::is_type_size_double_of, int8_t, s::opencl::cl_char,
                 false>();
  test_predicate<d::is_type_size_double_of, int16_t, s::opencl::cl_char>();
  test_predicate<d::is_type_size_double_of, int32_t, s::opencl::cl_char,
                 false>();

  // if void is found, the required type is not found
  test_trait<d::find_same_size_type_t, d::type_list<int8_t>, s::opencl::cl_char,
             int8_t>();
  test_trait<d::find_same_size_type_t, d::type_list<int16_t>,
             s::opencl::cl_char, void>();
  test_trait<d::find_twice_as_large_type_t, d::type_list<int8_t, int16_t>,
             s::opencl::cl_char, int16_t>();
  test_trait<d::find_twice_as_large_type_t, d::type_list<int8_t, int32_t>,
             s::opencl::cl_char, void>();

  return 0;
}
