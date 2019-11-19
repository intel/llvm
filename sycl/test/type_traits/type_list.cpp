// RUN: %clangxx -fsycl %s -o %t.out
//==---------------- type_list.cpp - SYCL type_list test -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <type_traits>

using namespace std;

namespace s = cl::sycl;
namespace d = cl::sycl::detail;

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
  static_assert(is_same<scalar_float::head, float>::value, "");
  static_assert(is_same<scalar_float::tail, d::empty_type_list>::value, "");

  using vector_float = d::type_list<s::vec<float, 1>, s::vec<float, 2>>;
  static_assert(is_same<vector_float::head, s::vec<float, 1>>::value, "");
  static_assert(is_same<vector_float::tail::head, s::vec<float, 2>>::value, "");
  static_assert(is_same<vector_float::tail::tail, d::empty_type_list>::value,
                "");

  using float_list = d::type_list<scalar_float, vector_float>;
  static_assert(is_same<float_list::head, float>::value, "");
  static_assert(is_same<float_list::tail::head, s::vec<float, 1>>::value, "");
  static_assert(is_same<float_list::tail::tail::head, s::vec<float, 2>>::value,
                "");
  static_assert(
      is_same<float_list::tail::tail::tail, d::empty_type_list>::value, "");

  using scalar_double = d::type_list<double>;
  using vector_double = d::type_list<s::vec<double, 1>, s::vec<double, 2>>;
  using double_list = d::type_list<scalar_double, vector_double>;
  using floating_list = d::type_list<float_list, double_list>;
  static_assert(is_same<floating_list::head, float>::value, "");
  static_assert(is_same<floating_list::tail::head, s::vec<float, 1>>::value,
                "");
  static_assert(
      is_same<floating_list::tail::tail::head, s::vec<float, 2>>::value, "");
  static_assert(is_same<floating_list::tail::tail::tail::head, double>::value,
                "");
  static_assert(is_same<floating_list::tail::tail::tail::tail::head,
                        s::vec<double, 1>>::value,
                "");
  static_assert(is_same<floating_list::tail::tail::tail::tail::tail::head,
                        s::vec<double, 2>>::value,
                "");
  static_assert(is_same<floating_list::tail::tail::tail::tail::tail::tail,
                        d::empty_type_list>::value,
                "");

  using scalar_int = d::type_list<int>;
  using scalar_long = d::type_list<long>;
  using vector_int = d::type_list<s::vec<int, 1>, s::vec<int, 2>>;
  using vector_long = d::type_list<s::vec<long, 1>, s::vec<long, 2>>;
  using int_list = d::type_list<scalar_int, vector_int>;
  using long_list = d::type_list<scalar_long, vector_long>;
  using integer_list = d::type_list<int_list, long_list>;
  static_assert(is_same<integer_list::head, int>::value, "");
  static_assert(is_same<integer_list::tail::head, s::vec<int, 1>>::value, "");
  static_assert(is_same<integer_list::tail::tail::head, s::vec<int, 2>>::value,
                "");
  static_assert(is_same<integer_list::tail::tail::tail::head, long>::value, "");
  static_assert(is_same<integer_list::tail::tail::tail::tail::head,
                        s::vec<long, 1>>::value,
                "");
  static_assert(is_same<integer_list::tail::tail::tail::tail::tail::head,
                        s::vec<long, 2>>::value,
                "");
  static_assert(is_same<integer_list::tail::tail::tail::tail::tail::tail,
                        d::empty_type_list>::value,
                "");

  using types = d::type_list<floating_list, integer_list>;
  static_assert(is_same<types::head, float>::value, "");
  static_assert(is_same<types::tail::head, s::vec<float, 1>>::value, "");
  static_assert(is_same<types::tail::tail::head, s::vec<float, 2>>::value, "");
  static_assert(is_same<types::tail::tail::tail::head, double>::value, "");
  static_assert(
      is_same<types::tail::tail::tail::tail::head, s::vec<double, 1>>::value,
      "");
  static_assert(is_same<types::tail::tail::tail::tail::tail::head,
                        s::vec<double, 2>>::value,
                "");
  static_assert(
      is_same<types::tail::tail::tail::tail::tail::tail::head, int>::value, "");
  static_assert(is_same<types::tail::tail::tail::tail::tail::tail::tail::head,
                        s::vec<int, 1>>::value,
                "");
  static_assert(
      is_same<types::tail::tail::tail::tail::tail::tail::tail::tail::head,
              s::vec<int, 2>>::value,
      "");
  static_assert(
      is_same<types::tail::tail::tail::tail::tail::tail::tail::tail::tail::head,
              long>::value,
      "");
  static_assert(is_same<types::tail::tail::tail::tail::tail::tail::tail::tail::
                            tail::tail::head,
                        s::vec<long, 1>>::value,
                "");
  static_assert(is_same<types::tail::tail::tail::tail::tail::tail::tail::tail::
                            tail::tail::tail::head,
                        s::vec<long, 2>>::value,
                "");
  static_assert(is_same<types::tail::tail::tail::tail::tail::tail::tail::tail::
                            tail::tail::tail::tail,
                        d::empty_type_list>::value,
                "");

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

  test_predicate<d::is_type_size_equal, int8_t, s::cl_char>();
  test_predicate<d::is_type_size_equal, int16_t, s::cl_char, false>();
  test_predicate<d::is_type_size_greater, int8_t, s::cl_char, false>();
  test_predicate<d::is_type_size_greater, int32_t, s::cl_char>();
  test_predicate<d::is_type_size_double_of, int8_t, s::cl_char, false>();
  test_predicate<d::is_type_size_double_of, int16_t, s::cl_char>();
  test_predicate<d::is_type_size_double_of, int32_t, s::cl_char, false>();
  test_predicate<d::is_type_size_less, int8_t, s::cl_int>();
  test_predicate<d::is_type_size_less, int32_t, s::cl_int, false>();
  test_predicate<d::is_type_size_half_of, int8_t, s::cl_int, false>();
  test_predicate<d::is_type_size_half_of, int16_t, s::cl_int>();
  test_predicate<d::is_type_size_half_of, int32_t, s::cl_int, false>();

  // if void is found, the required type is not found
  test_trait<d::find_same_size_type_t, d::type_list<int8_t>, s::cl_char,
             int8_t>();
  test_trait<d::find_same_size_type_t, d::type_list<int16_t>, s::cl_char,
             void>();
  test_trait<d::find_larger_type_t, d::type_list<int8_t, int16_t>, s::cl_char,
             int16_t>();
  test_trait<d::find_larger_type_t, d::type_list<int8_t>, s::cl_char, void>();
  test_trait<d::find_twice_as_large_type_t, d::type_list<int8_t, int16_t>,
             s::cl_char, int16_t>();
  test_trait<d::find_twice_as_large_type_t, d::type_list<int8_t, int32_t>,
             s::cl_char, void>();
  test_trait<d::find_smaller_type_t, d::type_list<int8_t>, s::cl_int, int8_t>();
  test_trait<d::find_smaller_type_t, d::type_list<int32_t>, s::cl_int, void>();
  test_trait<d::find_twice_as_small_type_t, d::type_list<int8_t, int16_t>,
             s::cl_int, int16_t>();
  test_trait<d::find_twice_as_small_type_t, d::type_list<int8_t, int32_t>,
             s::cl_int, void>();

  return 0;
}
