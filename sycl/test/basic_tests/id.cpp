// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// RUN: %clangxx -D__SYCL_DISABLE_ID_TO_INT_CONV__ -fsycl %s -o %t_dis.out
// RUN: %t_dis.out

//==--------------- id.cpp - SYCL id test ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using sycl::detail::Builder;

using namespace std;
int main() {
  /* id()
   * Construct a SYCL id with the value 0 for each dimension. */
  sycl::id<1> one_dim_zero_id;
  assert(one_dim_zero_id.get(0) == 0);
  sycl::id<2> two_dim_zero_id;
  assert(two_dim_zero_id.get(0) == 0 && two_dim_zero_id.get(1) == 0);
  sycl::id<3> three_dim_zero_id;
  assert(three_dim_zero_id.get(0) == 0 && three_dim_zero_id.get(1) == 0 &&
         three_dim_zero_id.get(2) == 0);

  /* id(size_t dim0)
   * Construct a 1D id with value dim0. Only valid when the template parameter
   * dimensions is equal to 1 */
  sycl::id<1> one_dim_id(64);
  assert(one_dim_id.get(0) == 64);

  /* id(size_t dim0, size_t dim1)
   * Construct a 2D id with values dim0, dim1. Only valid when the template
   * parameter dimensions is equal to 2. */
  sycl::id<2> two_dim_id(128, 256);
  assert(two_dim_id.get(0) == 128 && two_dim_id.get(1) == 256);

  /* id(size_t dim0, size_t dim1, size_t dim2)
   * Construct a 3D id with values dim0, dim1, dim2. Only valid when the
   * template parameter dimensions is equal to 3. */
  sycl::id<3> three_dim_id(64, 1, 2);
  assert(three_dim_id.get(0) == 64 && three_dim_id.get(1) == 1 &&
         three_dim_id.get(2) == 2);

  /* id(const range<dimensions> &range)
   * Construct an id from the dimensions of r. */
  sycl::range<1> one_dim_range(2);
  sycl::id<1> one_dim_id_range(one_dim_range);
  assert(one_dim_id_range.get(0) == 2);
  sycl::range<2> two_dim_range(4, 8);
  sycl::id<2> two_dim_id_range(two_dim_range);
  assert(two_dim_id_range.get(0) == 4 && two_dim_id_range.get(1) == 8);
  sycl::range<3> three_dim_range(16, 32, 64);
  sycl::id<3> three_dim_id_range(three_dim_range);
  assert(three_dim_id_range.get(0) == 16 && three_dim_id_range.get(1) == 32 &&
         three_dim_id_range.get(2) == 64);

  /* id(const item<dimensions> &item)
   * Construct an id from item.get_id().*/
  sycl::item<1, true> one_dim_item_with_offset =
      Builder::createItem<1, true>({4}, {2}, {1});
  sycl::id<1> one_dim_id_item(one_dim_item_with_offset);
  assert(one_dim_id_item.get(0) == 2);
  sycl::item<2, true> two_dim_item_with_offset =
      Builder::createItem<2, true>({8, 16}, {4, 8}, {1, 1});
  sycl::id<2> two_dim_id_item(two_dim_item_with_offset);
  assert(two_dim_id_item.get(0) == 4 && two_dim_id_item.get(1) == 8);
  sycl::item<3, true> three_dim_item_with_offset =
      Builder::createItem<3, true>({32, 64, 128}, {16, 32, 64}, {1, 1, 1});
  sycl::id<3> three_dim_id_item(three_dim_item_with_offset);
  assert(three_dim_id_item.get(0) == 16 && three_dim_id_item.get(1) == 32 &&
         three_dim_id_item.get(2) == 64);
  /* size_t get(int dimension)const
   * Return the value of the id for dimension dimension. */

  /* size_t &operator[](int dimension)const
   * Return a reference to the requested dimension of the id object. */
  sycl::id<1> one_dim_id_brackets(64);
  assert(one_dim_id_brackets[0] == 64);
  sycl::id<2> two_dim_id_brackets(128, 256);
  assert(two_dim_id_brackets[0] == 128 && two_dim_id_brackets[1] == 256);
  sycl::id<3> three_dim_id_brackets(64, 1, 2);
  assert(three_dim_id_brackets[0] == 64 && three_dim_id_brackets[1] == 1 &&
         three_dim_id_brackets[2] == 2);

  /* size_t &operator[](int dimension)const
   * Return a reference to the requested dimension of the id object. */

  /* bool operatorOP(const id<dimensions> &rhs) const
   * Where OP is: ==, !=.
   * Common by-value semantics.
   * T must be equality comparable on the host application and within SYCL
   * kernel functions. Equality between two instances of T (i.e. a == b) must be
   * true if the value of all members are equal and non-equality between two
   * instances of T (i.e. a != b) must be true if the value of any members are
   * not equal, unless either instance has become invalidated by a move
   * operation. Where T is id<dimensions>. */
  {
#define firstOneValue 10
#define secondOneValue 19
#define firstTwoValue 15
#define secondTwoValue 12
#define firstThreeValue 3
#define secondThreeValue 22

    sycl::id<1> one_dim_op_one(firstOneValue);
    sycl::id<1> one_dim_op_two(secondOneValue);
    sycl::id<1> one_dim_op_another_one(firstOneValue);

    sycl::id<2> two_dim_op_one(firstOneValue, firstTwoValue);
    sycl::id<2> two_dim_op_two(secondOneValue, secondTwoValue);
    sycl::id<2> two_dim_op_another_one(firstOneValue, firstTwoValue);

    sycl::id<3> three_dim_op_one(firstOneValue, firstTwoValue, firstThreeValue);
    sycl::id<3> three_dim_op_two(secondOneValue, secondTwoValue,
                                 secondThreeValue);
    sycl::id<3> three_dim_op_another_one(firstOneValue, firstTwoValue,
                                         firstThreeValue);

    // OP : ==
    // id<1> == id<1>
    assert((one_dim_op_one == one_dim_op_two) ==
           (firstOneValue == secondOneValue));
    assert((one_dim_op_one == one_dim_op_another_one) ==
           (firstOneValue == firstOneValue));
    // id<2> == id<2>
    assert((two_dim_op_one == two_dim_op_two) ==
           ((firstOneValue == secondOneValue) &&
            (firstTwoValue == secondTwoValue)));
    assert(
        (two_dim_op_one == two_dim_op_another_one) ==
        ((firstOneValue == firstOneValue) && (firstTwoValue == firstTwoValue)));
    // id<3> == id<3>
    assert((three_dim_op_one == three_dim_op_two) ==
           ((firstOneValue == secondOneValue) &&
            (firstTwoValue == secondTwoValue) &&
            (firstThreeValue == secondThreeValue)));
    assert((three_dim_op_one == three_dim_op_another_one) ==
           ((firstOneValue == firstOneValue) &&
            (firstTwoValue == firstTwoValue) &&
            (firstThreeValue == firstThreeValue)));
    // id<1> == size_t
    assert((one_dim_op_one == secondOneValue) ==
           (firstOneValue == secondOneValue));
    assert((one_dim_op_one == firstOneValue) ==
           (firstOneValue == firstOneValue));

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
    // size_t == id<1>
    assert((firstOneValue == one_dim_op_two) ==
           (firstOneValue == secondOneValue));
    assert((firstOneValue == one_dim_op_another_one) ==
           (firstOneValue == firstOneValue));
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

    // OP : !=
    // id<1> != id<1>
    assert((one_dim_op_one != one_dim_op_two) ==
           (firstOneValue != secondOneValue));
    assert((one_dim_op_one != one_dim_op_another_one) ==
           (firstOneValue != firstOneValue));
    // id<2> != id<2>
    assert(((two_dim_op_one != two_dim_op_two) ==
            (firstOneValue != secondOneValue)) ||
           (firstTwoValue != secondTwoValue));
    assert(((two_dim_op_one != two_dim_op_another_one) ==
            (firstOneValue != firstOneValue)) ||
           (firstTwoValue != firstTwoValue));
    // id<3> != id<3>
    assert((three_dim_op_one != three_dim_op_two) ==
           ((firstOneValue != secondOneValue) ||
            (firstTwoValue != secondTwoValue) ||
            (firstThreeValue != secondThreeValue)));
    assert((three_dim_op_one != three_dim_op_another_one) ==
           ((firstOneValue != firstOneValue) ||
            (firstTwoValue != firstTwoValue) ||
            (firstThreeValue != firstThreeValue)));
    // id<1> != size_t
    assert((one_dim_op_one != secondOneValue) ==
           (firstOneValue != secondOneValue));
    assert((one_dim_op_one != firstOneValue) ==
           (firstOneValue != firstOneValue));

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
    // size_t != id<1>
    assert((firstOneValue != one_dim_op_two) ==
           (firstOneValue != secondOneValue));
    assert((firstOneValue != one_dim_op_another_one) ==
           (firstOneValue != firstOneValue));
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

#undef firstOneValue
#undef secondOneValue
#undef firstTwoValue
#undef secondTwoValue
#undef firstThreeValue
#undef secondThreeValue
  }

  /* id<dimensions> operatorOP(const id<dimensions> &rhs) const
   * Where OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=.
   * Constructs and returns a new instance of the SYCL id class template with
   * the same dimensionality as this SYCL id, where each element of the new SYCL
   * id instance is the result of an element-wise OP operator between each
   * element of this SYCL id and each element of the rhs id. If the operator
   * returns a bool the result is the cast to size_t */
  {
    size_t value_1 = 10;
    size_t value_2 = 15;
    size_t value_3 = 3;

#define oneLeftValue value_1
#define oneRightValue 2
#define twoLeftValue value_2
#define twoRightValue 7
#define threeLeftValue value_3
#define threeRightValue 9

    sycl::id<1> one_dim_op_left(oneLeftValue);
    sycl::id<1> one_dim_op_right(oneRightValue);
    sycl::range<1> one_dim_op_range(oneRightValue);

    sycl::id<2> two_dim_op_left(oneLeftValue, twoLeftValue);
    sycl::id<2> two_dim_op_right(oneRightValue, twoRightValue);
    sycl::range<2> two_dim_op_range(oneRightValue, twoRightValue);

    sycl::id<3> three_dim_op_left(oneLeftValue, twoLeftValue, threeLeftValue);
    sycl::id<3> three_dim_op_right(oneRightValue, twoRightValue,
                                   threeRightValue);
    sycl::range<3> three_dim_op_range(oneRightValue, twoRightValue,
                                      threeRightValue);
#define OPERATOR_TEST(op)                                                      \
  assert((one_dim_op_left op one_dim_op_right)[0] ==                           \
         (oneLeftValue op oneRightValue));                                     \
  assert((one_dim_op_right op one_dim_op_left)[0] ==                           \
         (oneRightValue op oneLeftValue));                                     \
  assert((one_dim_op_left op oneRightValue)[0] ==                              \
         (oneLeftValue op oneRightValue));                                     \
  assert((oneLeftValue op one_dim_op_right)[0] ==                              \
         (oneLeftValue op oneRightValue));                                     \
  assert(((two_dim_op_left op two_dim_op_right)[0] ==                          \
          (oneLeftValue op oneRightValue)) &&                                  \
         ((two_dim_op_right op two_dim_op_left)[1] ==                          \
          (twoRightValue op twoLeftValue)));                                   \
  assert(((two_dim_op_left op oneRightValue)[0] ==                             \
          (oneLeftValue op oneRightValue)) &&                                  \
         ((twoLeftValue op two_dim_op_right)[1] ==                             \
          (twoLeftValue op twoRightValue)));                                   \
  assert(((three_dim_op_left op three_dim_op_right)[0] ==                      \
          (oneLeftValue op oneRightValue)) &&                                  \
         ((three_dim_op_left op three_dim_op_right)[1] ==                      \
          (twoLeftValue op twoRightValue)) &&                                  \
         ((three_dim_op_left op three_dim_op_right)[2] ==                      \
          (threeLeftValue op threeRightValue)));                               \
  assert(((three_dim_op_left op oneRightValue)[0] ==                           \
          (oneLeftValue op oneRightValue)) &&                                  \
         ((twoLeftValue op three_dim_op_right)[1] ==                           \
          (twoLeftValue op twoRightValue)) &&                                  \
         ((three_dim_op_left op threeRightValue)[2] ==                         \
          (threeLeftValue op threeRightValue)));

    OPERATOR_TEST(+)
    OPERATOR_TEST(-)
    OPERATOR_TEST(*)
    OPERATOR_TEST(/)
    OPERATOR_TEST(%)
    OPERATOR_TEST(<<)
    OPERATOR_TEST(>>)
    OPERATOR_TEST(&)
    OPERATOR_TEST(|)
    OPERATOR_TEST(^)
    OPERATOR_TEST(&&)
    OPERATOR_TEST(||)
    OPERATOR_TEST(<)
    OPERATOR_TEST(>)
    OPERATOR_TEST(<=)
    OPERATOR_TEST(>=)

#undef OPERATOR_TEST
#undef OPERATOR_TEST_BASIC

#undef oneLeftValue
#undef oneRightValue
#undef twoLeftValue
#undef twoRightValue
#undef threeLeftValue
#undef threeRightValue
  }

  /* id<dimensions> operatorOP(const id<dimensions> &rhs) const
   * Where OP is: +, -, *, /, %, <<, >>, &, |, ^.
   * Assigns each element of this SYCL id instance with the result of an
   * element-wise OP operator between each element of this SYCL id and each
   * element of the rhs id and returns a reference to this SYCL id. If the
   * operator returns a bool the result is the cast to size_t */
  {
    size_t value_1 = 10;
    size_t value_2 = 15;
    size_t value_3 = 3;

#define oneLeftValue value_1
#define oneRightValue 2
#define twoLeftValue value_2
#define twoRightValue 7
#define threeLeftValue value_3
#define threeRightValue 9

    sycl::id<1> one_dim_op_left(oneLeftValue);
    sycl::id<1> one_dim_op_right(oneRightValue);
    sycl::range<1> one_dim_op_range(oneRightValue);

    sycl::id<2> two_dim_op_left(oneLeftValue, twoLeftValue);
    sycl::id<2> two_dim_op_right(oneRightValue, twoRightValue);
    sycl::range<2> two_dim_op_range(oneRightValue, twoRightValue);

    sycl::id<3> three_dim_op_left(oneLeftValue, twoLeftValue, threeLeftValue);
    sycl::id<3> three_dim_op_right(oneRightValue, twoRightValue,
                                   threeRightValue);
    sycl::range<3> three_dim_op_range(oneRightValue, twoRightValue,
                                      threeRightValue);

#define OPERATOR_TEST(op)                                                      \
  one_dim_op_left[0] = oneLeftValue;                                           \
  one_dim_op_right[0] = oneRightValue;                                         \
  assert((one_dim_op_left op## = one_dim_op_right)[0] ==                       \
         (oneLeftValue op oneRightValue));                                     \
  one_dim_op_left[0] = oneLeftValue;                                           \
  assert((one_dim_op_left op## = oneRightValue)[0] ==                          \
         (oneLeftValue op oneRightValue));                                     \
  two_dim_op_left[0] = oneLeftValue;                                           \
  two_dim_op_left[1] = twoLeftValue;                                           \
  two_dim_op_right[0] = oneRightValue;                                         \
  two_dim_op_right[1] = twoRightValue;                                         \
  assert(((two_dim_op_left op## = two_dim_op_right)[0] ==                      \
          (oneLeftValue op oneRightValue)) &&                                  \
         (two_dim_op_left[1] == (twoLeftValue op twoRightValue)));             \
  two_dim_op_left[0] = oneLeftValue;                                           \
  two_dim_op_left[1] = twoLeftValue;                                           \
  assert(((two_dim_op_left op## = oneRightValue)[0] ==                         \
          (oneLeftValue op oneRightValue)) &&                                  \
         (two_dim_op_left[1] == (twoLeftValue op oneRightValue)));             \
  three_dim_op_left[0] = oneLeftValue;                                         \
  three_dim_op_left[1] = twoLeftValue;                                         \
  three_dim_op_left[2] = threeLeftValue;                                       \
  three_dim_op_right[0] = oneRightValue;                                       \
  three_dim_op_right[1] = twoRightValue;                                       \
  three_dim_op_right[2] = threeRightValue;                                     \
  assert(((three_dim_op_left op## = three_dim_op_right)[0] ==                  \
          (oneLeftValue op oneRightValue)) &&                                  \
         (three_dim_op_left[1] == (twoLeftValue op twoRightValue)) &&          \
         (three_dim_op_left[2] == (threeLeftValue op threeRightValue)));       \
  three_dim_op_left[0] = oneLeftValue;                                         \
  three_dim_op_left[1] = twoLeftValue;                                         \
  three_dim_op_left[2] = threeLeftValue;                                       \
  assert(((three_dim_op_left op## = oneRightValue)[0] ==                       \
          (oneLeftValue op oneRightValue)) &&                                  \
         (three_dim_op_left[1] == (twoLeftValue op oneRightValue)) &&          \
         (three_dim_op_left[2] == (threeLeftValue op oneRightValue)));         \
  one_dim_op_left[0] = oneLeftValue;                                           \
  one_dim_op_range[0] = oneRightValue;                                         \
  assert((one_dim_op_left op## = one_dim_op_range)[0] ==                       \
         (oneLeftValue op oneRightValue));                                     \
  two_dim_op_left[0] = oneLeftValue;                                           \
  two_dim_op_left[1] = twoLeftValue;                                           \
  two_dim_op_range[0] = oneRightValue;                                         \
  two_dim_op_range[1] = twoRightValue;                                         \
  assert(((two_dim_op_left op## = two_dim_op_range)[0] ==                      \
          (oneLeftValue op oneRightValue)) &&                                  \
         (two_dim_op_left[1] == (twoLeftValue op twoRightValue)));             \
  three_dim_op_left[0] = oneLeftValue;                                         \
  three_dim_op_left[1] = twoLeftValue;                                         \
  three_dim_op_left[2] = threeLeftValue;                                       \
  three_dim_op_range[0] = oneRightValue;                                       \
  three_dim_op_range[1] = twoRightValue;                                       \
  three_dim_op_range[2] = threeRightValue;                                     \
  assert(((three_dim_op_left op## = three_dim_op_range)[0] ==                  \
          (oneLeftValue op oneRightValue)) &&                                  \
         (three_dim_op_left[1] == (twoLeftValue op twoRightValue)) &&          \
         (three_dim_op_left[2] == (threeLeftValue op threeRightValue)));

    OPERATOR_TEST(+)
    OPERATOR_TEST(-)
    OPERATOR_TEST(*)
    OPERATOR_TEST(/)
    OPERATOR_TEST(%)
    OPERATOR_TEST(<<)
    OPERATOR_TEST(>>)
    OPERATOR_TEST(&)
    OPERATOR_TEST(|)
    OPERATOR_TEST(^)

#undef OPERATOR_TEST
#undef OPERATOR_TEST_BASIC

#undef oneLeftValue
#undef oneRightValue
#undef twoLeftValue
#undef twoRightValue
#undef threeLeftValue
#undef threeRightValue
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
/* operator size_t() const
 * Test implicit cast from id<1> to size_t and int value
 * Should fails on cast from id<2> and id<3> */
#define numValue 16
  {
    sycl::id<1> one_dim_id_cast_to_num(numValue);
    size_t number_1 = one_dim_id_cast_to_num;
    int number_2 = one_dim_id_cast_to_num;
    size_t number_3 = (size_t)one_dim_id_cast_to_num;
    int number_4 = (int)one_dim_id_cast_to_num;
    size_t number_5 = (int)one_dim_id_cast_to_num;
    assert((number_1 == numValue) && (number_2 == numValue) &&
          (number_3 == numValue) && (number_4 == numValue) &&
          (number_5 == numValue));
  }

#undef numValue
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

  {
    sycl::id<1> one_dim_id(64);
    sycl::id<1> one_dim_id_neg(-64);
    sycl::id<1> one_dim_id_copy(64);
    sycl::id<2> two_dim_id(64, 1);
    sycl::id<2> two_dim_id_neg(-64, -1);
    sycl::id<2> two_dim_id_copy(64, 1);
    sycl::id<3> three_dim_id(64, 1, 2);
    sycl::id<3> three_dim_id_neg(-64, -1, -2);
    sycl::id<3> three_dim_id_copy(64, 1, 2);

    assert((+one_dim_id) == one_dim_id);
    assert(-one_dim_id == one_dim_id_neg);
    assert((+two_dim_id) == two_dim_id);
    assert(-two_dim_id == two_dim_id_neg);
    assert((+three_dim_id) == three_dim_id);
    assert(-three_dim_id == three_dim_id_neg);

    assert((++one_dim_id) == (one_dim_id_copy + 1));
    assert((--one_dim_id) == (one_dim_id_copy));
    assert((++two_dim_id) == (two_dim_id_copy + 1));
    assert((--two_dim_id) == (two_dim_id_copy));
    assert((++three_dim_id) == (three_dim_id_copy + 1));
    assert((--three_dim_id) == (three_dim_id_copy));

    assert((one_dim_id++) == (one_dim_id_copy));
    assert((one_dim_id--) == (one_dim_id_copy + 1));
    assert((two_dim_id++) == (two_dim_id_copy));
    assert((two_dim_id--) == (two_dim_id_copy + 1));
    assert((three_dim_id++) == (three_dim_id_copy));
    assert((three_dim_id--) == (three_dim_id_copy + 1));
  }
}
