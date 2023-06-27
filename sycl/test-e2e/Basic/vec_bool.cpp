// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==-------------- vec_bool.cpp - SYCL vec<> for bool test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WIboolH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

constexpr int size = 2;

void init_arr(bool *arr, bool val) {
  for (int i = 0; i < size; ++i)
    arr[i] = val;
}

void init_vec(sycl::vec<bool, size> &vec, bool val) {
  for (int i = 0; i < size; ++i)
    vec[i] = val;
}

void check_result(const sycl::vec<bool, size> &res, bool *expected) {
  for (int i = 0; i < size; ++i) {
    assert(expected[i] == res[i] && "Incorrect result");
  }
}

int main() {
  sycl::queue q;

  bool false_val = false;
  bool true_val = true;
  sycl::vec<bool, size> vec_false;
  sycl::vec<bool, size> vec_true;
  init_vec(vec_false, false_val);
  init_vec(vec_true, true_val);

  bool expected[size];
  sycl::vec<bool, size> resVec;

  // Test negate (operator ~)
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = ~(accVecTrue[0]); });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test left shift (operator <<) 1
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task(
           [=]() { accResVec[0] = accVecTrue[0] << accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test left shift (operator <<) 2
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = accVecTrue[0] << true_val; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test left shift (operator <<) 3
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = true_val << accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test left shift (operator <<) 4
  {
    init_arr(expected, true);
    init_vec(resVec, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] <<= accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test add (operator +) 1
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = accVecTrue[0] + accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test add (operator +) 2
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = accVecTrue[0] + true_val; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test add (operator +) 3
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = true_val + accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test add (operator +) 4
  {
    init_arr(expected, true);
    init_vec(resVec, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] += accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test sub (operator -) 1
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecFalse(&vec_false, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecFalse(bufVecFalse, cgh, sycl::read_only);
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task(
           [=]() { accResVec[0] = accVecFalse[0] + accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test sub (operator -) 2
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecFalse(&vec_false, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecFalse(bufVecFalse, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = accVecFalse[0] + true_val; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test sub (operator -) 3
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecFalse(&vec_false, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] = false_val - accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test sub (operator -) 4
  {
    init_arr(expected, true);
    init_vec(resVec, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecFalse(&vec_false, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() { accResVec[0] -= accVecTrue[0]; });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  auto vec_two = sycl::vec<bool, size>(static_cast<bool>(2));
  // Test swizzle 1 (operator ^=)
  {
    init_arr(expected, false);
    init_vec(resVec, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] ^=
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 2 (operator >>=)
  {
    init_arr(expected, false);
    init_vec(resVec, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] >>=
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 3 (operator +)
  {
    init_arr(expected, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             accVecTrue[0] +
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 4 (operator -)
  {
    init_arr(expected, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             accVecTrue[0] -
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 5 (operator -)
  {
    init_arr(expected, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             true_val -
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 6 (operator +=)
  {
    init_arr(expected, true);
    init_vec(resVec, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] +=
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 7 (operator -=)
  {
    init_arr(expected, false);
    init_vec(resVec, true);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] -=
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 8 (operator >>)
  {
    init_arr(expected, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             accVecTrue[0] >>
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 9 (operator >>)
  {
    init_arr(expected, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             true_val >>
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 10 (operator ^)
  {
    init_arr(expected, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecTrue(&vec_true, 1);
    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTrue(bufVecTrue, cgh, sycl::read_only);
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             accVecTrue[0] ^
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);

  // Test swizzle 11 (operator ^)
  {
    init_arr(expected, false);

    sycl::buffer<sycl::vec<bool, size>> bufVecTwo(&vec_two, 1);
    sycl::buffer<sycl::vec<bool, size>> bufResVec(&resVec, 1);

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor accVecTwo(bufVecTwo, cgh, sycl::read_only);
       sycl::accessor accResVec(bufResVec, cgh, sycl::write_only);
       cgh.single_task([=]() {
         accResVec[0] =
             true_val ^
             accVecTwo[0].template swizzle<sycl::elem::s0, sycl::elem::s1>();
       });
     }).wait_and_throw();
  }
  check_result(resVec, expected);
}
