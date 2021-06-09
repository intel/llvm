// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

//==--------------- marray.cpp - SYCL marray test --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  // Constructing vector from a scalar
  sycl::marray<int, 1> marray_from_one_elem(1);

  // Check broadcasting operator=
  sycl::marray<float, 4> b_marray(1.0);
  b_marray = 0.5;
  assert(static_cast<float>(b_marray[0]) == static_cast<float>(0.5));
  assert(static_cast<float>(b_marray[1]) == static_cast<float>(0.5));
  assert(static_cast<float>(b_marray[2]) == static_cast<float>(0.5));
  assert(static_cast<float>(b_marray[3]) == static_cast<float>(0.5));

  // Check that [u]long[n] type aliases match marray<[unsigned] long, n> types.
  assert((std::is_same<sycl::marray<long, 2>, sycl::mlong2>::value));
  assert((std::is_same<sycl::marray<long, 3>, sycl::mlong3>::value));
  assert((std::is_same<sycl::marray<long, 4>, sycl::mlong4>::value));
  assert((std::is_same<sycl::marray<long, 8>, sycl::mlong8>::value));
  assert((std::is_same<sycl::marray<long, 16>, sycl::mlong16>::value));
  assert((std::is_same<sycl::marray<unsigned long, 2>, sycl::mulong2>::value));
  assert((std::is_same<sycl::marray<unsigned long, 3>, sycl::mulong3>::value));
  assert((std::is_same<sycl::marray<unsigned long, 4>, sycl::mulong4>::value));
  assert((std::is_same<sycl::marray<unsigned long, 8>, sycl::mulong8>::value));
  assert(
      (std::is_same<sycl::marray<unsigned long, 16>, sycl::mulong16>::value));

  mint3 t000;
  mint3 t222{2};
  mint3 t123{1, 2, 3};
  mint3 tcpy{t123};
  mint3 t___;
  sycl::marray<bool, 3> b___;

  // test default ctor
  assert(t000[0] == 0 && t000[1] == 0 && t000[2] == 0);

  // test constant ctor
  assert(t222[0] == 2 && t222[1] == 2 && t222[2] == 2);

  // test vararg ctor
  assert(t123[0] == 1 && t123[1] == 2 && t123[2] == 3);

  // test copy ctor
  assert(tcpy[0] == 1 && tcpy[1] == 2 && tcpy[2] == 3);

  // test iterators
  for (auto &a : t___) {
    a = 9;
  }
  assert(t___[0] == 9 && t___[1] == 9 && t___[2] == 9);

  // test relation operator forms
  t___ = t123 + t222;
  assert(t___[0] == 3 && t___[1] == 4 && t___[2] == 5);
  t___ = t123 - 1;
  assert(t___[0] == 0 && t___[1] == 1 && t___[2] == 2);
  t___ += t123;
  assert(t___[0] == 1 && t___[1] == 3 && t___[2] == 5);
  t___ -= 1;
  assert(t___[0] == 0 && t___[1] == 2 && t___[2] == 4);

  // test unary operator forms
  t___++;
  assert(t___[0] == 1 && t___[1] == 3 && t___[2] == 5);
  --t___;
  assert(t___[0] == 0 && t___[1] == 2 && t___[2] == 4);

  // test relation operator forms
  b___ = t123 > t222;
  assert(b___[0] == false && b___[1] == false && b___[2] == true);
  b___ = t123 < 2;
  assert(b___[0] == true && b___[1] == false && b___[2] == false);

  // test const operator forms
  t___ = -mint3{1, 2, 3};
  assert(t___[0] == -1 && t___[1] == -2 && t___[2] == -3);
  t___ = +mint3{1, 2, 3};
  assert(t___[0] == +1 && t___[1] == +2 && t___[2] == +3);
  t___ = ~mint3{1, 2, 3};
  assert(t___[0] == ~1 && t___[1] == ~2 && t___[2] == ~3);
  b___ = !mint3{0, 1, 2};
  assert(b___[0] == true && b___[1] == false && b___[2] == false);

  return 0;
}
