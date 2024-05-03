// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

//==--------------- marray.cpp - SYCL marray test --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <algorithm>

using namespace sycl;

#define CHECK_ALIAS_BY_SIZE(ALIAS_MTYPE, ELEM_TYPE, MARRAY_SIZE)               \
  static_assert(std::is_same_v<sycl::marray<ELEM_TYPE, MARRAY_SIZE>,           \
                               sycl::ALIAS_MTYPE##MARRAY_SIZE>);

template <size_t N> bool AllTrue(sycl::marray<bool, N> M) {
  return std::all_of(M.begin(), M.end(), [](const bool &V) { return V; });
}

#define CHECK_ALIAS(ALIAS_MTYPE, ELEM_TYPE)                                    \
  CHECK_ALIAS_BY_SIZE(ALIAS_MTYPE, ELEM_TYPE, 2)                               \
  CHECK_ALIAS_BY_SIZE(ALIAS_MTYPE, ELEM_TYPE, 3)                               \
  CHECK_ALIAS_BY_SIZE(ALIAS_MTYPE, ELEM_TYPE, 4)                               \
  CHECK_ALIAS_BY_SIZE(ALIAS_MTYPE, ELEM_TYPE, 8)                               \
  CHECK_ALIAS_BY_SIZE(ALIAS_MTYPE, ELEM_TYPE, 16)

// Check different combinations of the given binary operation. Some compare
// scalar values with the marrays, which is valid as all elements in the marrays
// should be the same.
#define CHECK_BINOP(OP, LHS, RHS)                                              \
  assert(AllTrue((LHS[0] OP RHS) == (LHS OP RHS)) &&                           \
         AllTrue((LHS OP RHS[0]) == (LHS OP RHS)) &&                           \
         AllTrue((LHS[0] OP RHS[0]) == (LHS OP RHS)));

struct NotDefaultConstructible {
  NotDefaultConstructible() = delete;
  constexpr NotDefaultConstructible(int){};
};

template <typename T> void CheckBinOps() {
  sycl::marray<T, 3> ref_arr0{T(0)};
  sycl::marray<T, 3> ref_arr1{T(1)};
  sycl::marray<T, 3> ref_arr2{T(2)};
  sycl::marray<T, 3> ref_arr3{T(3)};

  CHECK_BINOP(+, ref_arr1, ref_arr2)
  CHECK_BINOP(-, ref_arr1, ref_arr2)
  CHECK_BINOP(*, ref_arr1, ref_arr2)
  CHECK_BINOP(/, ref_arr1, ref_arr2)
  CHECK_BINOP(&&, ref_arr0, ref_arr2)
  CHECK_BINOP(||, ref_arr0, ref_arr2)
  CHECK_BINOP(==, ref_arr1, ref_arr2)
  CHECK_BINOP(!=, ref_arr1, ref_arr2)
  CHECK_BINOP(<, ref_arr1, ref_arr2)
  CHECK_BINOP(>, ref_arr1, ref_arr2)
  CHECK_BINOP(<=, ref_arr1, ref_arr2)
  CHECK_BINOP(>=, ref_arr1, ref_arr2)

  if constexpr (!std::is_same_v<T, sycl::half> && !std::is_same_v<T, float> &&
                !std::is_same_v<T, double>) {
    // Operators not supported on sycl::half, float, and double.
    CHECK_BINOP(%, ref_arr1, ref_arr2)
    CHECK_BINOP(&, ref_arr1, ref_arr3)
    CHECK_BINOP(|, ref_arr1, ref_arr3)
    CHECK_BINOP(^, ref_arr1, ref_arr3)
    CHECK_BINOP(>>, ref_arr1, ref_arr2)
    CHECK_BINOP(<<, ref_arr1, ref_arr2)
  }
}

template <typename DataT> void CheckConstexprVariadicCtors() {
  constexpr DataT default_val{1};

  constexpr sycl::marray<DataT, 5> marray_with_5_elements(
      default_val, default_val, default_val, default_val, default_val);
  constexpr sycl::marray<DataT, 3> marray_with_3_elements(
      default_val, default_val, default_val);

  constexpr sycl::marray<DataT, 6> m1(marray_with_5_elements, default_val);
  constexpr sycl::marray<DataT, 6> m2(default_val, marray_with_5_elements);
  constexpr sycl::marray<DataT, 7> m3(default_val, marray_with_5_elements,
                                      default_val);
  constexpr sycl::marray<DataT, 8> m4(marray_with_5_elements,
                                      marray_with_3_elements);
  constexpr sycl::marray<DataT, 9> m5(default_val, marray_with_5_elements,
                                      marray_with_3_elements);
  constexpr sycl::marray<DataT, 9> m6(marray_with_5_elements, default_val,
                                      marray_with_3_elements);
  constexpr sycl::marray<DataT, 9> m7(marray_with_5_elements,
                                      marray_with_3_elements, default_val);
  constexpr sycl::marray<DataT, 10> m8(default_val, marray_with_5_elements,
                                       default_val, marray_with_3_elements);
  constexpr sycl::marray<DataT, 10> m9(default_val, marray_with_5_elements,
                                       marray_with_3_elements, default_val);
  constexpr sycl::marray<DataT, 10> m10(marray_with_5_elements, default_val,
                                        marray_with_3_elements, default_val);
  constexpr sycl::marray<DataT, 11> m11(default_val, marray_with_5_elements,
                                        default_val, marray_with_3_elements,
                                        default_val);
}

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

  // Check alias types.
  CHECK_ALIAS(mbool, bool)
  CHECK_ALIAS(mchar, std::int8_t)
  CHECK_ALIAS(mschar, std::int8_t)
  CHECK_ALIAS(muchar, std::uint8_t)
  CHECK_ALIAS(mshort, std::int16_t)
  CHECK_ALIAS(mushort, std::uint16_t)
  CHECK_ALIAS(mint, std::int32_t)
  CHECK_ALIAS(muint, std::uint32_t)
  CHECK_ALIAS(mlong, std::int64_t)
  CHECK_ALIAS(mulong, std::uint64_t)
  CHECK_ALIAS(mlonglong, std::int64_t)
  CHECK_ALIAS(mulonglong, std::uint64_t)
  CHECK_ALIAS(mhalf, sycl::half)
  CHECK_ALIAS(mfloat, float)
  CHECK_ALIAS(mdouble, double)

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

  // Check direct binary operators
  CheckBinOps<bool>();
  CheckBinOps<std::int8_t>();
  CheckBinOps<std::uint8_t>();
  CheckBinOps<std::int16_t>();
  CheckBinOps<std::uint16_t>();
  CheckBinOps<std::int32_t>();
  CheckBinOps<std::uint32_t>();
  CheckBinOps<std::int64_t>();
  CheckBinOps<std::uint64_t>();
  CheckBinOps<sycl::half>();
  CheckBinOps<float>();
  CheckBinOps<double>();

  // check copyability
  constexpr sycl::marray<double, 5> ma;
  constexpr sycl::marray<double, 5> mb(ma);
  constexpr sycl::marray<double, 5> mc = ma;

  // check variadic ctor
  CheckConstexprVariadicCtors<bool>();
  CheckConstexprVariadicCtors<std::int8_t>();
  CheckConstexprVariadicCtors<std::uint8_t>();
  CheckConstexprVariadicCtors<std::int16_t>();
  CheckConstexprVariadicCtors<std::uint16_t>();
  CheckConstexprVariadicCtors<std::int32_t>();
  CheckConstexprVariadicCtors<std::uint32_t>();
  CheckConstexprVariadicCtors<std::int64_t>();
  CheckConstexprVariadicCtors<std::uint64_t>();
  CheckConstexprVariadicCtors<sycl::half>();
  CheckConstexprVariadicCtors<float>();
  CheckConstexprVariadicCtors<double>();
  CheckConstexprVariadicCtors<NotDefaultConstructible>();

  // check trivially copyability
  struct Copyable {
    int a;
    double b;
    const char *name;
  };

  static_assert(std::is_trivially_copyable<sycl::marray<Copyable, 5>>::value,
                "sycl::marray<Copyable, 5> is not trivially copyable type");
  static_assert(
      !std::is_trivially_copyable<sycl::marray<std::string, 5>>::value,
      "sycl::marray<std::string, 5> is trivially copyable type");

  // check device copyability
  static_assert(sycl::is_device_copyable<sycl::marray<std::tuple<>, 5>>::value,
                "sycl::marray<std::tuple<>, 5> is not device copyable type");
  static_assert(!sycl::is_device_copyable<sycl::marray<std::string, 5>>::value,
                "sycl::marray<std::string, 5> is device copyable type");

  return 0;
}
