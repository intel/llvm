// RUN: %clangxx -fsycl -fsyntax-only %s
//==---- integer_n_bit.cpp - SYCL integerNbit type traits test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/boost/mp11/list.hpp>
#include <sycl/sycl.hpp>

namespace s = sycl;
namespace d = sycl::detail;

template <bool... V> using bool_list = d::value_list<bool, V...>;

template <template <typename> class T, typename TL, typename BL> struct check {
  void operator()() {
    static_assert(
        std::is_same_v<d::boost::mp11::mp_rename_v<
                           d::boost::mp11::mp_transform<T, TL>, bool_list>,
                       BL>,
        "");
  }
};

using TypeList =
    d::type_list<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
                 uint64_t, s::vec<int8_t, 2>, s::vec<int16_t, 2>,
                 s::vec<int32_t, 2>, s::vec<int64_t, 2>, s::vec<uint8_t, 2>,
                 s::vec<uint16_t, 2>, s::vec<uint32_t, 2>, s::vec<uint64_t, 2>,
                 bool, float, double, s::half, s::vec<float, 2>,
                 s::vec<double, 2>, s::vec<s::half, 2>>;

int main() {
  check<d::is_igeninteger8bit, TypeList,
        bool_list<1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_igeninteger16bit, TypeList,
        bool_list<0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_igeninteger32bit, TypeList,
        bool_list<0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_igeninteger64bit, TypeList,
        bool_list<0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();

  check<d::is_ugeninteger8bit, TypeList,
        bool_list<0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_ugeninteger16bit, TypeList,
        bool_list<0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_ugeninteger32bit, TypeList,
        bool_list<0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_ugeninteger64bit, TypeList,
        bool_list<0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0>>()();

  check<d::is_geninteger8bit, TypeList,
        bool_list<1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_geninteger16bit, TypeList,
        bool_list<0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_geninteger32bit, TypeList,
        bool_list<0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0>>()();
  check<d::is_geninteger64bit, TypeList,
        bool_list<0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0>>()();

  return 0;
}
