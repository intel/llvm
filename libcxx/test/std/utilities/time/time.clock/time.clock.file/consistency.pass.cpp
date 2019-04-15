//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
//
// TODO: Remove this when filesystem gets integrated into the dylib
// REQUIRES: c++filesystem

// <chrono>

// file_clock

// check clock invariants

#include <chrono>

template <class T>
void test(const T &) {}

int main(int, char**)
{
    typedef std::chrono::file_clock C;
    static_assert((std::is_same<C::rep, C::duration::rep>::value), "");
    static_assert((std::is_same<C::period, C::duration::period>::value), "");
    static_assert((std::is_same<C::duration, C::time_point::duration>::value), "");
    static_assert((std::is_same<C::time_point::clock, C>::value), "");
    static_assert(!C::is_steady, "");
    test(std::chrono::file_clock::is_steady);

  return 0;
}
