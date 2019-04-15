//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_const

#include <type_traits>

#include "test_macros.h"

template <class T, class U>
void test_remove_const_imp()
{
    static_assert((std::is_same<typename std::remove_const<T>::type, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::remove_const_t<T>, U>::value), "");
#endif
}

template <class T>
void test_remove_const()
{
    test_remove_const_imp<T, T>();
    test_remove_const_imp<const T, T>();
    test_remove_const_imp<volatile T, volatile T>();
    test_remove_const_imp<const volatile T, volatile T>();
}

int main(int, char**)
{
    test_remove_const<void>();
    test_remove_const<int>();
    test_remove_const<int[3]>();
    test_remove_const<int&>();
    test_remove_const<const int&>();
    test_remove_const<int*>();
    test_remove_const<const int*>();

  return 0;
}
