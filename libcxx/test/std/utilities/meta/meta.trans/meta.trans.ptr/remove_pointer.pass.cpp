//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_pointer

#include <type_traits>
#include "test_macros.h"

template <class T, class U>
void test_remove_pointer()
{
    static_assert((std::is_same<typename std::remove_pointer<T>::type, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::remove_pointer_t<T>,     U>::value), "");
#endif
}

int main(int, char**)
{
    test_remove_pointer<void, void>();
    test_remove_pointer<int, int>();
    test_remove_pointer<int[3], int[3]>();
    test_remove_pointer<int*, int>();
    test_remove_pointer<const int*, const int>();
    test_remove_pointer<int**, int*>();
    test_remove_pointer<int** const, int*>();
    test_remove_pointer<int*const * , int* const>();
    test_remove_pointer<const int** , const int*>();

    test_remove_pointer<int&, int&>();
    test_remove_pointer<const int&, const int&>();
    test_remove_pointer<int(&)[3], int(&)[3]>();
    test_remove_pointer<int(*)[3], int[3]>();
    test_remove_pointer<int*&, int*&>();
    test_remove_pointer<const int*&, const int*&>();

  return 0;
}
