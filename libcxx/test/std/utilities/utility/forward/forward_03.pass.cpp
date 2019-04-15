//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test forward

#include <utility>
#include <cassert>

#include "test_macros.h"

struct A
{
};

A source() {return A();}
const A csource() {return A();}

typedef char one;
struct two {one _[2];};
struct four {one _[4];};
struct eight {one _[8];};

one test(A&);
two test(const A&);

int main(int, char**)
{
    A a;
    const A ca = A();

    ((void)a); // Prevent unused warning
    ((void)ca); // Prevent unused warning

#if TEST_STD_VER < 11
    static_assert(sizeof(test(std::forward<A&>(a))) == 1, "");
    static_assert(sizeof(test(std::forward<A>(a))) == 1, "");

    // Libc++'s C++03 implementation of 'forward' cannot accept true non-const
    // rvalues.
    // static_assert(sizeof(test(std::forward<A>(source()))) == 2, "");

    static_assert(sizeof(test(std::forward<const A&>(a))) == 2, "");
    static_assert(sizeof(test(std::forward<const A&>(source()))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(a))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(source()))) == 2, "");

    static_assert(sizeof(test(std::forward<const A&>(ca))) == 2, "");
    static_assert(sizeof(test(std::forward<const A&>(csource()))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(ca))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(csource()))) == 2, "");
#endif

  return 0;
}
