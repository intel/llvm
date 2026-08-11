// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs
// Build a shared base header that defines the polymorphic class B in its own
// module, so both FirstModule and SecondModule import (and merge) the *same* B.
// RUN: echo "#ifndef BASE_H"                    >> %t/Inputs/base.h
// RUN: echo "#define BASE_H"                    >> %t/Inputs/base.h
// RUN: echo "struct B { virtual int g(int); };" >> %t/Inputs/base.h
// RUN: echo "#endif"                            >> %t/Inputs/base.h
// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h
// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h
// Build module map file
// RUN: echo "module BaseModule {"       >> %t/Inputs/module.modulemap
// RUN: echo "    header \"base.h\""      >> %t/Inputs/module.modulemap
// RUN: echo "}"                          >> %t/Inputs/module.modulemap
// RUN: echo "module FirstModule {"      >> %t/Inputs/module.modulemap
// RUN: echo "    header \"first.h\""     >> %t/Inputs/module.modulemap
// RUN: echo "}"                          >> %t/Inputs/module.modulemap
// RUN: echo "module SecondModule {"     >> %t/Inputs/module.modulemap
// RUN: echo "    header \"second.h\""    >> %t/Inputs/module.modulemap
// RUN: echo "}"                          >> %t/Inputs/module.modulemap
// Run test
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -std=c++2c -Wno-declcall-extension \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I%t/Inputs -verify %s

// The two definitions of S differ only in the declcall devirtualization flag:
// FirstModule uses a qualified virtual call (B::g, which devirtualizes),
// SecondModule uses an unqualified virtual call (g, which does not). Both
// declcall operands rebuild the same &B::g operand AST, so only the
// Devirtualize bit distinguishes them. That bit must participate in the ODR
// hash so the mismatch is diagnosed rather than silently merged.

#include "base.h"
#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST)
struct S {
  int (B::*p)(int) = declcall(((B *)0)->B::g(0)); // devirtualized
};
#elif defined(SECOND)
struct S {
  int (B::*p)(int) = declcall(((B *)0)->g(0)); // virtual
};
#else
S s;
// expected-error@first.h:* {{'S' has different definitions in different modules; first difference is definition in module 'FirstModule' found field 'p' with an initializer}}
// expected-note@second.h:* {{but in 'SecondModule' found field 'p' with a different initializer}}
#endif

// Keep the FIRST/SECOND macros contained to their own module so they do not
// leak into the main translation unit (which would re-enter a branch above and
// spuriously redefine 'S').
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
