// RUN: %clang_cc1 -x c++ -fsycl-is-device -std=c++11 -fsyntax-only -verify -pedantic %s

// no error expected
using type1 = __attribute__((pipe("read_only"))) const int;

// no error expected
using type2 = __attribute__((pipe("write_only"))) const int;

// expected-error@+1 {{'42' mode for pipe attribute is not supported. Allowed modes: 'read_only', 'write_only'}}
using type3 = __attribute__((pipe("42"))) const int;

// expected-error@+1{{'pipe' attribute requires a string}}
using type4 = __attribute__((pipe(0))) const int;

// expected-error@+1{{'pipe' attribute takes one argument}}
using type5 = __attribute__((pipe)) const int;
