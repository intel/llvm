// RUN: %clang_cc1 -verify -fsyntax-only -std=c++11 %s
//
// expected-no-diagnostics 
int foo(bool __pipe);
int foo(bool __read_only);
int foo(bool __write_only);
