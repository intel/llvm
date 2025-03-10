// RUN: %clang_cc1 -fsycl-is-device -fsycl-allow-func-ptr=defined -internal-isystem %S/Inputs -fsyntax-only -verify -sycl-std=2020 -std=c++17 %s

#include "sycl.hpp"

template <typename T> SYCL_EXTERNAL void foo(T input);

template <>
void foo(int input) {}
template <>
void foo(double input);

SYCL_EXTERNAL void usage() {
  auto FP = &foo<int>;
  auto FP1 = &foo<char>; // expected-error {{taking address of a function without a definition and not marked with 'intel::device_indirectly_callable'}}
  auto FP2 = &foo<double>;
}

template <>
void foo(double input) {}
