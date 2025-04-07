// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s

#include "sycl.hpp"

// expected-error@+1 {{free function can not be variadic template function}}
template<typename ...Ts>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated_variadic(Ts... args) {
}

template void templated_variadic(int, int);
