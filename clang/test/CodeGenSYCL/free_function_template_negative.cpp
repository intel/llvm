// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=expected -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s

#include "sycl.hpp"

// expected-error@+1 {{free function can not be a variadic template function}}
template <typename ...Ts>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated_variadic(Ts... args) {
}

template void templated_variadic<int, float>(int, float);
