// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=expected -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s

#include "sycl.hpp"

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
foo(int start, ...) { // expected-error {{free function kernel cannot be a variadic template function}}
}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo1(int start, ...) { // expected-error {{free function kernel cannot be a variadic template function}}
}
