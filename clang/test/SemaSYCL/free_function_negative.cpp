// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=expected -fsycl-int-header=%t.h %s

#include "sycl.hpp"

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
foo(int start, ...) { // expected-error {{free function kernel cannot be a variadic function}}
}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo1(int start, ...) { // expected-error {{free function kernel cannot be a variadic function}}
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
singleTaskKernelDefaultValues(int Value = 1) {
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelDefaultValues(int Value = 1) {
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void  
singleTaskKernelDefaultValues(int Ivalue = 1, unsigned int Uvalue = 3) {
}

// expected-error@+2 {{a function with a default argument value cannot be used to define SYCL free function kernel}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelDefaultValues(int Ivalue = 1, unsigned int Uvalue = 3) {
}

// expected-error@+2 {{SYCL free function kernel should have return type 'void'}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] int 
singleTaskKernelReturnType(int Value) {
}

// expected-error@+2 {{SYCL free function kernel should have return type 'void'}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] int 
ndRangeKernelReturnType(int Value) {
}
