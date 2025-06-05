// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify=expected -fsycl-int-header=%t.h %s

#include "sycl.hpp"

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
foo(int start, ...) { // expected-error {{free function kernel cannot be a variadic function}}
}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo1(int start, ...) { // expected-error {{free function kernel cannot be a variadic function}}
}

// expected-note@+1 {{conflicting attribute is here}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 1)]] void
foo2(int start);

// expected-error@+1 {{attribute 'add_ir_attributes_function' is already applied with different arguments}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo2(int start) {
}

// expected-note@+1 {{previous declaration is here}}
void foo3(int start, int *ptr);

// expected-error@+2 {{the first occurrence of SYCL kernel free function should be declared with 'sycl-nd-range-kernel' or 'sycl-single-task-kernel' compile time properties}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo3(int start, int *ptr){}

// expected-note@+1 {{previous declaration is here}}
void foo4(float start, float *ptr);

// expected-error@+2 {{the first occurrence of SYCL kernel free function should be declared with 'sycl-nd-range-kernel' or 'sycl-single-task-kernel' compile time properties}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo4(float start, float *ptr);

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo4(float start, float *ptr);

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
foo4(float start, float *ptr){}


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

class TestClass {
public:

// expected-error@+2 {{kernel function 'ndRangeKernelMethod' must be a free function or static member function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{kernel function 'singleTaskKernelMethod' must be a free function or static member function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
singleTaskKernelMethod(int Value) {
}

// expected-error@+2 {{static class method can not be used as SYCL kernel free function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
static void StaticndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{static class method can not be used as SYCL kernel free function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
static void StaticsingleTaskKernelMethod(int Value) {
}

};

struct TestStruct {

// expected-error@+2 {{kernel function 'ndRangeKernelMethod' must be a free function or static member function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void 
ndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{kernel function 'singleTaskKernelMethod' must be a free function or static member function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] void 
singleTaskKernelMethod(int Value) {
}

// expected-error@+2 {{static class method can not be used as SYCL kernel free function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
static void StaticndRangeKernelMethod(int Value) {
}

// expected-error@+2 {{static class method can not be used as SYCL kernel free function}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 2)]] 
static void StaticsingleTaskKernelMethod(int Value) {
}

};
