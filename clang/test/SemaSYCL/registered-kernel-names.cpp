// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// The test checks issuing diagnostics for registered kernel names.

// expected-error@+1 {{attempting to register a function that is not a SYCL free function as 'kernelnamefoo'}}
void foo();

constexpr const char *str = "foo";

// expected-error@+1 {{'__registered_kernels__' attribute must have at least one argument}}
[[__sycl_detail__::__registered_kernels__(
)]];

// expected-error@+2 {{argument to the '__registered_kernels__' attribute must be an initializer list expression}}
[[__sycl_detail__::__registered_kernels__(
  1
)]];

// expected-error@+2 {{each initializer list argument to the '__registered_kernels__' attribute must contain a pair of values}}
[[__sycl_detail__::__registered_kernels__(
  {}
)]];

// expected-error@+2 {{each initializer list argument to the '__registered_kernels__' attribute must contain a pair of values}}
[[__sycl_detail__::__registered_kernels__(
  { "foo" }
)]];

// expected-error@+2 {{unable to resolve free function kernel 'foo'}}
[[__sycl_detail__::__registered_kernels__(
  { "foo", "foo" }
)]];

// expected-error@+2 {{'__registered_kernels__' attribute requires a string}}
[[__sycl_detail__::__registered_kernels__(
  { str, 1 }
)]];

// expected-error@+2 {{each initializer list argument to the '__registered_kernels__' attribute must contain a pair of values}}
[[__sycl_detail__::__registered_kernels__(
  { "foo", 1, foo }
)]];

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void func1();

namespace N {
[[__sycl_detail__::__registered_kernels__(
  {"kernelnamefoo", foo}
)]];
}

// expected-error@+3 {{unable to resolve free function kernel 'func'}}
namespace {
[[__sycl_detail__::__registered_kernels__(
  {"func", func1<int, int>}
)]];
}

// expected-error@+2 {{free function kernel has already been registered with 'reg1'; cannot register with 'reg2'}}
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void good1() {
}

[[__sycl_detail__::__registered_kernels__(
  {"reg1", good1},
  {"reg2", good1}
)]];


struct S1 {
// expected-error@+1 {{'int &' cannot be used as the type of a kernel parameter}}
  int &ri;
};

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void func_with_S1_param(T s) {
}

[[__sycl_detail__::__registered_kernels__(
  {"ref field", func_with_S1_param<S1>}
)]];
