// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s

// The test checks issuing diagnostics for registered kernel names.

// expected-error@+1 {{attempting to register a function that is not a SYCL free function as 'kernelnamefoo'}}
void foo();

constexpr const char *str = "foo";

// expected-error@+1 {{'__registered_kernels__' atribute must have at least one argument}}
[[__sycl_detail__::__registered_kernels__(
)]];

// expected-error@+2 {{argument to the '__registered_kernels__' atribute must be an initializer list expression}}
[[__sycl_detail__::__registered_kernels__(
  1
)]];

// expected-error@+2 {{each initializer list argument to the '__registered_kernels__' atribute must contain a pair of values}}
[[__sycl_detail__::__registered_kernels__(
  {}
)]];

// expected-error@+2 {{each initializer list argument to the '__registered_kernels__' atribute must contain a pair of values}}
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

// expected-error@+2 {{each initializer list argument to the '__registered_kernels__' atribute must contain a pair of values}}
[[__sycl_detail__::__registered_kernels__(
  { "foo", 1, foo }
)]];

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void func();

namespace N {
[[__sycl_detail__::__registered_kernels__(
  {"kernelnamefoo", foo}
)]];
}

// expected-error@+3 {{unable to resolve free function kernel 'func'}}
namespace {
[[__sycl_detail__::__registered_kernels__(
  {"func", func<int, int>}
)]];
}

// expected-error@+3 {{free function kernel has already been registered with 'reg1'; cannot register with 'reg2'}}
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void good() {
}

[[__sycl_detail__::__registered_kernels__(
  {"reg1", good},
  {"reg2", good}
)]];
