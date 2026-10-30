// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only %s -verify

// Operand checks of __builtin_sycl_resolve_kernel_function: it takes exactly
// one argument, which must be a call to a function at namespace scope. Free
// function kernels cannot be member functions or function objects.

#define ND_KERNEL                                                              \
  [[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]   \
  __attribute__((sycl_device))

ND_KERNEL void k(int *p) {}

struct S {
  void member(int *p);
  static void static_member(int *p);
  void operator()(int *p);
};

void test(int *ip, S s) {
  // expected-error@+1 {{builtin takes one argument}}
  (void)__builtin_sycl_resolve_kernel_function();
  // The name-plus-arguments spelling is not accepted.
  // expected-error@+1 {{builtin takes one argument}}
  (void)__builtin_sycl_resolve_kernel_function(k, ip);

  // expected-error@+1 {{must be a call to a SYCL free function kernel}}
  (void)__builtin_sycl_resolve_kernel_function(ip);
  void (*fptr)(int *) = &k;
  // expected-error@+1 {{must be a call to a SYCL free function kernel}}
  (void)__builtin_sycl_resolve_kernel_function(fptr(ip));

  // expected-error@+1 {{must be a call to a SYCL free function kernel}}
  (void)__builtin_sycl_resolve_kernel_function(s.member(ip));
  // expected-error@+1 {{must be a call to a SYCL free function kernel}}
  (void)__builtin_sycl_resolve_kernel_function(S::static_member(ip));
  // expected-error@+1 {{must be a call to a SYCL free function kernel}}
  (void)__builtin_sycl_resolve_kernel_function(s(ip));
  // expected-error@+1 {{must be a call to a SYCL free function kernel}}
  (void)__builtin_sycl_resolve_kernel_function([](int *) {}(ip));
}
