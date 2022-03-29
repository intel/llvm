// RUN: %clang_cc1 -fsycl-is-device %s -verify

// Tests that an error diagnostic is issued instead of a crash (via an
// assertion) when some invalid cases are encountered in the processing
// of kernel_parallel_for_work_group.

template <typename, typename, typename Kernel>
// expected-error@+2{{unable to find lambda or function object in the kernel parameter; perhaps it was invoked with the wrong signature?}}
__attribute__((sycl_kernel)) void kernel_parallel_for_work_group(const Kernel&) {
  unknown(); // expected-error{{use of undeclared identifier 'unknown'}}
}
void foo() {
  auto lambda = [] {};
  kernel_parallel_for_work_group<int, int>(lambda);
}
