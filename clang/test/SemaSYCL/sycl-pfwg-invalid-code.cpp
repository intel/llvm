// RUN: %clang_cc1 -fsycl-is-device %s -verify

// Tests that an error diagnostic is issued without a crash (due to a
// triggered assertion) when some invalid cases are encountered in 
// the processing of kernel_parallel_for_work_group.

template <typename, typename, typename Kernel>
__attribute__((sycl_kernel))
void kernel_parallel_for_work_group(const Kernel&) {
  unknown(); // expected-error{{use of undeclared identifier 'unknown'}}
}
void foo() {
  auto lambda = [] {};
  kernel_parallel_for_work_group<int, int>(lambda);
}
