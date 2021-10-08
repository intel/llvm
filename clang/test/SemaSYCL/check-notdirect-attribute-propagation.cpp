// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -sycl-std=2017 -Wno-sycl-2017-compat -triple spir64 -verify
// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -sycl-std=2017 -Wno-sycl-2017-compat -triple spir64 -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl-is-device -sycl-std=2017 -triple spir64 | FileCheck %s

#ifndef TRIGGER_ERROR
[[intel::no_global_work_offset]] void not_direct_one() {} // expected-no-diagnostics

[[intel::reqd_sub_group_size(1)]] void func_one() {
  not_direct_one();
}

#else
[[sycl::reqd_work_group_size(2, 2, 2)]] void not_direct_two() {} // expected-note 2 {{conflicting attribute is here}}

[[intel::max_work_group_size(1, 1, 1)]] // expected-note 3 {{conflicting attribute is here}}
void
func_two() {
  not_direct_two();
}

[[sycl::reqd_work_group_size(4, 4, 4)]] // expected-note 1 {{conflicting attribute is here}}
void
func_three() {
  not_direct_two();
}
#endif

template <typename Name, typename Type>
[[clang::sycl_kernel]] void __my_kernel__(const Type &bar) {
  bar();
#ifndef TRIGGER_ERROR
  func_one();
#else
  func_two();
  func_three();
#endif
}

template <typename Name, typename Type>
void parallel_for(Type lambda) {
  __my_kernel__<Name>(lambda);
}

void invoke_foo2() {
#ifndef TRIGGER_ERROR
  // CHECK-LABEL:  FunctionDecl {{.*}} invoke_foo2 'void ()'
  // CHECK:        `-FunctionDecl {{.*}}KernelName 'void ()'
  // CHECK:        -IntelReqdSubGroupSizeAttr {{.*}}
  // CHECK:        `-SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
  parallel_for<class KernelName>([]() {});
#else
  parallel_for<class KernelName>([]() {}); // expected-error 3 {{conflicting attributes applied to a SYCL kernel or SYCL_EXTERNAL function}}
#endif
}
