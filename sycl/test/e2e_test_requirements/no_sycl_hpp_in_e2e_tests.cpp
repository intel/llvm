// REQUIRES: linux
//
// RUN: grep -r -l 'sycl.hpp' %S/../../test-e2e | FileCheck  %s
// RUN: grep -r -l 'sycl.hpp' %S/../../test-e2e | wc -l | FileCheck %s --check-prefix CHECK-NUM-MATCHES
//
// CHECK-DAG: README.md
// CHECK-DAG: lit.cfg.py
//
// CHECK-NUM-MATCHES: 5
//
// This test verifies that `<sycl/sycl.hpp>` isn't used in E2E tests. Instead,
// fine-grained includes should used, see
// https://github.com/intel/llvm/tree/sycl/sycl/test-e2e#sycldetailcorehpp
