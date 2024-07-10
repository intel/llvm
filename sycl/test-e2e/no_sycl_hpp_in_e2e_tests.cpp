// REQUIRES: linux
//
// RUN: grep -r -l 'sycl.hpp' %S | FileCheck  %s
// RUN: grep -r -l 'sycl.hpp' %S | wc -l | FileCheck %s --check-prefix CHECK-NUM-MATCHES
//
// CHECK-DAG: README.md
// CHECK-DAG: no_sycl_hpp_in_e2e_tests.cpp
// CHECK-DAG: lit.cfg.py
//
// CHECK-NUM-MATCHES: 4
//
// This test verifies that `<sycl/sycl.hpp>` isn't used in E2E tests. Instead,
// fine-grained includes should used, see
// https://github.com/intel/llvm/tree/sycl/sycl/test-e2e#sycldetailcorehpp
