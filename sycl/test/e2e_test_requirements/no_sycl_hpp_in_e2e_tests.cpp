// REQUIRES: linux
//
// RUN: grep -r -l 'sycl.hpp' %S/../../test-e2e | FileCheck  %s
// RUN: grep -r -l 'sycl.hpp' %S/../../test-e2e | grep -v 'test-e2e/PerformanceTests/' | wc -l | FileCheck %s --check-prefix CHECK-NUM-MATCHES
//
// CHECK-DAG: README.md
// CHECK-DAG: lit.cfg.py
//
// CHECK-NUM-MATCHES: 30
// localdisk2/kparasyr/LLVM-FORK/llvm/sycl/test/e2e_test_requirements/no_sycl_hpp_in_e2e_tests.cpp/
//  This test verifies that `<sycl/sycl.hpp>` isn't used in E2E tests. Instead,
//  fine-grained includes should used, see
//  https://github.com/intel/llvm/tree/sycl/sycl/test-e2e#sycldetailcorehpp
