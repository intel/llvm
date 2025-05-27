// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/hierarchical_parallelism | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/hierarchical_parallelism>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/hierarchical_parallelism
// CHECK-EMPTY:
