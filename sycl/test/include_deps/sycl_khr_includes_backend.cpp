// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/backend | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/backend>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/backend
// CHECK-EMPTY:
