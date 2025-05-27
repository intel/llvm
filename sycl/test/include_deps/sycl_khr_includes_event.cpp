// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/event | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/event>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/event
// CHECK-EMPTY:
