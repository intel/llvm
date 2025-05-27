// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/interop_handle | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/interop_handle>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/interop_handle
// CHECK-EMPTY:
