// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/khr/includes/property_list | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/khr/includes/property_list>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: khr/includes/property_list
// CHECK-EMPTY:
