// RUN: %clang -### -fsycl  %s -fsycl-allow-func-ptr=off 2>&1 | FileCheck %s --check-prefixes=CHECK-OFF,CHECK
// RUN: %clang -### -fsycl  %s -fno-sycl-allow-func-ptr 2>&1 | FileCheck %s --check-prefixes=CHECK-OFF,CHECK
// RUN: %clang -### -fsycl  %s -fsycl-allow-func-ptr 2>&1 | FileCheck %s --check-prefixes=CHECK-LABELED,CHECK
// RUN: %clang -### -fsycl  %s -fsycl-allow-func-ptr=labeled 2>&1 | FileCheck %s --check-prefixes=CHECK-LABELED,CHECK
// RUN: %clang -### -fsycl  %s -fsycl-allow-func-ptr=defined 2>&1 | FileCheck %s --check-prefixes=CHECK-DEFINED,CHECK

// CHECK: "-triple" "spir64{{.*}}" "-fsycl-is-device"
// CHECK-OFF-SAME: "-fsycl-allow-func-ptr=off"
// CHECK-LABELED-SAME: "-fsycl-allow-func-ptr=labeled"
// CHECK-DEFINED-SAME: "-fsycl-allow-func-ptr=defined"
