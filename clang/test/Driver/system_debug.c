// Ensure -fno-system-debug is passed to cc1 appropriately

// RUN: %clang -fsystem-debug -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-POS
// RUN: %clang -fno-system-debug -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-NEG
// RUN: %clang -fno-system-debug -fsystem-debug -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-POS
// RUN: %clang -fsystem-debug -fno-system-debug -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-NEG

// CHECK-POS-NOT: -fno-system-debug

// CHECK-NEG: -cc1
// CHECK-NEG-SAME: -fno-system-debug

