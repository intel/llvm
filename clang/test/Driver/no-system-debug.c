// Ensure -fno-system-debug is passed to cc1
// RUN: %clang -fno-system-debug -### -c %s 2>&1 | FileCheck %s

// CHECK: -cc1
// CHECK-SAME: -fno-system-debug

