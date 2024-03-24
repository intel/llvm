// Ensure -fsystem-debug is passed to cc1
// RUN: %clang -fsystem-debug -### -c %s 2>&1 | FileCheck %s

// CHECK: -cc1
// CHECK-SAME: -system-debug

