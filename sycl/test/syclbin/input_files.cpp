// RUN: not syclbin-dump nonexistent.syclbin 2>&1 | FileCheck %s --check-prefix CHECK-NONEXISTENT-FILE
// RUN: not syclbin-dump %S/Inputs/malformed.syclbin 2>&1 | FileCheck %s --check-prefix CHECK-MALFORMED-FILE
// CHECK-NONEXISTENT-FILE: Failed to open or read file nonexistent.syclbin:
// CHECK-MALFORMED-FILE: Failed to parse SYCLBIN file: Incorrect SYCLBIN magic number.
