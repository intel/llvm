// RUN: not syclbin-dump nonexistent.syclbin 2>&1 | FileCheck %s --check-prefix CHECK-NONEXISTENT-FILE
// RUN: not syclbin-dump %S/Inputs/malformed.syclbin 2>&1 | FileCheck %s --check-prefix CHECK-MALFORMED-FILE
// UNSUPPORTED: linux
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19404
// CHECK-NONEXISTENT-FILE: Failed to open or read file nonexistent.syclbin
// CHECK-MALFORMED-FILE: Invalid data was encountered while parsing the file
