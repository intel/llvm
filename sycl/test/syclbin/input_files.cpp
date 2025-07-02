// RUN: syclbin-dump nonexistent.syclbin | FileCheck %s --check-prefix CHECK-NONEXISTENT-FILE
// RUN: syclbin-dump malformed.syclbin | FileCheck %s --check-prefix CHECK-MALFORMED-FILE

// CHECK-NONEXISTENT-FILE: Failed to open or read file nonexistent.syclbin
// CHECK-MALFORMED-FILE: Invalid data was encountered while parsing the file
