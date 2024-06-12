// This test verifies that the integration footer is appended to the source
// file.

// RUN: %clang_cc1  -E -fsycl-is-host %S/Inputs/checkfooter.cpp \
// RUN: -include %S/Inputs/file1.h -include-footer %S/Inputs/file2.h \
// RUN: | FileCheck %s

// CHECK: # 1 "[[INPUTFILE:.+\.cpp]]"
// CHECK: # 1 "[[INTHEADER:.+\.h]]" 1
// CHECK: int file1() {
// CHECK:   return 1;
// CHECK: }
// CHECK: # 2 "<built-in>" 2
// CHECK: # 1 "[[INPUTFILE:.+\.cpp]]" 2
// CHECK: int main() {
// CHECK:   int i = 0;
// CHECK:   return i++;
// CHECK: }
// CHECK: # 1 "[[INTFOOTER:.+\.h]]" 1
// CHECK: int file2() {
// CHECK:   return 2;
// CHECK: }
