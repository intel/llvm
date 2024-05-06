// This test verifies that the integration footer is appended to the source
// file.

// RUN: %clangxx  -E -fsycl %S/Inputs/checkfooter.cpp \
// RUN: -Xclang -include -Xclang %S/Inputs/file1.h \
// RUN: -Xclang -include-footer -Xclang %S/Inputs/file2.h \
// RUN: | FileCheck %s

// CHECK: // __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
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

// CHECK: // __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK: // __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-pc-windows-msvc
// CHECK: # 1 "[[TEMP_INPUTFILE:.+\.cpp]]"
// CHECK: # 1 "[[INTHEADER]]" 1
// CHECK: int file1() {
// CHECK:   return 1;
// CHECK: }
// CHECK: # 3 "<built-in>" 2
// CHECK: # 1 "[[TEMP_INPUTFILE]]" 2
// CHECK: # 1 "[[INPUTFILE:.+\.cpp]]"
// CHECK: int main() {
// CHECK:   int i = 0;
// CHECK:   return i++;
// CHECK: }

// CHECK: # 7 "[[INPUTFILE]]" 2
// CHECK: # 6 "<built-in> 2
// CHECK: # 1 "[[INTFOOTER]]" 1
// CHECK: int file2() {
// CHECK:   return 2;
// CHECK: }
// CHECK: // __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-pc-windows-msvc

