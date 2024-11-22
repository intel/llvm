// This test checks that in the presence of the option -fdebug-prefix-map
// the DICompileUnit information is correct, i.e  test filename and directory
// path are correct.

// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -fsycl-int-header=%t.header.h -fsycl-int-footer=%t.footer.h \
// RUN: -main-file-name %s \
// RUN: -fmacro-prefix-map=%S/= -fcoverage-prefix-map=%S/= \
// RUN: -fdebug-prefix-map=%S/= \
// RUN: -debug-info-kind=constructor -emit-llvm -O0 -o - %s \
// RUN: | FileCheck %s


// CHECK: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: ![[#FILE1:]]
// CHECK-NEXT: ![[#FILE1]] = !DIFile(filename: "debug-info-file-prefix-map.cpp"
// CHECK: ![[#FILE2:]] = !DIFile(filename: "debug-info-file-prefix-map.cpp"
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: ![[#FILE1]]

void a(__builtin_va_list);
using ::a;
