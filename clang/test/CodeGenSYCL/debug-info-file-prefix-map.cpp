// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -fsycl-int-header=%t.header.h -fsycl-int-footer=%t.footer.h \
// RUN: -main-file-name %S/Inputs/builtin.cpp -fsycl-use-main-file-name \
// RUN: -full-main-file-name "%S/Inputs/builtin.cpp" \
// RUN: -fmacro-prefix-map=%S/Inputs/= -fcoverage-prefix-map=%S/Inputs/= \
// RUN: -fdebug-prefix-map=%S/Inputs/= \
// RUN: -debug-info-kind=constructor -emit-llvm -O0 -o - "%S/Inputs/builtin.cpp" \
// RUN: | FileCheck %s


// CHECK: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: ![[#FILE1:]]
// CHECK-NEXT: ![[#FILE1]] = !DIFile(filename: "builtin.cpp", directory: "{{.*}}")
// CHECK: ![[#FILE2:]] = !DIFile(filename: "builtin.cpp", directory: "{{.*}}")
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: ![[#FILE2]]
