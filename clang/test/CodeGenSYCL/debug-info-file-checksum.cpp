// RUN: %clang_cc1 -fsycl-is-device -I %S %S/Inputs/debug-info-file-checksum.cpp \
// RUN: -triple spir64-unknown-unknown \
// RUN: -main-file-name "%S/Inputs/debug-info-file-checksum.cpp" \
// RUN: -fsycl-use-main-file-name -gcodeview -debug-info-kind=constructor -S -emit-llvm \
// RUN: -O0 -o - | FileCheck %s

// Check that "checksum" is created correctly for the compiled file and that the same checksum is
// generated for the input file appended with the footer.

// CHECK: !DICompileUnit({{.*}} file: ![[#FILE:]]
// CHECK: ![[#FILE:]] = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}Inputs{{.+}}debug-info-file-checksum.cpp"
// CHECK-SAME: checksumkind: CSK_MD5, checksum: "1d5684eee4c20312552d44759d75c757"

// RUN: %clang_cc1 -fsycl-is-device -I %S %S/Inputs/debug-info-file-checksum-with-footer.cpp \
// RUN: -triple spir64-unknown-unknown \
// RUN: -main-file-name "%S/Inputs/debug-info-file-checksum.cpp" \
// RUN: -fsycl-use-main-file-name -gcodeview -debug-info-kind=constructor -S -emit-llvm \
// RUN: -O0 -o - | FileCheck --check-prefix=CHECKSUM %s

// CHECKSUM: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1
// CHECKSUM: !1 = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}Inputs{{.+}}debug-info-file-checksum.cpp"
// CHECKSUM-SAME: checksumkind: CSK_MD5, checksum: "1d5684eee4c20312552d44759d75c757")
