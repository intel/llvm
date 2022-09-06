// RUN: %clang_cc1 -fsycl-is-device %S/Inputs/checksum.cpp \
// RUN: -triple spir64-unknown-unknown \
// RUN: -main-file-name "%S/Inputs/checksum.cpp" \
// RUN: -fsycl-use-main-file-name -gcodeview -debug-info-kind=constructor \
// RUN: -emit-llvm -O0 -o - | FileCheck %s

// Check that "checksum" is created correctly for the compiled file and
// that the same checksum is generated for the input file appended with
// the footer.

// CHECK: !DICompileUnit({{.*}} file: ![[#FILE:]]
// CHECK: ![[#FILE:]] = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}Inputs{{.+}}checksum.cpp"
// CHECK-SAME: checksumkind: CSK_MD5, checksum: "259269f735d83ec32c46a11352458493")

// RUN: %clang_cc1 -fsycl-is-host %S/Inputs/checksum-with-footer.cpp \
// RUN: -triple x86_64-pc-windows-msvc \
// RUN: -main-file-name %S/Inputs/checksum.cpp \
// RUN: -fsycl-use-main-file-name -gcodeview -debug-info-kind=constructor \
// RUN: -S -emit-llvm -O0  -o - | FileCheck %s


// CHECKSUM: distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1
// CHECKSUM: !1 = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}Inputs{{.+}}checksum.cpp"
// CHECKSUM-SAME: checksumkind: CSK_MD5, checksum: "259269f735d83ec32c46a11352458493")
