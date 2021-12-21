// RUN: %clangxx -fsycl -fsycl-device-only -I %S %S/Inputs/debug-info-checksum.cpp \
// RUN:   -Xclang -dwarf-version=5 -S -emit-llvm -O0 -g -o - | FileCheck %s
//
// Verify that DICompileUnit points to a correct file and its checksum is also
// correct.
//
// CHECK: !DICompileUnit({{.*}} file: ![[#FILE:]]
// CHECK: ![[#FILE]] = !DIFile(filename: "{{.*}}clang{{[/|\]}}test{{[/|\]}}CodeGenSYCL{{[/|\]}}Inputs{{[/|\]}}debug-info-checksum.cpp"
// CHECK-SAME: checksumkind: CSK_MD5, checksum: "ab8d532478663109402bf4435f520f78"
