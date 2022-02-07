// RUN: %clang -emit-llvm -S -g -gcodeview -x c %S/Inputs/debug-info-file-checksum.c -o - | FileCheck %s
// RUN: %clang -emit-llvm -S -gdwarf-5 -x c %S/Inputs/debug-info-file-checksum.c -o - | FileCheck %s

// Check that "checksum" is created correctly for the compiled file.

// CHECK: !DIFile(filename:{{.*}}, directory:{{.*}}, checksumkind: CSK_MD5, checksum: "c85df301cb651d1ee0435591c6d9c558")

// Ensure #line directives (in already pre-processed files) do not emit checksums
// RUN: %clang -emit-llvm -S -g -gcodeview -x c %S/Inputs/debug-info-file-checksum-pre.cpp -o - | FileCheck %s --check-prefix NOCHECKSUM

// NOCHECKSUM: !DIFile(filename: "{{.*}}code-coverage-filter1.h", directory: "{{[^"]*}}")
// NOCHECKSUM: !DIFile(filename: "{{.*}}code-coverage-filter2.h", directory: "{{[^"]*}}")
// NOCHECKSUM: !DIFile(filename: "{{.*}}debug-info-file-checksum.c", directory: "{{[^"]*}}")

// Ensure #line directives without name do emit checksums
// RUN: %clang -emit-llvm -S -g -gcodeview -x c %S/Inputs/debug-info-file-checksum-line.cpp -o - | FileCheck %s --check-prefix CHECKSUM

// CHECKSUM: !DIFile(filename: "{{.*}}debug-info-file-checksum-line.cpp", directory:{{.*}}, checksumkind: CSK_MD5, checksum: "4456fffb9436b349cea43c698a726b08")
