// RUN: %clang -emit-llvm -S -g -gcodeview -x c++ \
// RUN: %S/Inputs/debug-info-file-with-footer-checksum.cpp -o - \
// RUN: | FileCheck %s --check-prefix CHECKSUM

// Verify that DICompileUnit points to a correct file and that a checksum is
// created.
//
// CHECKSUM: !DIFile(filename: "{{.*}}debug-info-file-with-footer-checksum.cpp", directory: "{{.*}}", checksumkind: CSK_MD5, checksum: "8ab6353658dc280a777387a89de62816")
