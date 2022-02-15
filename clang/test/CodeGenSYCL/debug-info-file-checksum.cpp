// RUN: %clang -emit-llvm -fsycl -S -g -gcodeview  \
// RUN: %S/Inputs/debug-info-file-checksum.cpp -o - \
// RUN: | FileCheck %s --check-prefix CHECKSUM-FD
//
// RUN: %clang -emit-llvm -fsycl -S -g -gcodeview  \
// RUN: %S/Inputs/debug-info-file-with-footer-checksum.cpp -o - \
// RUN: | FileCheck %s --check-prefix CHECKSUM-FD-FOOTER

// Verify that DIFile points to a correct file and that a checksum is created for both files.

// CHECKSUM-FD: !DIFile(filename: "{{.*}}debug-info-file-checksum.cpp", directory: "{{.*}}", checksumkind: CSK_MD5, checksum: "cb171f02b4cc8520b2c25f0869a3a43e")

// CHECKSUM-FD-FOOTER: !DIFile(filename: "{{.*}}debug-info-file-with-footer-checksum.cpp", directory: "{{.*}}", checksumkind: CSK_MD5, checksum: "9ed707faf26dca424db5517a838ab4cc")

