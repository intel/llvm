// RUN: %clang_cc1 -fsycl-is-host -I %S %S/Inputs/debug-info-checksum.cpp \
// RUN: -triple x86_64-unknown-linux-gpu \
// RUN: -main-file-name "%S/Inputs/debug-info-file-with-footer-checksum.cpp" \
// RUN: -fsycl-use-main-file-name -dwarf-version=5 -S -emit-llvm \
// RUN: -O0 -debug-info-kind=constructor -o - \
// RUN: | FileCheck %s --check-prefix CHECKSUM

// Verify that DIFile points to a correct file and that a checksum is created.
//
// CHECKSUM: !DIFile(filename: "{{.*}}debug-info-file-with-footer-checksum.cpp", directory: "{{.*}}", checksumkind: CSK_MD5, checksum: "8ab6353658dc280a777387a89de62816")
