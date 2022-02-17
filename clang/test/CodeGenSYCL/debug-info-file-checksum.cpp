// RUN: %clang -fsycl -g -O0 -emit-llvm -S -g -gcodeview -x c++ \
// RUN: %S/Inputs/debug-info-file-checksum.cpp -o - | FileCheck %s

// CHECK: !DIFile(filename:{{.*}}, directory:{{.*}}, checksumkind: CSK_MD5, checksum: "14764419a8de6ff3ba764463c7fc55b5")

// RUN: %clang -fsycl -g -O0  -emit-llvm -S -g -gcodeview -x c++ \
// RUN: %S/Inputs/debug-info-file-checksum-with-footer.cpp -o - \
// RUN: | FileCheck %s --check-prefix CHECKSUM

// CHECKSUM: !DIFile(filename: "{{.*}}debug-info-file-checksum-with-footer.cpp", directory: {{.*}}, checksumkind: CSK_MD5, checksum: "9ed707faf26dca424db5517a838ab4cc")
