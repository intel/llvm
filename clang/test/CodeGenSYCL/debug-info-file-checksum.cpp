// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -fsycl-int-header=%t.header.h -fsycl-int-footer=%t.footer.h \
// RUN: -main-file-name checksum.cpp -fsycl-use-main-file-name \
// RUN: -full-main-file-name "%S/checksum.cpp" \
// RUN: -gcodeview -debug-info-kind=limited -emit-llvm -O0 -o - "%S/checksum.cpp" \
// RUN: | FileCheck %s -check-prefix=COMP1

// RUN: append-file "%S/checksum.cpp" \
// RUN: --append=%t.footer.h \
// RUN: --orig-filename="%S/checksum.cpp" \
// RUN: --output=%t.checksum.cpp --use-include

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN: -include %t.header.h -dependency-filter %t.header.h \
// RUN: -main-file-name checksum.cpp -fsycl-use-main-file-name \
// RUN: -full-main-file-name %S/checksum.cpp \
// RUN: -gcodeview -debug-info-kind=limited -emit-llvm -O0 -o - \
// RUN: %t.checksum.cpp \
// RUN: | FileCheck %s -check-prefix=COMP2

// COMP1: !DICompileUnit({{.*}} file: ![[#FILE1:]]
// COMP1: ![[#FILE1]] = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}checksum.cpp"
// COMP1-SAME: checksumkind: CSK_MD5, checksum: "259269f735d83ec32c46a11352458493")

// COMP2: !DICompileUnit({{.*}} file: ![[#FILE2:]]
// COMP2: ![[#FILE2]] = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}checksum.cpp"
// COMP2-SAME: checksumkind: CSK_MD5, checksum: "259269f735d83ec32c46a11352458493")
