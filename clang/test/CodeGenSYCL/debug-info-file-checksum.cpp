// This test checks that a checksum is created correctly for the compiled file,
// and that the same checksum is generated for host and target compilation.
// It also checks that DICompileUnit in host and target compilation is referring
// to the original source file name (not the temporary file created by the
// compilation process) .

// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -fsycl-int-header=%t.header.h -fsycl-int-footer=%t.footer.h \
// RUN: -main-file-name %S/Inputs/checksum.cpp -fsycl-use-main-file-name \
// RUN: -full-main-file-name "%S/Inputs/checksum.cpp" \
// RUN: -gcodeview -debug-info-kind=limited -emit-llvm -O0 -o - "%S/Inputs/checksum.cpp" \
// RUN: | FileCheck %s -check-prefix=COMP1

// RUN: append-file "%S/Inputs/checksum.cpp" \
// RUN: --append=%t.footer.h \
// RUN: --orig-filename="%S/Inputs/checksum.cpp" \
// RUN: --output=%t.checksum.cpp --use-include

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN: -include %t.header.h -dependency-filter %t.header.h \
// RUN: -main-file-name %S/Inputs/checksum.cpp -fsycl-use-main-file-name \
// RUN: -full-main-file-name %S/Inputs/checksum.cpp \
// RUN: -gcodeview -debug-info-kind=limited -emit-llvm -O0 -o - \
// RUN: %t.checksum.cpp \
// RUN: | FileCheck %s -check-prefix=COMP2

// COMP1: !DICompileUnit({{.*}} file: ![[#FILE1:]]
// COMP1: ![[#FILE1]] = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}checksum.cpp"
// COMP1-SAME: checksumkind: CSK_MD5, checksum: "259269f735d83ec32c46a11352458493")

// COMP2: !DICompileUnit({{.*}} file: ![[#FILE2:]]
// COMP2: ![[#FILE2]] = !DIFile(filename: "{{.*}}clang{{.+}}test{{.+}}CodeGenSYCL{{.+}}checksum.cpp"
// COMP2-SAME: checksumkind: CSK_MD5, checksum: "259269f735d83ec32c46a11352458493")

// TODO: Fails on windows because of the use of append-file command that returns
// path with "\\" on windows. getPresumedLoc is failing with Literal String
// parser returning erroneous filename.
// XFAIL: windows-msvc

