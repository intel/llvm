// This test verifies that the integration footer is appended to the source
// file.

// RUN: %clang_cc1  -E -fsycl-is-host %S/Inputs/checkfooter1.cpp \
// RUN: -include-internal-header %S/Inputs/file1.h \
// RUN: -include-internal-footer %S/Inputs/file2.h \
// RUN: | FileCheck %s -check-prefix CHECKFOOTER1

// RUN: %clang_cc1  -E -fsycl-is-host %S/Inputs/checkfooter2.cpp \
// RUN: -include-internal-header %S/Inputs/header.h \
// RUN: -include-internal-footer %S/Inputs/footer.h \
// RUN: | FileCheck %s -check-prefix CHECKFOOTER2

// CHECKFOOTER1: # 1 "[[INPUTFILE1:.+\.cpp]]"
// CHECKFOOTER1: int main() {
// CHECKFOOTER1:  int i = 0;
// CHECKFOOTER1:  return i++;
// CHECKFOOTER1:}

// CHECKFOOTER2: # 1 "[[INPUTFILE_CF2:.+\.cpp]]"
// CHECKFOOTER2: # 1 "[[INPUTFILE_CF2:.+\.cpp]]"
// CHECKFOOTER2-NEXT: int main() {
// CHECKFOOTER2-NEXT:   foo();
// CHECKFOOTER2-NEXT: }
