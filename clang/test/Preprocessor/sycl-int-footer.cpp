// This test verifies that the integration footer is appended to the source
// file.

// RUN: %clang_cc1  -E -fsycl-is-host %S/Inputs/checkfooter1.cpp \
// RUN: -include %S/Inputs/file1.h -include-footer %S/Inputs/file2.h \
// RUN: | FileCheck %s -check-prefix CHECKFOOTER1

// RUN: %clang_cc1  -E -fsycl-is-host %S/Inputs/checkfooter2.cpp \
// RUN: -include %S/Inputs/header.h -include-footer %S/Inputs/footer.h \
// RUN: | FileCheck %s -check-prefix CHECKFOOTER2

// CHECKFOOTER1: # 1 "[[INPUTFILE1:.+\.cpp]]"
// CHECKFOOTER1: # 1 "[[INTHEADER1:.+\.h]]" 1
// CHECKFOOTER1: int file1() {
// CHECKFOOTER1:   return 1;
// CHECKFOOTER1: }
// CHECKFOOTER1: # 1 "[[INPUTFILE2:.+\.cpp]]" 2
// CHECKFOOTER1-NEXT: int main() {
// CHECKFOOTER1-NEXT:   int i = 0;
// CHECKFOOTER1-NEXT:   return i++;
// CHECKFOOTER1-NEXT: }
// CHECKFOOTER1: # 1 "[[INTFOOTER1:.+\.h]]" 1
// CHECKFOOTER1-NEXT: int file2() {
// CHECKFOOTER1-NEXT:   return 2;
// CHECKFOOTER1-NEXT: }

// CHECKFOOTER2: # 1 "[[INPUTFILE_CF2:.+\.cpp]]"
// CHECKFOOTER2: # 1 "[[INTHEADER:.+\.h]]" 1
// CHECKFOOTER2-NEXT: # 1 "[[INTHEADER3:.+\.h]]" 1
// CHECKFOOTER2-NEXT: # 1 "[[INTHEADER4:.+\.h]]" 1
// CHECKFOOTER2: void foo3();
// CHECKFOOTER2-NEXT: # 2 "[[INTHEADER3]]" 2
// CHECKFOOTER2: int bar(int size);
// CHECKFOOTER2: # 2 "[[INTHEADER2:.+\.h]]" 2
// CHECKFOOTER2: void foo() {
// CHECKFOOTER2-NEXT:   bar(22);
// CHECKFOOTER2-NEXT: }
// CHECKFOOTER2: # 1 "[[INPUTFILE_CF2:.+\.cpp]]"
// CHECKFOOTER2-NEXT: int main() {
// CHECKFOOTER2-NEXT:   foo();
// CHECKFOOTER2-NEXT: }
// CHECKFOOTER2: # 1 "[[INTFOOTER2:.+\.h]]" 1
// CHECKFOOTER2-NEXT: # 1 "[[INTFOOTER3:.+\.h]]" 1
// CHECKFOOTER2: # 2 "[[INTFOOTER2]]" 2
// CHECKFOOTER2: void foo2(void);
// CHECKFOOTER2: # 4 "[[INPUTFILE_CF2]]" 2
