// RUN: %clang -### -target x86_64 -c -faltmathlib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB %s
// RUN: %clang -### -target x86_64 -c -faltmathlib=SVMLAltMathLibrary %s 2>&1 | FileCheck -check-prefix CHECK-SMVL %s
// RUN: %clang -### -target x86_64 -c -faltmathlib=TestAltMathLibrary %s 2>&1 | FileCheck -check-prefix CHECK-TEST %s
// RUN: not %clang -c -target x86_64 -faltmathlib=something %s 2>&1 | FileCheck %s -check-prefix=ERR

// CHECK-NOLIB: "-faltmathlib=none"
// CHECK-SMVL: "-faltmathlib=SVMLAltMathLibrary"
// CHECK-TEST: "-faltmathlib=TestAltMathLibrary"

// ERR: error: invalid value 'something' in '-faltmathlib=something'

