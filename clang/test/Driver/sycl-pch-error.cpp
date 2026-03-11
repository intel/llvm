// This test checks that an error is emitted when
// an invalid PCH(Precompiled Header) file is used in -fsycl mode.

// RUN: touch %t.h

// Linux
// RUN: not %clang -c -fsycl -include-pch %t.h %s 2> %t1.txt
// RUN: FileCheck %s -input-file=%t1.txt
// CHECK: input is not a PCH file

// Windows
// /Yu
// RUN: not %clang_cl -fsycl /Yu%t.h /c -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU %s
// CHECK-YU: No such file or directory
