// This test checks that an error is emitted when 
// PCH(Precompiled Header) file generation is forced in -fsycl mode.

// RUN: touch %t.h

// Linux
// RUN: %clang -c -fsycl -x c++-header %t.h -###  %s 2> %t1.txt
// RUN: FileCheck %s -input-file=%t1.txt
// CHECK: Precompiled header generation is not supported with '-fsycl'

// Windows
// RUN: %clang_cl -c -fsycl -x c++-header %t.h -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ERROR %s
// CHECK-ERROR: Precompiled header generation is not supported with '-fsycl'

// /Yc
// RUN: %clang_cl -fsycl /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// CHECK-YC: Precompiled header generation is not supported with '-fsycl'
