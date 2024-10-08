// This test checks that an error is emitted when 
// PCH(Precompiled Header) file generation is forced in -fsycl mode.

// RUN: touch %t.h

// Linux
// RUN: not %clang -c -fsycl -x c++-header %t.h -###  %s 2> %t1.txt
// RUN: FileCheck %s -input-file=%t1.txt
// CHECK: precompiled header generation is not supported with '-fsycl'

// Windows
// RUN: not %clang_cl -c -fsycl -x c++-header %t.h -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ERROR %s
// CHECK-ERROR: precompiled header generation is not supported with '-fsycl'

// /Yc
// RUN: not %clang_cl -fsycl /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// CHECK-YC: precompiled header generation is not supported with '-fsycl'
