// This test checks that an error is emitted when 
// PCH(Precompiled Header) file is included in -fsycl mode.
// The PCH file is created without the -fsycl option.

// RUN: touch %t.h

// RUN: %clang -c -x c++-header %t.h

// Linux
// -fsycl and -include
// RUN: %clang -fsycl -include %t.h -###  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL %s
// CHECK-SYCL: -fsycl-is-host
// CHECK-SYCL-SAME: -include-pch

// -fsycl and -include-pch
// RUN: %clang -fsycl -include-pch %t.h.gch -###  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PCH %s
// CHECK-PCH: -fsycl-is-host
// CHECK-PCH-SAME: -include-pch
