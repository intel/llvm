// This test checks that a PCH(Precompiled Header) file is 
// included while performing host compilation in -fsycl mode.

// RUN: touch %t.h

// PCH file.
// RUN: %clang -x c-header -c %t.h -o %t.h.gch

// Linux
// -fsycl and -include
// RUN: %clang -fsycl -include %t.h -###  %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK-SYCL-HOST,CHECK-SYCL-DEVICE %s
// CHECK-SYCL-DEVICE: -fsycl-is-device
// CHECK-SYCL-DEVICE-NOT: -include-pch
// CHECK-SYCL-HOST: -fsycl-is-host
// CHECK-SYCL-HOST-SAME: -include-pch


// -fsycl and -include-pch
// RUN: %clang -fsycl -include-pch %t.h.gch -###  %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK-PCH-HOST,CHECK-PCH-DEVICE %s
// CHECK-PCH-DEVICE: -fsycl-is-device
// CHECK-PCH-DEVICE-NOT: -include-pch
// CHECK-PCH-HOST: -fsycl-is-host
// CHECK-PCH-HOST-SAME: -include-pch

// Windows
// RUN: %clang_cl -fsycl --target=x86_64-unknown-linux-gnu /Yupchfile.h /FIpchfile.h -### %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHECK-YU-HOST,CHECK-YU-DEVICE %s
// CHECK-YU-DEVICE: -fsycl-is-device
// CHECK-YU-DEVICE-NOT: -include-pch 
// CHECK-YU-HOST: -fsycl-is-host
// CHECK-YU-HOST-SAME: -include-pch
