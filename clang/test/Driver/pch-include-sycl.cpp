// This test checks that an error is emitted when 
// PCH(Precompiled Header) file is included in -fsycl mode.
// The PCH file is created without the -fsycl option.

// RUN: touch %t.h

// RUN: %clang -c -x c++-header %t.h

// Linux
// -fsycl and -include
// RUN: %clang -fsycl -include %t.h -###  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL %s
// CHECK-SYCL: Precompiled header generation is not supported with '-fsycl'

// -fsycl-device-only and -include
// RUN: %clang -fsycl-device-only -include %t.h -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEVICE-ONLY %s
// CHECK-DEVICE-ONLY: Precompiled header generation is not supported with '-fsycl'

// -fsycl, -fsycl-device-only and -include
// RUN: %clang -fsycl -fsycl-device-only -include %t.h -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-BOTH %s
// CHECK-BOTH: Precompiled header generation is not supported with '-fsycl'

// -fsycl and -include-pch
// RUN: %clang -fsycl -include-pch %t.h -###  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-INCLUDE %s
// CHECK-SYCL-INCLUDE: Precompiled header generation is not supported with '-fsycl'

// -fsycl-device-only and -include-pch
// RUN: %clang -fsycl-device-only -include-pch %t.h -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEVICE-ONLY-INCLUDE %s
// CHECK-DEVICE-ONLY-INCLUDE: Precompiled header generation is not supported with '-fsycl'

// -fsycl,-fsycl-device-only and -include-pch
// RUN: %clang -fsycl -fsycl-device-only -include-pch %t.h -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-BOTH-INCLUDE %s
// CHECK-BOTH-INCLUDE: Precompiled header generation is not supported with '-fsycl'

// Windows
// -fsycl
// RUN: %clang_cl -fsycl --target=x86_64-unknown-linux-gnu /Yupchfile.h /FIpchfile.h -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=CHECK-YU %s
// CHECK-YU: Precompiled header generation is not supported with '-fsycl'

// -fsycl-device-only
// RUN: %clang_cl -fsycl-device-only --target=x86_64-unknown-linux-gnu /Yupchfile.h /FIpchfile.h -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=CHECK-YU-DEVICE %s
// CHECK-YU-DEVICE: Precompiled header generation is not supported with '-fsycl'

// -fsycl and -fsycl-device-only
// RUN: %clang_cl -fsycl -fsycl-device-only --target=x86_64-unknown-linux-gnu /Yupchfile.h /FIpchfile.h -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=CHECK-YU-BOTH %s
// CHECK-YU-BOTH: Precompiled header generation is not supported with '-fsycl'
