// RUN: %clang -### -fsycl %s 2>&1 | FileCheck --check-prefix=CHECK-DEFAULT %s

// CHECK-DEFAULT: "-rpath"

// RUN: %clang -### -fsycl \
// RUN:   -fsycl-implicit-rpath %s 2>&1 | FileCheck --check-prefix=CHECK-FLAG %s

// CHECK-FLAG: "-rpath"

// RUN: %clang -### -fsycl \
// RUN:   -fno-sycl-implicit-rpath %s 2>&1 | FileCheck --check-prefix=CHECK-NOFLAG %s

// CHECK-NOFLAG-NOT: "-rpath"

// REQUIRES: system-linux

