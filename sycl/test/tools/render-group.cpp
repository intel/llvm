// REQUIRES: linux

// Compile Inputs/mock_renderd_access.c to a shared library and then use
// LD_PRELOAD to mock the stat and open functions.
// RUN: %clang -shared -fPIC -o %t-mock_stat.so %S/Inputs/mock_renderd_access.c

// Check the case when /dev/dri/renderD12* don't exist.
// RUN: env MOCK_GLOB_MODE=notfound LD_PRELOAD=%t-mock_stat.so sycl-ls --verbose --ignore-device-selectors 2>&1 | FileCheck %s -check-prefix=CHECK-NOTFOUND
// We don't expect any warning about permissions in this case.
// CHECK-NOTFOUND-NOT: WARNING: Unable to access /dev/dri/renderD12

// Check the case when /dev/dri/renderD12* exist but not accessible.
// RUN: env MOCK_GLOB_MODE=exists MOCK_OPEN_MODE=deny LD_PRELOAD=%t-mock_stat.so sycl-ls --verbose --ignore-device-selectors 2>&1 | FileCheck %s --check-prefix=CHECK-DENY
// CHECK-DENY: WARNING: Unable to access /dev/dri/renderD128 due to permissions (EACCES).
// CHECK-DENY-NEXT: You might be missing the 'render' group locally.
// CHECK-DENY-NEXT: Try: sudo usermod -a -G render $USER
// CHECK-DENY-NEXT: Then log out and log back in.

// Check the case when /dev/dri/renderD12* exist but only the first one is
// accessible.
// RUN: env MOCK_GLOB_MODE=exists MOCK_OPEN_MODE=deny_second LD_PRELOAD=%t-mock_stat.so sycl-ls --verbose --ignore-device-selectors 2>&1 | FileCheck %s --check-prefix=CHECK-DENY-SECOND
// CHECK-DENY-SECOND-NOT: WARNING: Unable to access /dev/dri/renderD128 due to permissions (EACCES).
// CHECK-DENY-SECOND: WARNING: Unable to access /dev/dri/renderD129 due to permissions (EACCES).
// CHECK-DENY-SECOND-NEXT: You might be missing the 'render' group locally.
// CHECK-DENY-SECOND-NEXT: Try: sudo usermod -a -G render $USER
// CHECK-DENY-SECOND-NEXT: Then log out and log back in.

// Check the case when /dev/dri/renderD12* exist and is accessible.
// RUN: env MOCK_GLOB_MODE=exists MOCK_OPEN_MODE=allow LD_PRELOAD=%t-mock_stat.so sycl-ls --verbose --ignore-device-selectors 2>&1 | FileCheck %s --check-prefix=CHECK-GRANT
// CHECK-GRANT-NOT: WARNING: Unable to access /dev/dri/renderD12
