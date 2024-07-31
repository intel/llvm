// This test tests the option -fsycl-use-integration-headers

// Ensure that by default the -fsycl-use-integration-headers gets passed to the cc1 invocation.
// RUN: %clang -### -c -fsycl %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: "-fsycl-use-integration-headers" "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"

// The next two tests make sure that the -fsycl-use-integration-headers
// commmand line arguments get properly passed to the cc1 invocation.
// RUN: %clang -### -c -fsycl -fsycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=SYCL
// SYCL: "-fsycl-use-integration-headers" "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"
// RUN: %clang -### -c -fsycl-device-only -fsycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=DEVO
// DEVO: "-fsycl-use-integration-headers" "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"

// The next two tests make sure that the -fno-sycl-use-integration-headers
// commmand line arguments get properly passed to the cc1 invocation.
// RUN: %clang -### -c -fsycl -fno-sycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=SYCL_NO
// SYCL_NO-NOT: "-fsycl-use-integration-headers"
// SYCL_NO-NOT: "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"
// RUN: %clang -### -c -fsycl-device-only -fno-sycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=DEVO_NO
// DEVO_NO-NOT: "-fsycl-use-integration-headers"
// DEVO_NO-NOT: "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"

// The next three tests check that we detect errors when:
//   1.  either option is used without -fsycl or -fsycl-device-only
//   2.  -fno-sycl-use-integration-headers is used with -fsycl-host-compiler=
// RUN: not %clang -### -c -fsycl-use-integration-headers %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-f[no-]sycl-use-integration-headers' requires SYCL, use '-fsycl' or '-fsycl-device-only' to enable it
// RUN: not %clang -### -c -fno-sycl-use-integration-headers %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-NO-FNOSYCL %s
// CHK-NO-FNOSYCL: error: '-f[no-]sycl-use-integration-headers' requires SYCL, use '-fsycl' or '-fsycl-device-only' to enable it
// RUN: not %clang -### -c -fsycl -fno-sycl-use-integration-headers -fsycl-host-compiler=g++  %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-FSYCL-CONFLICT %s
// CHK-FSYCL-CONFLICT: error: the option -fsycl-host-compiler= conflicts with -fno-sycl-use-integration-headers

// The next two tests make sure that users can override the defined/undefined
// macro "__INTEL_SYCL_USE_INTEGRATION_HEADERS" if necessary.
// RUN: %clang -c -fsycl -fsycl-use-integration-headers -U__INTEL_SYCL_USE_INTEGRATION_HEADERS %s -dM -E 2>&1 | FileCheck %s -check-prefix=SYCL_NO_MACRO
// SYCL_NO_MACRO-NOT: #define __INTEL_SYCL_USE_INTEGRATION_HEADERS
// RUN: %clang -c -fsycl -fno-sycl-use-integration-headers -D__INTEL_SYCL_USE_INTEGRATION_HEADERS %s -dM -E 2>&1 | FileCheck %s -check-prefix=SYCL_MACRO
// SYCL_MACRO: #define __INTEL_SYCL_USE_INTEGRATION_HEADERS
