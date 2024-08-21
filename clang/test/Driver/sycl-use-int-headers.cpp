// This test tests the option -f[no-]sycl-use-integration-headers

// Ensure that by default:
//   1. with -fsycl, -fsycl-use-integration-headers gets passed to the cc1 invocation.
//   2. without -fsycl, -fsycl-use-integration-headers does not get passed.
// RUN: %clang -### -c -fsycl %s 2>&1 | FileCheck %s -check-prefix=CHECK-S-DEFAULT
// CHECK-S-DEFAULT: "-fsycl-use-integration-headers" "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"
// RUN: %clang -### -c %s 2>&1 | FileCheck %s -check-prefix=CHECK-NOSYCL-DEFAULT
// CHECK-NOSYCL-DEFAULT-NOT: "-fsycl-use-integration-headers"
// CHECK-NOSYCL-DEFAULT-NOT: "-fno-sycl-use-integration-headers"

// The next two tests make sure that the -fsycl-use-integration-headers
// commmand line argument gets properly passed to the cc1 invocation.
// RUN: %clang -### -c -fsycl -fsycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=SYCL
// SYCL: "-fsycl-use-integration-headers" "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"
// RUN: %clang -### -c -fsycl -fno-sycl-use-integration-headers -fsycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=SYCL_NEG_POS
// SYCL_NEG_POS: "-fsycl-use-integration-headers" "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"

// The next two tests make sure that the -fno-sycl-use-integration-headers
// commmand line argument does not pass -fsycl-use-integration-headers to
// the cc1 invocation.
// RUN: %clang -### -c -fsycl -fno-sycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=SYCL_NO
// SYCL_NO-NOT: "-fsycl-use-integration-headers"
// SYCL_NO-NOT: "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"
// RUN: %clang -### -c -fsycl -fsycl-use-integration-headers -fno-sycl-use-integration-headers %s 2>&1 | FileCheck %s -check-prefix=SYCL_POS_NEG
// SYCL_POS_NEG-NOT: "-fsycl-use-integration-headers"
// SYCL_POS_NEG-NOT: "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"

// This test tests that neither -fsycl-use-integration-headers nor
// -fno-sycl-use-integration-headers is passed to the cc1 invocation
// when -fsycl is not specified.
// RUN: %clang -### -c %s 2>&1 | FileCheck %s -check-prefix=NO_SYCL_OPT
// NO_SYCL_OPT-NOT: "-fsycl-use-integration-headers"
// NO_SYCL_OPT-NOT: "-fno-sycl-use-integration-headers"
// NO_SYCL_OPT-NOT: "-D__INTEL_SYCL_USE_INTEGRATION_HEADERS"

// The next three tests check that we detect errors when:
//   1.  either option is used without -fsycl
//   2.  -fno-sycl-use-integration-headers is used with -fsycl-host-compiler=
// RUN: not %clang -### -c -fsycl-use-integration-headers %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-f[no-]sycl-use-integration-headers' requires SYCL, use '-fsycl' to enable it
// RUN: not %clang -### -c -fno-sycl-use-integration-headers %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-NO-FNOSYCL %s
// CHK-NO-FNOSYCL: error: '-f[no-]sycl-use-integration-headers' requires SYCL, use '-fsycl' to enable it
// RUN: not %clang -### -c -fsycl -fno-sycl-use-integration-headers -fsycl-host-compiler=g++  %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-FSYCL-CONFLICT %s
// CHK-FSYCL-CONFLICT: error: the option -fsycl-host-compiler conflicts with -fno-sycl-use-integration-headers

// The next two tests make sure that users can override the defined/undefined
// macro "__INTEL_SYCL_USE_INTEGRATION_HEADERS" if necessary.
// RUN: %clang -c -fsycl -fsycl-use-integration-headers -U__INTEL_SYCL_USE_INTEGRATION_HEADERS %s -dM -E 2>&1 | FileCheck %s -check-prefix=SYCL_NO_MACRO
// SYCL_NO_MACRO-NOT: #define __INTEL_SYCL_USE_INTEGRATION_HEADERS
// RUN: %clang -c -fsycl -fno-sycl-use-integration-headers -D__INTEL_SYCL_USE_INTEGRATION_HEADERS %s -dM -E 2>&1 | FileCheck %s -check-prefix=SYCL_MACRO
// SYCL_MACRO: #define __INTEL_SYCL_USE_INTEGRATION_HEADERS
