// This test tests the option -fsycl-use-builtins-for-intgration

// Ensure that by default the -fsycl-use-builtins-for-intgration doesn't get passed to the cc1 invocation.
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT-NOT: "fsycl-use-builtins-for-integration"

// The next two tests make sure that the -fsycl-use-builtins-for-integration
// commmand line arguments get properly passed to the cc1 invocation.
// RUN: %clang -### -fsycl -fsycl-use-builtins-for-integration %s 2>&1 | FileCheck %s -check-prefix=PRIM
// PRIM: "-fsycl-use-builtins-for-integration" "-D__INTEL_SYCL_USE_BUILTINS_FOR_INTEGRATION"
// RUN: %clang -### -fsycl-device-only -fsycl-use-builtins-for-integration %s 2>&1 | FileCheck %s -check-prefix=DEVO
// DEVO: "-fsycl-use-builtins-for-integration" "-D__INTEL_SYCL_USE_BUILTINS_FOR_INTEGRATION"

// The next two tests check that we detect errors when:
//   1.  this option is used without -fsycl or -fsycl-device-only
//   2.  this option is used with -fsycl-host-compiler=
// RUN: not %clang -### -fsycl-use-builtins-for-integration %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-NO-FSYCL %s
// CHK-NO-FSYCL: error: '-fsycl-use-builtins-for-integration' should be used only in conjunction with a SYCL option such as '-fsycl' or '-fsycl-device-only'
// RUN: not %clang -### -fsycl -fsycl-use-builtins-for-integration -fsycl-host-compiler=g++  %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-FSYCL-CONFLICT %s
// CHK-FSYCL-CONFLICT: error: the option -fsycl-host-compiler= conflicts with -fsycl-use-builtins-for-integration
