/// Check that compiling for sycl device is disabled by default:
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fsycl-is-device"

/// Check "-fsycl-is-device" is passed when compiling for device, including when --config is used:
// RUN:   %clang -### -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-DEV %s
// RUN:   %clang -### --config=%S/Inputs/empty.cfg -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-DEV %s
// CHECK-SYCL-DEV: "-fsycl-is-device"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers" "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"

/// Check that "-Wno-sycl-strict" is set on compiler invocation with "-fsycl"
/// or "-fsycl-device-only" or both:
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// RUN:   %clang -### -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// CHECK-SYCL-NO_STRICT: clang{{.*}} "-Wno-sycl-strict"

/// Check that -sycl-std=2020 is set if no std version is provided by user
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-STD_VERSION %s
// CHECK-SYCL-STD_VERSION: clang{{.*}} "-sycl-std=2020"

/// Check that -aux-triple is set correctly
// RUN:   %clang -### -fsycl -target aarch64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-AUX-TRIPLE %s
// TODO: %clang -### -fsycl -fsycl-device-only -target aarch64-linux-gnu
// CHECK-SYCL-AUX-TRIPLE: clang{{.*}} "-aux-triple" "aarch64-unknown-linux-gnu"

/// Verify output files are properly specified given -o
// RUN: %clang -### -fsycl -fsycl-device-only -o dummy.out %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-OUTPUT-FILE %s
// RUN: %clang_cl -### -fsycl -fsycl-device-only -o dummy.out %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-OUTPUT-FILE %s
// CHECK-OUTPUT-FILE: clang{{.*}} "-o" "dummy.out"

/// -fsycl-device-only with preprocessing should only do the device compile
// RUN: %clang -ccc-print-phases -E -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// RUN: %clang_cl -ccc-print-phases -E -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// RUN: %clang_cl -ccc-print-phases -P -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// RUN: %clang_cl -ccc-print-phases -EP -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// PHASES-PREPROCESS: 0: input, {{.*}}, c++, (device-sycl)
// PHASES-PREPROCESS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// PHASES-PREPROCESS: 2: offload, "device-sycl (spir64-unknown-unknown)" {1}, none

// RUN: %clang -ccc-print-phases -MM -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROC-DEPS %s
// RUN: %clang -ccc-print-phases -M -fsycl --offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROC-DEPS %s
// PHASES-PREPROC-DEPS: 0: input, {{.*}}, c++, (device-sycl)
// PHASES-PROPROC-DEPS: 1: preprocessor, {0}, dependencies, (device-sycl)
// PHASES-PREPROC-DEPS: 2: offload, "device-sycl (spir64-unknown-unknown)" {1}, none

/// Check that "-fno-offload-use-alloca-addrspace-for-srets" is not set by
/// default on the command-line in a non-sycl compilation.
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ALLOCA-ADDRSPACE %s
// CHECK-ALLOCA-ADDRSPACE-NOT: clang{{.*}} "-fno-offload-use-alloca-addrspace-for-srets"

/// Check that "-fno-offload-use-alloca-addrspace-for-srets" is set if it is
/// not specified on the command-line by the user with -fsycl
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-ALLOCA-ADDRSPACE %s
// CHECK-NO-ALLOCA-ADDRSPACE: clang{{.*}} "-fno-offload-use-alloca-addrspace-for-srets"
