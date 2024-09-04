/// Check that compiling for sycl device is disabled by default:
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-fsycl-is-device"

/// Check "-fsycl-is-device" is passed when compiling for device:
// RUN:   %clang -### -fsycl-device-only --no-offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-DEV %s
// CHECK-SYCL-DEV: "-fsycl-is-device"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl" "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include"

/// Check that "-Wno-sycl-strict" is set on compiler invocation with "-fsycl"
/// or "-fsycl-device-only" or both:
// RUN:   %clang -### -fsycl --no-offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// RUN:   %clang -### -fsycl-device-only --no-offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// RUN:   %clang -### -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NO_STRICT %s
// CHECK-SYCL-NO_STRICT: clang{{.*}} "-Wno-sycl-strict"

/// Check that -sycl-std=2017 is set if no std version is provided by user
// RUN:   %clang -### -fsycl --no-offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-STD_VERSION %s
// CHECK-SYCL-STD_VERSION: clang{{.*}} "-sycl-std=2020"

/// Check that -aux-triple is set correctly
// RUN:   %clang -### -fsycl --no-offload-new-driver -target aarch64-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-AUX-TRIPLE %s
// TODO: %clang -### -fsycl --no-offload-new-driver -fsycl-device-only -target aarch64-linux-gnu
// CHECK-SYCL-AUX-TRIPLE: clang{{.*}} "-aux-triple" "aarch64-unknown-linux-gnu"

/// Verify output files are properly specified given -o
// RUN: %clang -### -fsycl --no-offload-new-driver -fsycl-device-only -o dummy.out %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-OUTPUT-FILE %s
// RUN: %clang_cl -### -fsycl --no-offload-new-driver -fsycl-device-only -o dummy.out %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-OUTPUT-FILE %s
// CHECK-OUTPUT-FILE: clang{{.*}} "-o" "dummy.out"

/// -fsycl-device-only with preprocessing should only do the device compile
// RUN: %clang -ccc-print-phases -E -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// RUN: %clang_cl -ccc-print-phases -E -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// RUN: %clang_cl -ccc-print-phases -P -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// RUN: %clang_cl -ccc-print-phases -EP -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROCESS %s
// PHASES-PREPROCESS: 0: input, {{.*}}, c++, (device-sycl)
// PHASES-PREPROCESS: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// PHASES-PREPROCESS: 2: offload, "device-sycl (spir64-unknown-unknown)" {1}, c++-cpp-output

// RUN: %clang -ccc-print-phases -MM -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROC-DEPS %s
// RUN: %clang -ccc-print-phases -M -fsycl --no-offload-new-driver -fsycl-device-only %s 2>&1 \
// RUN:  | FileCheck -check-prefix=PHASES-PREPROC-DEPS %s
// PHASES-PREPROC-DEPS: 0: input, {{.*}}, c++, (device-sycl)
// PHASES-PROPROC-DEPS: 1: preprocessor, {0}, dependencies, (device-sycl)
// PHASES-PREPROC-DEPS: 2: offload, "device-sycl (spir64-unknown-unknown)" {1}, dependencies
