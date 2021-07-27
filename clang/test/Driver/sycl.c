// RUN: %clang -### -fsycl -c %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=1.2.1 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=121 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=2017 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=2020 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl -sycl-std=sycl-1.2.1 %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fno-sycl -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -sycl-std=2017 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clangxx -### -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### -fsycl -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clangxx -### -sycl-std=some-std %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang_cl -### -fsycl -sycl-std=2017 -- %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cl -### -fsycl -- %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clang_cl -### -sycl-std=some-std -- %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED

// ENABLED: "-cc1"{{.*}} "-fsycl-is-device"
// ENABLED-SAME: "-sycl-std={{[-.sycl0-9]+}}"
// ENABLED-SAME: "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"

// NOT_ENABLED-NOT: "-fsycl-is-device"
// NOT_ENABLED-NOT: "-fsycl-std-layout-kernel-params"

// DISABLED-NOT: "-fsycl-is-device"
// DISABLED-NOT: "-sycl-std={{.*}}"
// DISABLED-NOT: "-fsycl-std-layout-kernel-params"

// RUN: %clang -### -fsycl-device-only -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -fsycl-device-only -S %s 2>&1 | FileCheck %s --check-prefix=TEXTUAL
// RUN: %clang -### -fsycl-device-only -fsycl  %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -fsycl-device-only -fno-sycl-use-bitcode -c %s 2>&1 | FileCheck %s --check-prefix=NO-BITCODE
// RUN: %clang -### -target spir64-unknown-linux-sycldevice -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: %clang -### -fsycl-device-only -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=COMBINED
// RUN: %clangxx -### -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cl -### -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clangxx -### -fsycl-device-only -fno-sycl-unnamed-lambda %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-LAMBDA
// RUN: %clang_cl -### -fsycl-device-only -fno-sycl-unnamed-lambda %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-LAMBDA

// DEFAULT: "-triple" "spir64-unknown-{{.*}}-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-sycl-std=2020"{{.*}} "-emit-llvm-bc"
// DEFAULT: "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// DEFAULT: "-internal-isystem" "{{.*lib.*clang.*include}}"
// DEFAULT: "-std=c++17"
// DEFAULT-NOT: "{{.*}}llvm-spirv"{{.*}}
// DEFAULT-NOT: "-std=c++11"
// DEFAULT-NOT: "-std=c++14"
// NO-BITCODE: "-triple" "spir64-unknown-{{.*}}-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// NO-BITCODE: "{{.*}}llvm-spirv"{{.*}}
// TARGET: "-triple" "spir64-unknown-linux-sycldevice"{{.*}} "-emit-llvm-bc"
// COMBINED: "-triple" "spir64-unknown-{{.*}}-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// TEXTUAL: "-triple" "spir64-unknown-{{.*}}-sycldevice{{.*}}" "-fsycl-is-device"{{.*}} "-emit-llvm"
// CHECK-NOT-LAMBDA: "-fno-sycl-unnamed-lambda"

/// -fsycl-device-only triple checks
// RUN: %clang -fsycl-device-only -target x86_64-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-64 %s
// RUN: %clang_cl -fsycl-device-only --target=x86_64-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-64 %s
// DEVICE-64: clang{{.*}} "-triple" "spir64-unknown-unknown-sycldevice" "-aux-triple" "x86_64-unknown-linux-gnu"

// RUN: %clang -fsycl-device-only -target i386-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-32 %s
// RUN: %clang_cl -fsycl-device-only --target=i386-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-32 %s
// DEVICE-32: clang{{.*}} "-triple" "spir-unknown-unknown-sycldevice" "-aux-triple" "i386-unknown-linux-gnu"

/// Verify that the sycl header directory is before /usr/include
// RUN: %clangxx -### -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=HEADER_ORDER
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=HEADER_ORDER
// HEADER_ORDER-NOT: clang{{.*}} "/usr/include"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}

/// Verify -fsycl-device-only phases
// RUN: %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES
// DEFAULT-PHASES: 0: input, "{{.*}}", c++, (device-sycl)
// DEFAULT-PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// DEFAULT-PHASES: 2: compiler, {1}, ir, (device-sycl)
// DEFAULT-PHASES: 3: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {2}, ir
// DEFAULT-PHASES-NOT: linker

// -fsycl-help tests
// RUN: mkdir -p %t-sycl-dir
// RUN: touch %t-sycl-dir/aoc
// RUN: chmod +x %t-sycl-dir/aoc
// Test with a bad argument is expected to fail
// RUN: not %clang -fsycl-help=foo %s 2>&1 | FileCheck %s --check-prefix=SYCL-HELP-BADARG
// RUN: %clang -### -fsycl-help=gen %s 2>&1 | FileCheck %s --check-prefix=SYCL-HELP-GEN
// RUN: env PATH=%t-sycl-dir %clang -### -fsycl-help=fpga %s 2>&1 | FileCheck %s --check-prefixes=SYCL-HELP-FPGA,SYCL-HELP-FPGA-OUT -DDIR=%t-sycl-dir
// RUN: %clang -### -fsycl-help=x86_64 %s 2>&1 | FileCheck %s --check-prefix=SYCL-HELP-CPU
// RUN: %clang -### -fsycl-help %s 2>&1 | FileCheck %s --check-prefixes=SYCL-HELP-GEN,SYCL-HELP-FPGA,SYCL-HELP-CPU
// SYCL-HELP-BADARG: unsupported argument 'foo' to option 'fsycl-help='
// SYCL-HELP-GEN: Emitting help information for ocloc
// SYCL-HELP-GEN: Use triple of 'spir64_gen-unknown-unknown-sycldevice' to enable ahead of time compilation
// SYCL-HELP-FPGA-OUT: "[[DIR]]{{[/\\]+}}aoc" "-help" "-sycl"
// SYCL-HELP-FPGA: Emitting help information for aoc
// SYCL-HELP-FPGA: Use triple of 'spir64_fpga-unknown-unknown-sycldevice' to enable ahead of time compilation
// SYCL-HELP-CPU: Emitting help information for opencl-aot
// SYCL-HELP-CPU: Use triple of 'spir64_x86_64-unknown-unknown-sycldevice' to enable ahead of time compilation

// -fsycl-id-queries-fit-in-int
// RUN: %clang -### -fsycl -fsycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=ID_QUERIES
// RUN: %clang_cl -### -fsycl -fsycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=ID_QUERIES
// RUN: %clang -### -fsycl-device-only -fsycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=ID_QUERIES
// RUN: %clang_cl -### -fsycl-device-only -fsycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=ID_QUERIES
// RUN: %clang -### -fsycl -fno-sycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=NO_ID_QUERIES
// RUN: %clang_cl -### -fsycl -fno-sycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=NO_ID_QUERIES
// RUN: %clang -### -fsycl-device-only -fno-sycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=NO_ID_QUERIES
// RUN: %clang_cl -### -fsycl-device-only -fno-sycl-id-queries-fit-in-int  %s 2>&1 | FileCheck %s --check-prefix=NO_ID_QUERIES
// ID_QUERIES: "-fsycl-id-queries-fit-in-int"
// NO_ID_QUERIES: "-fno-sycl-id-queries-fit-in-int"

// RUN: %clang -### -fsycl  %s 2>&1 | FileCheck %s --check-prefix=DEFAULT_STD
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=DEFAULT_STD
// RUN: %clang_cl -### -fsycl -- %s 2>&1 | FileCheck %s --check-prefix=DEFAULT_STD

// DEFAULT_STD: "-sycl-std=2020"
