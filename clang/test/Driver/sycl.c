// RUN: %clang -### -fsycl -c %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### -fno-sycl -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang -### %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clangxx -### -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### -fsycl -fno-sycl %s 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: %clangxx -### %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
// RUN: %clangxx -### -sycl-std=some-std %s 2>&1 | FileCheck %s --check-prefix=NOT_ENABLED
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

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl-device-only -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir64
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir64
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl-device-only -S %s 2>&1 | FileCheck %s --check-prefix=TEXTUAL -DSPIRARCH=spir64
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl-device-only -fsycl  %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir64
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl-device-only -fsycl-device-obj=spirv -c %s 2>&1 | FileCheck %s --check-prefix=NO-BITCODE -DSPIRARCH=spir64
// RUN: %clang -### -target spir64-unknown-linux -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=TARGET -DSPIRARCH=spir64
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl-device-only -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=COMBINED -DSPIRARCH=spir64
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir64
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir64

/// 32-bit target checks
// RUN: %clang -### -target i386-unknown-linux-gnu -fsycl-device-only -c %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir
// RUN: %clang -### -target i386-unknown-linux-gnu -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir
// RUN: %clang -### -target i386-unknown-linux-gnu -fsycl-device-only -S %s 2>&1 | FileCheck %s --check-prefix=TEXTUAL -DSPIRARCH=spir
// RUN: %clang -### -target i386-unknown-linux-gnu -fsycl-device-only -fsycl  %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir
// RUN: %clang -### -target i386-unknown-linux-gnu -fsycl-device-only -fsycl-device-obj=spirv -c %s 2>&1 | FileCheck %s --check-prefix=NO-BITCODE -DSPIRARCH=spir
// RUN: %clang -### -target spir-unknown-linux -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=TARGET -DSPIRARCH=spir
// RUN: %clang -### -target i386-unknown-linux-gnu -fsycl-device-only -c -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=COMBINED -DSPIRARCH=spir
// RUN: %clangxx -### -target i386-unknown-linux-gnu -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir
// RUN: %clang_cl -### --target=i386-pc-windows-msvc -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT -DSPIRARCH=spir

// DEFAULT: "-triple" "[[SPIRARCH]]-unknown-{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-sycl-std=2020"{{.*}} "-emit-llvm-bc"
// DEFAULT: "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"
// DEFAULT: "-internal-isystem" "{{.*lib.*clang.*include}}"
// DEFAULT: "-std=c++17"
// DEFAULT-NOT: "{{.*}}llvm-spirv"{{.*}}
// DEFAULT-NOT: "-std=c++11"
// DEFAULT-NOT: "-std=c++14"
// NO-BITCODE: "-triple" "[[SPIRARCH]]-unknown-{{.*}}"{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// NO-BITCODE: "{{.*}}llvm-spirv"{{.*}}
// TARGET: "-triple" "[[SPIRARCH]]-unknown-linux"{{.*}} "-emit-llvm-bc"
// COMBINED: "-triple" "[[SPIRARCH]]-unknown-{{.*}}" "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// TEXTUAL: "-triple" "[[SPIRARCH]]-unknown-{{.*}}" "-fsycl-is-device"{{.*}} "-emit-llvm"

// RUN: %clangxx -### -fsycl-device-only -fno-sycl-unnamed-lambda %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-LAMBDA
// RUN: %clang_cl -### -fsycl-device-only -fno-sycl-unnamed-lambda %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-LAMBDA
// CHECK-NOT-LAMBDA: "-fno-sycl-unnamed-lambda"

// -fsycl-force-inline-kernel-lambda
// RUN: %clangxx -### -fsycl-device-only -fno-sycl-force-inline-kernel-lambda  %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INLINE
// RUN: %clang_cl -### -fsycl-device-only -fno-sycl-force-inline-kernel-lambda  %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INLINE
// RUN: %clangxx -### -fsycl-device-only -O0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INLINE
// RUN: %clang_cl -### -fsycl-device-only -Od %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INLINE
// RUN: %clangxx -### -fsycl-device-only -O1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-INLINE
// RUN: %clang_cl -### -fsycl-device-only -O2 %s 2>&1 | FileCheck %s --check-prefix=CHECK-INLINE
// RUN: %clangxx -### -fsycl-device-only -fintelfpga %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-INLINE
// CHECK-NOT-INLINE: "-fno-sycl-force-inline-kernel-lambda"
// CHECK-INLINE-NOT: "-fno-sycl-force-inline-kernel-lambda"

/// -fsycl-device-only triple checks
// RUN: %clang -fsycl-device-only -target x86_64-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-64 %s
// RUN: %clang_cl -fsycl-device-only --target=x86_64-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-64 %s
// DEVICE-64: clang{{.*}} "-triple" "spir64-unknown-unknown" "-aux-triple" "x86_64-unknown-linux-gnu"

// RUN: %clang -fsycl-device-only -target i386-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-32 %s
// RUN: %clang_cl -fsycl-device-only --target=i386-unknown-linux-gnu -### %s 2>&1 \
// RUN:  | FileCheck --check-prefix=DEVICE-32 %s
// DEVICE-32: clang{{.*}} "-triple" "spir-unknown-unknown" "-aux-triple" "i386-unknown-linux-gnu"

/// Verify that the sycl header directory is before /usr/include
// RUN: %clangxx -### -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=HEADER_ORDER
// RUN: %clangxx -### -fsycl %s 2>&1 | FileCheck %s --check-prefix=HEADER_ORDER
// HEADER_ORDER-NOT: clang{{.*}} "/usr/include"{{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}

/// Verify -fsycl-device-only phases
// RUN: %clang -### -ccc-print-phases -target x86_64-unknown-linux-gnu -fsycl-device-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT-PHASES
// DEFAULT-PHASES: 0: input, "{{.*}}", c++, (device-sycl)
// DEFAULT-PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// DEFAULT-PHASES: 2: compiler, {1}, ir, (device-sycl)
// DEFAULT-PHASES: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
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
// SYCL-HELP-BADARG: unsupported argument 'foo' to option '-fsycl-help='
// SYCL-HELP-GEN: Emitting help information for ocloc
// SYCL-HELP-GEN: Use triple of 'spir64_gen-unknown-unknown' to enable ahead of time compilation
// SYCL-HELP-FPGA: Emitting help information for aoc
// SYCL-HELP-FPGA: Use triple of 'spir64_fpga-unknown-unknown' to enable ahead of time compilation
// SYCL-HELP-FPGA-OUT: "[[DIR]]{{[/\\]+}}aoc" "-help" "-sycl"
// SYCL-HELP-CPU: Emitting help information for opencl-aot
// SYCL-HELP-CPU: Use triple of 'spir64_x86_64-unknown-unknown' to enable ahead of time compilation

// -fsycl-help redirect to file should retain proper information ordering
// RUN: %clang -### -fsycl-help %s > %t.help-out 2>&1
// RUN: FileCheck %s -check-prefix SYCL_HELP_ORDER --input-file=%t.help-out
// SYCL_HELP_ORDER: Emitting help information for ocloc
// SYCL_HELP_ORDER: ocloc{{(\.exe)?}}" "--help"
// SYCL_HELP_ORDER: Emitting help information for aoc
// SYCL_HELP_ORDER: aoc{{(\.exe)?}}" "-help" "-sycl"
// SYCL_HELP_ORDER: Emitting help information for opencl-aot
// SYCL_HELP_ORDER: opencl-aot{{(\.exe)?}}" "--help"

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

/// Verify correct match of offload arch with multiple sycl targets
// RUN: %clang -fsycl -fsycl-targets=nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx908 -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_86 -c -ccc-print-phases %s 2>&1 | FileCheck %s --check-prefix=MULTIPLE_TARGETS
// MULTIPLE_TARGETS: offload, "device-sycl (nvptx64-nvidia-cuda:sm_86)"
// MULTIPLE_TARGETS: offload, "device-sycl (amdgcn-amd-amdhsa:gfx908)"

/// Verify debug format for spir target.
// RUN: %clang -### -target x86_64-windows-msvc -fsycl -g -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG-WIN %s
// RUN: %clang_cl -### -fsycl -Zi -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEBUG-WIN %s
// DEBUG-WIN: {{.*}}"-fsycl-is-device"{{.*}}"-gcodeview"
// DEBUG-WIN: {{.*}}"-fsycl-is-host"{{.*}}"-gcodeview"
// DEBUG-WIN-NOT: dwarf-version
