/// Check "-aux-target-cpu" and "target-cpu" are passed when compiling for SYCL offload device and host codes:
//  RUN:  %clang -### -fsycl --offload-new-driver -c %s 2>&1 | FileCheck -check-prefix=CHECK-OFFLOAD %s
//  CHECK-OFFLOAD: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  CHECK-OFFLOAD-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]"
//  CHECK-OFFLOAD: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-host"
//  CHECK-OFFLOAD-SAME: "-target-cpu" "[[HOST_CPU_NAME]]"

/// Check "-aux-target-cpu" with "-aux-target-feature" and "-target-cpu" with "-target-feature" are passed
/// when compiling for SYCL offload device and host codes:
//  RUN:  %clang -fsycl --offload-new-driver -mavx -c %s -### -o %t.o 2>&1 | FileCheck -check-prefix=OFFLOAD-AVX %s
//  OFFLOAD-AVX: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  OFFLOAD-AVX-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]" "-aux-target-feature" "+avx"
//  OFFLOAD-AVX: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-host"
//  OFFLOAD-AVX-SAME: "-target-cpu" "[[HOST_CPU_NAME]]" "-target-feature" "+avx"

/// Check that the needed -fsycl --offload-new-driver -fsycl-is-device and -fsycl-is-host options
/// are passed to all of the needed compilation steps regardless of final
/// phase.
// RUN:  %clang -### -fsycl --offload-new-driver -c %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl --offload-new-driver -E %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl --offload-new-driver -S %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-host"

/// Check that -fcoverage-mapping is disabled for device
// RUN: %clang -### -fsycl --offload-new-driver -fprofile-instr-generate -fcoverage-mapping -target x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_COVERAGE_MAPPING %s
// CHECK_COVERAGE_MAPPING: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-is-device"{{.*}}
// CHECK_COVERAGE_MAPPING-NOT: "-fprofile-instrument=clang"
// CHECK_COVERAGE_MAPPING-NOT: "-fcoverage-mapping"
// CHECK_COVERAGE_MAPPING: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-host"{{.*}} "-fprofile-instrument=clang"{{.*}} "-fcoverage-mapping"{{.*}}

/// Check that -fprofile-arcs -ftest-coverage is disabled for device
// RUN: %clang -### -fsycl --offload-new-driver -fprofile-arcs -ftest-coverage -target x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_TEST_COVERAGE %s
// CHECK_TEST_COVERAGE: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-is-device"{{.*}}
// CHECK_TEST_COVERAGE-NOT: "-coverage-notes-file={{.*}}"
// CHECK_TEST_COVERAGE: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-host"{{.*}} "-coverage-notes-file={{.*}}"

/// -S -emit-llvm should generate IR for device.
// RUN: %clangxx -### -fsycl --offload-new-driver -S -emit-llvm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_S_LLVM %s
// CHECK_S_LLVM: clang{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm-bc"
// CHECK_S_LLVM: clang-offload-packager{{.*}}
// CHECK_S_LLVM: clang{{.*}} "-fsycl-is-host"{{.*}} "-emit-llvm"

// RUN:  touch %t_empty.o
// RUN:  %clangxx -### -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu -fsycl-targets=spir64_x86_64 %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix DEFAULT_LINK %s
// RUN:  %clangxx -### -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu -fsycl-targets=spir64_fpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix DEFAULT_LINK %s
// RUN:  %clangxx -### -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu -fsycl-targets=spir64_gen %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix DEFAULT_LINK %s
// DEFAULT_LINK: clang-linker-wrapper{{.*}}

/// Passing in the default triple should allow for -Xsycl-target options, both the
/// "=<triple>" and the default spelling
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64 -Xsycl-target-backend=spir64 -DFOO -Xsycl-target-linker=spir64 -DFOO2 %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -Xsycl-target-backend=spir64 -DFOO -Xsycl-target-linker=spir64 -DFOO2 %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT %s
// SYCL_TARGET_OPT: clang-linker-wrapper{{.*}} "--sycl-backend-compile-options=-DFOO" "--sycl-target-link-options=-DFOO2"

// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64_x86_64 -Xsycl-target-backend -DFOO %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT_AOT,SYCL_TARGET_OPT_CPU %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64_gen -Xsycl-target-backend -DFOO %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT_AOT,SYCL_TARGET_OPT_GPU %s
// SYCL_TARGET_OPT_AOT-NOT: error: cannot deduce implicit triple value for '-Xsycl-target-backend'
// SYCL_TARGET_OPT_CPU: clang-linker-wrapper{{.*}} "--cpu-tool-arg=-DFOO"
// SYCL_TARGET_OPT_GPU: clang-linker-wrapper{{.*}} "--gpu-tool-arg=-DFOO"

/// Check -fsycl-targets=spir64 enables addition of -ffine-grained-bitfield-accesses option
// RUN:   %clangxx -### -fsycl-device-only --offload-new-driver %s 2>&1 | FileCheck -check-prefixes=CHECK_BITFIELD_OPTION %s
// CHECK_BITFIELD_OPTION: clang{{.*}} "-ffine-grained-bitfield-accesses"

/// Using linker specific items at the end of the command should not fail when
/// we are performing a non-linking compilation behavior
// RUN: %clangxx -E -fsycl --offload-new-driver %S/Inputs/SYCL/liblin64.a \
// RUN:          -target x86_64-unknown-linux-gnu -### 2>&1 \
// RUN:  | FileCheck -check-prefix IGNORE_INPUT %s
// RUN: %clangxx -c -fsycl --offload-new-driver %S/Inputs/SYCL/liblin64.a \
// RUN:          -target x86_64-unknown-linux-gnu -### 2>&1 \
// RUN:  | FileCheck -check-prefix IGNORE_INPUT %s
// IGNORE_INPUT: input unused

/// Check if the clang with fsycl adds C++ libraries to the link line
//  RUN:  %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHECK-FSYCL-WITH-CLANG %s
// CHECK-FSYCL-WITH-CLANG: "-lstdc++"

/// Check for correct handling of -fsycl-fp64-conv-emu option for different targets
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64 -fsycl-fp64-conv-emu %s 2>&1 | FileCheck -check-prefix=CHECK_WARNING %s
// CHECK_WARNING: warning: '-fsycl-fp64-conv-emu' option is supported only for AOT compilation of Intel GPUs. It will be ignored for other targets [-Wunused-command-line-argument]

// RUN: %clang -### -Wno-unused-command-line-argument -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=spir64 -fsycl-fp64-conv-emu %s 2>&1 | FileCheck -check-prefix=CHECK_NO_WARNING %s
// CHECK_NO_WARNING-NOT: warning: '-fsycl-fp64-conv-emu' option is supported only for AOT compilation of Intel GPUs. It will be ignored for other targets [-Wunused-command-line-argument]

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -fsycl-targets=intel_gpu_pvc -fsycl-fp64-conv-emu %s 2>&1 | FileCheck -check-prefix=CHECK_FSYCL_FP64_CONV_EMU %s
// CHECK_FSYCL_FP64_CONV_EMU-NOT: clang{{.*}} "-cc1" "-triple x86_64-unknown-linux-gnu" {{.*}} "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU-DAG: clang{{.*}} "-cc1" "-triple" "spir64_gen{{.*}}" "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU-DAG: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=pvc,kind=sycl,compile-opts={{.*}}-options -ze-fp64-gen-conv-emu{{.*}}"
// RUN: %clang_cl -fsycl --offload-new-driver -fsycl-targets=spir64_gen-unknown-unknown %s -fsycl-fp64-conv-emu -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK_FSYCL_FP64_CONV_EMU_WIN
// CHECK_FSYCL_FP64_CONV_EMU_WIN-NOT: clang{{.*}} "-cc1" "-triple x86_64-unknown-linux-gnu" {{.*}} "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU_WIN-DAG: clang{{.*}} "-cc1" "-triple" "spir64_gen{{.*}}" "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU_WIN-DAG: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=,kind=sycl,compile-opts={{.*}}-options -ze-fp64-gen-conv-emu{{.*}}"
