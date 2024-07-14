/// Check "-aux-target-cpu" and "target-cpu" are passed when compiling for SYCL offload device and host codes:
//  RUN:  %clang -### -fsycl -c %s 2>&1 | FileCheck -check-prefix=CHECK-OFFLOAD %s
//  CHECK-OFFLOAD: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  CHECK-OFFLOAD-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]"
//  CHECK-OFFLOAD-NEXT: append-file{{.*}}
//  CHECK-OFFLOAD-NEXT: clang{{.*}} "-cc1" {{.*}}
//  CHECK-OFFLOAD-NEXT-SAME: "-fsycl-is-host"
//  CHECK-OFFLOAD-NEXT-SAME: "-target-cpu" "[[HOST_CPU_NAME]]"

/// Check "-aux-target-cpu" with "-aux-target-feature" and "-target-cpu" with "-target-feature" are passed
/// when compiling for SYCL offload device and host codes:
//  RUN:  %clang -fsycl -mavx -c %s -### -o %t.o 2>&1 | FileCheck -check-prefix=OFFLOAD-AVX %s
//  OFFLOAD-AVX: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
//  OFFLOAD-AVX-SAME: "-aux-target-cpu" "[[HOST_CPU_NAME:[^ ]+]]" "-aux-target-feature" "+avx"
//  OFFLOAD-AVX-NEXT: append-file{{.*}}
//  OFFLOAD-AVX-NEXT: clang{{.*}} "-cc1" {{.*}}
//  OFFLOAD-AVX-NEXT-SAME: "-fsycl-is-host"
//  OFFLOAD-AVX-NEXT-SAME: "-target-cpu" "[[HOST_CPU_NAME]]" "-target-feature" "+avx"

/// Check that the needed -fsycl -fsycl-is-device and -fsycl-is-host options
/// are passed to all of the needed compilation steps regardless of final
/// phase.
// RUN:  %clang -### -fsycl -c %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl -E %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl -S %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// RUN:  %clang -### -fsycl %s 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-device"
// CHECK-OPTS: clang{{.*}} "-cc1" {{.*}} "-fsycl-is-host"

/// Check that -fcoverage-mapping is disabled for device
// RUN: %clang -### -fsycl -fprofile-instr-generate -fcoverage-mapping -target x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_COVERAGE_MAPPING %s
// CHECK_COVERAGE_MAPPING: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-is-device"{{.*}}
// CHECK_COVERAGE_MAPPING-NOT: "-fprofile-instrument=clang"
// CHECK_COVERAGE_MAPPING-NOT: "-fcoverage-mapping"
// CHECK_COVERAGE_MAPPING: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-host"{{.*}} "-fprofile-instrument=clang"{{.*}} "-fcoverage-mapping"{{.*}}

/// Check that -fprofile-arcs -ftest-coverage is disabled for device
// RUN: %clang -### -fsycl -fprofile-arcs -ftest-coverage -target x86_64-unknown-linux-gnu -c %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_TEST_COVERAGE %s
// CHECK_TEST_COVERAGE: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-is-device"{{.*}}
// CHECK_TEST_COVERAGE-NOT: "-coverage-notes-file={{.*}}"
// CHECK_TEST_COVERAGE: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-host"{{.*}} "-coverage-notes-file={{.*}}"

/// check for PIC for device wrap compilation when using -shared or -fPIC
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -shared %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_SHARED %s
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -fPIC %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_SHARED %s
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -fPIE %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_SHARED %s
// CHECK_SHARED: llc{{.*}} "-relocation-model=pic"

/// check for code-model settings for llc device wrap compilation
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -mcmodel=large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_CODE_MODEL -DARG=large %s
// RUN: %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -mcmodel=medium %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_CODE_MODEL -DARG=medium %s
// CHECK_CODE_MODEL: llc{{.*}} "--code-model=[[ARG]]"

/// -S -emit-llvm should generate textual IR for device.
// RUN: %clangxx -### -fsycl -S -emit-llvm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK_S_LLVM %s
// CHECK_S_LLVM: clang{{.*}} "-fsycl-is-device"{{.*}} "-emit-llvm"{{.*}} "-o" "[[DEVICE:.+\.ll]]"
// CHECK_S_LLVM: clang{{.*}} "-fsycl-is-host"{{.*}} "-emit-llvm"{{.*}} "-o" "[[HOST:.+\.ll]]"
// CHECK_S_LLVM: clang-offload-bundler{{.*}} "-type=ll"{{.*}} "-input=[[DEVICE]]" "-input=[[HOST]]"

/// Check for default device triple compilations based on object, archive or
/// forced from command line.
// RUN:  touch %t_empty.o
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fintelfpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_OBJ %s
// IMPLIED_DEVICE_OBJ: clang-offload-bundler{{.*}} "-type=o"{{.*}} "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_{{.*}}-unknown-unknown,{{.*}}sycl-spir64-unknown-unknown"{{.*}} "-unbundle"

// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fintelfpga %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix IMPLIED_DEVICE_LIB %s
// IMPLIED_DEVICE_LIB: clang-offload-bundler{{.*}} "-type=aoo"{{.*}} "-targets=sycl-spir64_{{.*}}-unknown-unknown,sycl-spir64-unknown-unknown"{{.*}} "-unbundle"

/// Check that the default device triple is not used with -fno-sycl-link-spirv
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fsycl-targets=spir64_x86_64 %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_CPU %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fsycl-targets=spir64_fpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_FPGA %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-link-spirv -fsycl-targets=spir64_gen %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_GEN %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fno-sycl-link-spirv -fintelfpga %S/Inputs/SYCL/objlin64.o %s 2>&1 \
// RUN:    | FileCheck -check-prefixes=NO_IMPLIED_DEVICE_OPT,NO_IMPLIED_DEVICE_FPGA %s
// NO_IMPLIED_DEVICE_CPU: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown"
// NO_IMPLIED_DEVICE_FPGA: clang{{.*}} "-triple" "spir64_fpga-unknown-unknown"
// NO_IMPLIED_DEVICE_GEN: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// NO_IMPLIED_DEVICE_OPT-NOT: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown"{{.*}} "-check-section"
// NO_IMPLIED_DEVICE_OPT-NOT: clang-offload-bundler{{.*}} "-targets={{.*}}spir64-unknown-unknown{{.*}}" "-unbundle"

// RUN:  %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -fsycl-targets=spir64_x86_64 %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// RUN:  %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -fsycl-targets=spir64_fpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// RUN:  %clangxx -### -fsycl -target x86_64-unknown-linux-gnu -fsycl-targets=spir64_gen %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fintelfpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_IMPLIED_DEVICE %s
// NO_IMPLIED_DEVICE: clang-offload-bundler{{.*}} "-type=o" "-targets=sycl-spir64-unknown-unknown"{{.*}} "-check-section"
// NO_IMPLIED_DEVICE-NOT: clang-offload-bundler{{.*}} "-targets={{.*}}spir64-unknown-unknown{{.*}}" "-unbundle"

/// Passing in the default triple should allow for -Xsycl-target options, both the
/// "=<triple>" and the default spelling
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 -Xsycl-target-backend=spir64 -DFOO -Xsycl-target-linker=spir64 -DFOO2 %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -Xsycl-target-backend=spir64 -DFOO -Xsycl-target-linker=spir64 -DFOO2 %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT %s
// SYCL_TARGET_OPT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-DFOO" "-link-opts=-DFOO2"
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_x86_64 -Xsycl-target-backend -DFOO %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT_AOT %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend -DFOO %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT_AOT %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga -Xsycl-target-backend -DFOO %S/Inputs/SYCL/objlin64.o 2>&1 \
// RUN:    | FileCheck -check-prefixes=SYCL_TARGET_OPT_AOT %s
// SYCL_TARGET_OPT_AOT-NOT: error: cannot deduce implicit triple value for '-Xsycl-target-backend'
// SYCL_TARGET_OPT_AOT: {{opencl-aot|ocloc|aoc}}{{.*}} "-DFOO"

/// Do not process directories when checking for default sections in fat objs
// RUN:  %clangxx -### -Wl,-rpath,%S -fsycl -fsycl-targets=spir64_x86_64 %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_DIR_CHECK %s
// RUN:  %clangxx -### -Xlinker -rpath -Xlinker %S -fsycl -fsycl-targets=spir64_fpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_DIR_CHECK %s
// RUN:  %clangxx -### -Wl,-rpath,%S -fsycl -fsycl-targets=spir64_gen %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_DIR_CHECK %s
// RUN:  %clangxx -### -Wl,-rpath,%S -fintelfpga %t_empty.o %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_DIR_CHECK %s
// NO_DIR_CHECK-NOT: clang-offload-bundler: error: '{{.*}}': Is a directory

// Device section checking only occur when offloading is enabled
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu -fsycl %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix CHECK_SECTION %s
// RUN:  %clangxx -### -target x86_64-unknown-linux-gnu %S/Inputs/SYCL/liblin64.a %s 2>&1 \
// RUN:    | FileCheck -check-prefix NO_CHECK_SECTION %s
// CHECK_SECTION: {{(/|\\)}}clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown"{{.*}} "-check-section"
// CHECK_SECTION: {{(/|\\)}}clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown"{{.*}} "-check-section"
// CHECK_SECTION: {{(/|\\)}}clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr_emu-intel-unknown"{{.*}} "-check-section"
// NO_CHECK_SECTION-NOT: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr-intel-unknown"{{.*}} "-check-section"
// NO_CHECK_SECTION-NOT: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocx-intel-unknown"{{.*}} "-check-section"
// NO_CHECK_SECTION-NOT: clang-offload-bundler{{.*}} "-type=ao" "-targets=sycl-fpga_aocr_emu-intel-unknown"{{.*}} "-check-section"

/// Check -fsycl-targets=spir64 enables addition of -ffine-grained-bitfield-accesses option
// RUN:   %clangxx -### -fsycl-device-only %s 2>&1 | FileCheck -check-prefixes=CHECK_BITFIELD_OPTION %s
// CHECK_BITFIELD_OPTION: clang{{.*}} "-ffine-grained-bitfield-accesses"

/// Using linker specific items at the end of the command should not fail when
/// we are performing a non-linking compilation behavior
// RUN: %clangxx -E -fsycl %S/Inputs/SYCL/liblin64.a \
// RUN:          -target x86_64-unknown-linux-gnu -### 2>&1 \
// RUN:  | FileCheck -check-prefix IGNORE_INPUT %s
// RUN: %clangxx -c -fsycl %S/Inputs/SYCL/liblin64.a \
// RUN:          -target x86_64-unknown-linux-gnu -### 2>&1 \
// RUN:  | FileCheck -check-prefix IGNORE_INPUT %s
// IGNORE_INPUT: input unused

/// Check if the clang with fsycl adds C++ libraries to the link line
//  RUN:  %clang -### -target x86_64-unknown-linux-gnu -fsycl %s 2>&1 | FileCheck -check-prefix=CHECK-FSYCL-WITH-CLANG %s
// CHECK-FSYCL-WITH-CLANG: "-lstdc++"

/// Check selective passing of -emit-only-kernels-as-entry-points to sycl-post-link tool
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga %s 2>&1 | FileCheck -check-prefix=CHECK_SYCL_POST_LINK_OPT_PASS %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen %s 2>&1 | FileCheck -check-prefix=CHECK_SYCL_POST_LINK_OPT_PASS %s
// CHECK_SYCL_POST_LINK_OPT_PASS: sycl-post-link{{.*}}emit-only-kernels-as-entry-points
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen -fno-sycl-remove-unused-external-funcs %s 2>&1 | FileCheck -check-prefix=CHECK_SYCL_POST_LINK_OPT_NO_PASS %s
// CHECK_SYCL_POST_LINK_OPT_NO_PASS-NOT: sycl-post-link{{.*}}emit-only-kernels-as-entry-points

/// Check selective passing of -support-dynamic-linking to sycl-post-link tool
// TODO: Enable when SYCL RT supports dynamic linking
// RUNx: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_fpga -shared %s 2>&1 | FileCheck -check-prefix=CHECK_SYCL_POST_LINK_SHARED_PASS %s
// RUNx: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen -shared %s 2>&1 | FileCheck -check-prefix=CHECK_SYCL_POST_LINK_SHARED_PASS %s
// CHECK_SYCL_POST_LINK_SHARED_PASS: sycl-post-link{{.*}}support-dynamic-linking
// RUNx: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64_gen %s 2>&1 | FileCheck -check-prefix=CHECK_SYCL_POST_LINK_SHARED_NO_PASS %s
// CHECK_SYCL_POST_LINK_SHARED_NO_PASS-NOT: sycl-post-link{{.*}}support-dynamic-linking

/// Check for correct handling of -fsycl-fp64-conv-emu option for different targets
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 -fsycl-fp64-conv-emu %s 2>&1 | FileCheck -check-prefix=CHECK_WARNING %s
// CHECK_WARNING: warning: '-fsycl-fp64-conv-emu' option is supported only for AOT compilation of Intel GPUs. It will be ignored for other targets [-Wunused-command-line-argument]
// RUN: %clang -### -Wno-unused-command-line-argument -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 -fsycl-fp64-conv-emu %s 2>&1 | FileCheck -check-prefix=CHECK_NO_WARNING %s
// CHECK_NO_WARNING-NOT: warning: '-fsycl-fp64-conv-emu' option is supported only for AOT compilation of Intel GPUs. It will be ignored for other targets [-Wunused-command-line-argument]
// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl -fsycl-targets=intel_gpu_pvc -fsycl-fp64-conv-emu %s 2>&1 | FileCheck -check-prefix=CHECK_FSYCL_FP64_CONV_EMU %s
// CHECK_FSYCL_FP64_CONV_EMU-NOT: clang{{.*}} "-cc1" "-triple x86_64-unknown-linux-gnu" {{.*}} "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU-DAG: clang{{.*}} "-cc1" "-triple" "spir64_gen{{.*}}" "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU-DAG: ocloc{{.*}} "-options" "-ze-fp64-gen-conv-emu"
// RUN: %clang_cl -fsycl -fsycl-targets=spir64_gen-unknown-unknown %s -fsycl-fp64-conv-emu -### 2>&1 \
// RUN:  | FileCheck %s -check-prefixes=CHECK_FSYCL_FP64_CONV_EMU_WIN
// CHECK_FSYCL_FP64_CONV_EMU_WIN-NOT: clang{{.*}} "-cc1" "-triple x86_64-unknown-linux-gnu" {{.*}} "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU_WIN-DAG: clang{{.*}} "-cc1" "-triple" "spir64_gen{{.*}}" "-fsycl-fp64-conv-emu"
// CHECK_FSYCL_FP64_CONV_EMU_WIN-DAG: ocloc{{.*}} "-options" "-ze-fp64-gen-conv-emu"
