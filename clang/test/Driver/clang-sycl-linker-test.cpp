// Tests the clang-sycl-linker tool.
//
// REQUIRES: spirv-registered-target
//
// Test the dry run of a simple case to link two input files.
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_1.bc
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-spirv.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SIMPLE-FO
// SIMPLE-FO: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SIMPLE-FO-NEXT: "{{.*}}sycl-post-link{{.*}}" {{.*}} [[LLVMLINKOUT]].bc
// SIMPLE-FO-NEXT: LLVM backend: input: {{.*}}.bc, output: {{.*}}_0.spv
//
// Test that IMG_SPIRV image kind is set for non-AOT compilation.
// RUN: llvm-objdump --offloading %t-spirv.out | FileCheck %s --check-prefix=IMAGE-KIND-SPIRV
// IMAGE-KIND-SPIRV: kind            spir-v
//
// Test the dry run of a simple case with device library files specified.
// RUN: mkdir -p %t.dir
// RUN: touch %t.dir/lib1.bc
// RUN: touch %t.dir/lib2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBS
// DEVLIBS: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles: {{.*}}lib1.bc, {{.*}}lib2.bc  output: [[LLVMLINKOUT:.*]].bc
// DEVLIBS-NEXT: "{{.*}}sycl-post-link{{.*}}" {{.*}} [[LLVMLINKOUT]].bc
// DEVLIBS-NEXT: LLVM backend: input: {{.*}}.bc, output: a_0.spv
//
// Test a simple case with a random file (not bitcode) as input.
// RUN: touch %t.o
// RUN: not clang-sycl-linker -triple=spirv64 %t.o -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
// FILETYPEERROR: Unsupported file type
//
// Test to see if device library related errors are emitted.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs= -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR1
// DEVLIBSERR1: Number of device library files cannot be zero
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs=lib1.bc,lib2.bc,lib3.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR2
// DEVLIBSERR2: '{{.*}}lib3.bc' SYCL device library file is not found
//
// Test AOT compilation for an Intel GPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-aot-gpu.out 2>&1 \
// RUN:     --ocloc-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-GPU
// AOT-INTEL-GPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-GPU-NEXT: "{{.*}}sycl-post-link{{.*}}" {{.*}} [[LLVMLINKOUT]].bc
// AOT-INTEL-GPU-NEXT: LLVM backend: input: {{.*}}.bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-GPU-NEXT: "{{.*}}ocloc{{.*}}" {{.*}}-device bmg_g21 -a -b {{.*}}-output [[SPIRVTRANSLATIONOUT]]_0.out -file [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test that IMG_Object image kind is set for AOT compilation (Intel GPU).
// RUN: llvm-objdump --offloading %t-aot-gpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
// IMAGE-KIND-OBJECT: kind            elf
//
// Test AOT compilation for an Intel CPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=graniterapids %t_1.bc %t_2.bc -o %t-aot-cpu.out 2>&1 \
// RUN:     --opencl-aot-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-CPU
// AOT-INTEL-CPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-CPU-NEXT: "{{.*}}sycl-post-link{{.*}}" {{.*}} [[LLVMLINKOUT]].bc
// AOT-INTEL-CPU-NEXT: LLVM backend: input: {{.*}}.bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-CPU-NEXT: "{{.*}}opencl-aot{{.*}}" {{.*}}--device=cpu -a -b {{.*}}-o [[SPIRVTRANSLATIONOUT]]_0.out [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test that IMG_Object image kind is set for AOT compilation (Intel CPU).
// RUN: llvm-objdump --offloading %t-aot-cpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
//
// Check that the output file must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOOUTPUT
// NOOUTPUT: Output file must be specified
//
// Check that the target triple must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOTARGET
// NOTARGET: Target triple must be specified
//
// ============================================================================
// Tests for sycl-post-link functionality
// ============================================================================
//
// Test that --use-sycl-post-link-tool (default) invokes external tool.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-tool.out 2>&1 \
// RUN:   --use-sycl-post-link-tool \
// RUN:   | FileCheck %s --check-prefix=USE-TOOL
// USE-TOOL: sycl-post-link{{.*}}.bc
//
// Test that --no-use-sycl-post-link-tool uses library API and requires --sycl-module-split-mode.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-lib.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=none \
// RUN:   | FileCheck %s --check-prefix=USE-LIB
// USE-LIB: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// USE-LIB-NOT: "{{.*}}sycl-post-link{{.*}}"
//
// Test that --sycl-module-split-mode cannot be used with --use-sycl-post-link-tool.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc -o %t-err.out 2>&1 \
// RUN:   --use-sycl-post-link-tool --sycl-module-split-mode=kernel \
// RUN:   | FileCheck %s --check-prefix=MODE-TOOL-ERROR
// MODE-TOOL-ERROR: --sycl-module-split-mode cannot be used with --use-sycl-post-link-tool
//
// Test --sycl-module-split-mode=kernel.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-perkernel.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=kernel \
// RUN:   | FileCheck %s --check-prefix=SPLIT-PER-KERNEL
// SPLIT-PER-KERNEL: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
//
// Test --sycl-module-split-mode=source.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-pertu.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=source \
// RUN:   | FileCheck %s --check-prefix=SPLIT-PER-SOURCE
// SPLIT-PER-SOURCE: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
//
// Test --sycl-module-split-mode=auto.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-auto.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=auto \
// RUN:   | FileCheck %s --check-prefix=SPLIT-AUTO
// SPLIT-AUTO: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
//
// Test --sycl-module-split-mode=none.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-none.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=none \
// RUN:   | FileCheck %s --check-prefix=SPLIT-NONE
// SPLIT-NONE: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
//
// Test invalid split mode.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc -o %t-invalid.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=invalid \
// RUN:   | FileCheck %s --check-prefix=INVALID-SPLIT-MODE
// INVALID-SPLIT-MODE: Invalid split mode: invalid
//
// Test --sycl-device-code-split-esimd option is passed to tool mode.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-esimd.out 2>&1 \
// RUN:   --sycl-device-code-split-esimd \
// RUN:   | FileCheck %s --check-prefix=ESIMD-SPLIT
// ESIMD-SPLIT: sycl-post-link{{.*}}-split-esimd{{.*}}
//
// Test --no-sycl-device-code-split-esimd option is passed to tool mode.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-no-esimd.out 2>&1 \
// RUN:   --no-sycl-device-code-split-esimd \
// RUN:   | FileCheck %s --check-prefix=NO-ESIMD-SPLIT
// NO-ESIMD-SPLIT: sycl-post-link
// NO-ESIMD-SPLIT-NOT: -split-esimd
//
// Test --sycl-add-default-spec-consts-image for AOT target.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-spec-consts.out 2>&1 \
// RUN:   --sycl-add-default-spec-consts-image \
// RUN:   | FileCheck %s --check-prefix=SPEC-CONSTS-AOT
// SPEC-CONSTS-AOT: sycl-post-link{{.*}}-emit-only-kernels-as-entry-points{{.*}}
//
// Test --sycl-remove-unused-external-funcs for Intel GPU target.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-remove-funcs.out 2>&1 \
// RUN:   --sycl-remove-unused-external-funcs \
// RUN:   | FileCheck %s --check-prefix=REMOVE-FUNCS
// REMOVE-FUNCS: sycl-post-link{{.*}}-emit-only-kernels-as-entry-points{{.*}}
//
// Test --no-sycl-remove-unused-external-funcs keeps functions.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-keep-funcs.out 2>&1 \
// RUN:   --no-sycl-remove-unused-external-funcs \
// RUN:   | FileCheck %s --check-prefix=KEEP-FUNCS
// KEEP-FUNCS: sycl-post-link
// KEEP-FUNCS-NOT: -emit-only-kernels-as-entry-points
//
// Test --sycl-post-link-options passes custom options to tool.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-custom.out 2>&1 \
// RUN:   --sycl-post-link-options="--custom-opt1 --custom-opt2" \
// RUN:   | FileCheck %s --check-prefix=CUSTOM-OPTS
// CUSTOM-OPTS: sycl-post-link{{.*}}--custom-opt1 --custom-opt2{{.*}}
//
// Test triple-specific settings for SPIR-V target (native mode for spec constants).
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-spirv-native.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPIRV-NATIVE
// SPIRV-NATIVE: sycl-post-link{{.*}}-spec-const=native{{.*}}
//
// Test --no-sycl-add-default-spec-consts-image (default behavior).
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-no-spec-consts.out 2>&1 \
// RUN:   --no-sycl-add-default-spec-consts-image \
// RUN:   | FileCheck %s --check-prefix=NO-SPEC-CONSTS
// NO-SPEC-CONSTS: sycl-post-link
// NO-SPEC-CONSTS-NOT: -emit-default-spec-consts
//
// Test --sycl-allow-device-image-dependencies option.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-img-deps.out 2>&1 \
// RUN:   --sycl-allow-device-image-dependencies \
// RUN:   | FileCheck %s --check-prefix=IMAGE-DEPS
// IMAGE-DEPS: sycl-post-link
// IMAGE-DEPS-NOT: -emit-only-kernels-as-entry-points
//
// Test --sycl-thin-lto option suppresses -symbols flag (but not -emit-*-symbols).
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-thin-lto.out 2>&1 \
// RUN:   --sycl-thin-lto \
// RUN:   | FileCheck %s --check-prefix=THIN-LTO
// THIN-LTO: sycl-post-link{{.*}}-emit-exported-symbols
// THIN-LTO-NOT: {{[[:space:]]}}-symbols{{[[:space:]]}}
//
// Test --syclbin option enables kernel name emission.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-syclbin.out 2>&1 \
// RUN:   --syclbin=executable \
// RUN:   | FileCheck %s --check-prefix=SYCLBIN
// SYCLBIN: sycl-post-link{{.*}}-emit-kernel-names{{.*}}
//
// Test --sycl-device-library-location option is passed to tool.
// RUN: mkdir -p %t-devlib.dir
// RUN: touch %t-devlib.dir/libsycl-native-bfloat16.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-devlib.out 2>&1 \
// RUN:   --sycl-device-library-location=%t-devlib.dir \
// RUN:   | FileCheck %s --check-prefix=DEVLIB-LOC
// DEVLIB-LOC: sycl-post-link{{.*}}--device-lib-dir={{.*}}devlib.dir{{.*}}
//
// ============================================================================
// Tests for sycl-post-link function behavior
// ============================================================================
//
// Test getSYCLPostLinkSettings() - verify triple-specific configuration.
// For Intel GPU (spirv64 with arch), should emit param info and entry point optimization.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-intel-gpu-settings.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INTEL-GPU-SETTINGS
// INTEL-GPU-SETTINGS: sycl-post-link{{.*}}-spec-const=native
// INTEL-GPU-SETTINGS: -emit-only-kernels-as-entry-points
// INTEL-GPU-SETTINGS: -emit-param-info
//
// Test getSYCLPostLinkSettings() - verify ESIMD is split by default for SPIR-V.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-esimd-default.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ESIMD-DEFAULT
// ESIMD-DEFAULT: sycl-post-link{{.*}}-split-esimd{{.*}}-lower-esimd
//
// Test getTripleBasedSYCLPostLinkOpts() - verify it generates correct command-line args.
// Should include properties, symbols, exported/imported symbols for all targets.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-cmdline-args.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CMDLINE-ARGS
// CMDLINE-ARGS: sycl-post-link{{.*}}-properties{{.*}}-symbols{{.*}}-emit-exported-symbols{{.*}}-emit-imported-symbols
//
// Test runSYCLPostLinkTool() - verify tool is invoked with input/output files.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-tool-invoke.out 2>&1 \
// RUN:   --use-sycl-post-link-tool \
// RUN:   | FileCheck %s --check-prefix=TOOL-INVOKE
// TOOL-INVOKE: sycl-post-link{{.*}}-o {{.*}}.table {{.*}}.bc
//
// Test runSYCLPostLinkLibrary() - verify library mode doesn't print sycl-post-link command.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-lib-invoke.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=none \
// RUN:   | FileCheck %s --check-prefix=LIB-INVOKE
// LIB-INVOKE: sycl-device-link:
// LIB-INVOKE-NOT: "sycl-post-link"
// LIB-INVOKE: LLVM backend:
//
// Test that library mode with split mode produces split modules.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-lib-split.out 2>&1 \
// RUN:   --no-use-sycl-post-link-tool --sycl-module-split-mode=kernel \
// RUN:   | FileCheck %s --check-prefix=LIB-SPLIT
// LIB-SPLIT: sycl-device-link:
// LIB-SPLIT: LLVM backend:
//
// Test combination of multiple options together.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-combo.out 2>&1 \
// RUN:   --sycl-device-code-split-esimd --sycl-thin-lto \
// RUN:   --sycl-post-link-options="--debug" \
// RUN:   | FileCheck %s --check-prefix=COMBO
// COMBO: sycl-post-link{{.*}}-split-esimd{{.*}}--debug
// COMBO-NOT: {{[[:space:]]}}-symbols{{[[:space:]]}}
