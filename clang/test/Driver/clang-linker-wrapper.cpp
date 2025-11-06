/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t.o
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t.devicelib.cpp
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t.devicelib.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-CMDS-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-NEXT: llvm-spirv{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: {{.*}}
// CHK-CMDS-NEXT: clang{{.*}} -c -o [[LLCOUT:.*]] [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

// Check sycl-module-split-mode command line option.
// This option uses split library instead of sycl-post-link tool.
// RUN: clang-linker-wrapper -no-use-sycl-post-link-tool -sycl-module-split-mode=auto -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SPLIT-CMDS %s
// CHK-SPLIT-CMDS: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-SPLIT-CMDS-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-SPLIT-CMDS-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-SPLIT-CMDS-NEXT: sycl-module-split: input: [[SECONDLLVMLINKOUT]].bc, output: [[SYCLMODULESPLITOUT:.*]].bc
// CHK-SPLIT-CMDS-NEXT: llvm-spirv{{.*}} LLVM_SPIRV_OPTIONS -o [[SPIRVOUT:.*]].spv [[SYCLMODULESPLITOUT]].bc
// LLVM-SPIRV is not called in dry-run
// CHK-SPLIT-CMDS-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: [[SPIRVOUT]].spv
// CHK-SPLIT-CMDS-NEXT: clang{{.*}} -c -o [[LLCOUT:.*]] [[WRAPPEROUT]].bc
// CHK-SPLIT-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

// Check errors with -[no-]use-sycl-post-link-tool.
// RUN: not clang-linker-wrapper -sycl-module-split-mode=auto -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SYCL-POST-LINK-TOOL-ERROR %s
// CHK-SYCL-POST-LINK-TOOL-ERROR: error: -sycl-module-split-mode should be used with the -no-use-sycl-post-link-tool command line option.

// RUN: not clang-linker-wrapper -use-sycl-post-link-tool -no-use-sycl-post-link-tool -sycl-module-split-mode=auto -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SYCL-POST-LINK-TOOL-ERROR2 %s
// CHK-SYCL-POST-LINK-TOOL-ERROR2: error: -use-sycl-post-link-tool and -no-use-sycl-post-link-tool options can't be used together.

/// check for PIC for device wrap compilation when using -shared
// RUN: clang-linker-wrapper -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" -shared "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SHARED %s
// CHK-SHARED: clang{{.*}} -fPIC

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for Intel GPU)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=intel_gpu_pvc -c --offload-new-driver -o %t1.o
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t1.devicelib.cpp
// RUN: %clang %t1.devicelib.cpp -fsycl -fsycl-targets=intel_gpu_pvc -c --offload-new-driver -o %t1.devicelib.o
//
// Run clang-linker-wrapper test (with and without -sycl-embed-ir). -sycl-embed-ir should have no effect for Intel targets.
//
// RUN: clang-linker-wrapper -sycl-device-libraries=%t1.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t1.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-GEN %s
// RUN: clang-linker-wrapper -sycl-embed-ir -sycl-device-libraries=%t1.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t1.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-GEN %s
// CHK-CMDS-AOT-GEN: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-AOT-GEN-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-GEN-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-GEN-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: llvm-spirv{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: ocloc{{.*}} -output_no_suffix -spirv_input -device pvc{{.*}} -output {{.*}} -file {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: {{.*}}, compile-opts: , link-opts:
// CHK-CMDS-AOT-GEN-NEXT: clang{{.*}} -c -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for Intel CPU)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=spir64_x86_64 -c --offload-new-driver -o %t2.o
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t2.devicelib.cpp
// RUN: %clang %t2.devicelib.cpp -fsycl -fsycl-targets=spir64_x86_64 -c --offload-new-driver -o %t2.devicelib.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-libraries=%t2.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t2.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-CPU %s
// CHK-CMDS-AOT-CPU: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-AOT-CPU-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-CPU-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-CPU-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: llvm-spirv{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: opencl-aot{{.*}} --device=cpu -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: clang{{.*}} -c -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for NVPTX)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -c -nocudalib -fno-sycl-libspirv --offload-new-driver -o %t3.o
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t3.devicelib.cpp
// RUN: %clang %t3.devicelib.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -nocudalib -fno-sycl-libspirv -c --offload-new-driver -o %t3.devicelib.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-libraries=%t3.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t3.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-NV %s
// CHK-CMDS-AOT-NV: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-AOT-NV-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-NV-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-NV-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-NV-NEXT: clang{{.*}} -o [[CLANGOUT:.*]] -dumpdir a.out.nvptx64.sm_50.img. --target=nvptx64-nvidia-cuda -march={{.*}}
// CHK-CMDS-AOT-NV-NEXT: ptxas{{.*}} --output-file [[PTXASOUT:.*]] [[CLANGOUT]]
// CHK-CMDS-AOT-NV-NEXT: fatbinary{{.*}} --create [[FATBINOUT:.*]] --image=profile={{.*}},file=[[CLANGOUT]] --image=profile={{.*}},file=[[PTXASOUT]]
// CHK-CMDS-AOT-NV-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: [[FATBINOUT]]
// CHK-CMDS-AOT-NV-NEXT: clang{{.*}} -c -o [[LLCOUT:.*]] [[WRAPPEROUT]]
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for AMD)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx803 -fgpu-rdc -nogpulib -fno-sycl-libspirv -c --offload-new-driver -o %t4.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t4.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-AMD %s
// CHK-CMDS-AOT-AMD: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-AOT-AMD-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-AMD-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[FIRSTLLVMLINKOUT]].bc
// CHK-CMDS-AOT-AMD-NEXT: clang{{.*}} -o [[CLANGOUT:.*]] -dumpdir a.out.amdgcn.gfx803.img. --target=amdgcn-amd-amdhsa -mcpu={{.*}}
// CHK-CMDS-AOT-AMD-NEXT: clang-offload-bundler{{.*}} -input=[[CLANGOUT]] -output=[[BUNDLEROUT:.*]]
// CHK-CMDS-AOT-AMD-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: [[BUNDLEROUT]]
// CHK-CMDS-AOT-AMD-NEXT: clang{{.*}} -c -o [[LLCOUT:.*]] [[WRAPPEROUT]]
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for -sycl-embed-ir for standalone clang-linker-wrapper run for sycl (NVPTX)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -c -nocudalib -fno-sycl-libspirv --offload-new-driver -o %t3.o
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t3.devicelib.cpp
// RUN: %clang %t3.devicelib.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -nocudalib -fno-sycl-libspirv -c --offload-new-driver -o %t3.devicelib.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-libraries=%t3.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" -sycl-embed-ir "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t3.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-NV-EMBED-IR %s
// CHK-CMDS-AOT-NV-EMBED-IR: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: offload-wrapper: output: [[WRAPPEROUT1:.*]].bc, input: {{.*}}.bc
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: clang{{.*}} -c -o [[LLCOUT1:.*]] [[WRAPPEROUT1]]
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: clang{{.*}} -o [[CLANGOUT:.*]] -dumpdir a.out.nvptx64.sm_50.img. --target=nvptx64-nvidia-cuda -march={{.*}}
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: ptxas{{.*}} --output-file [[PTXASOUT:.*]] [[CLANGOUT]]
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: fatbinary{{.*}} --create [[FATBINOUT:.*]] --image=profile={{.*}},file=[[CLANGOUT]] --image=profile={{.*}},file=[[PTXASOUT]]
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: [[FATBINOUT]]
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: clang{{.*}} -c -o [[LLCOUT2:.*]] [[WRAPPEROUT]]
// CHK-CMDS-AOT-NV-EMBED-IR-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT1]] [[LLCOUT2]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for -sycl-embed-ir for standalone clang-linker-wrapper run for sycl (AMD)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx803 -fgpu-rdc -nogpulib -fno-sycl-libspirv -c --offload-new-driver -o %t4.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" -sycl-embed-ir "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t4.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-AMD-EMBED-IR %s
// CHK-CMDS-AOT-AMD-EMBED-IR: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[FIRSTLLVMLINKOUT]].bc
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: offload-wrapper: output: [[WRAPPEROUT1:.*]].bc, input: {{.*}}.bc
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: clang{{.*}} -c -o [[LLCOUT1:.*]] [[WRAPPEROUT1]]
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: clang{{.*}} -o [[CLANGOUT:.*]] -dumpdir a.out.amdgcn.gfx803.img. --target=amdgcn-amd-amdhsa -mcpu={{.*}}
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: clang-offload-bundler{{.*}} -input=[[CLANGOUT]] -output=[[BUNDLEROUT:.*]]
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: offload-wrapper: output: [[WRAPPEROUT2:.*]].bc, input: [[BUNDLEROUT]]
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: clang{{.*}} -c -o [[LLCOUT2:.*]] [[WRAPPEROUT2]]
// CHK-CMDS-AOT-AMD-EMBED-IR-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT1]] [[LLCOUT2]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

// Error handling when --linker-path is not provided for clang-linker-wrapper
// RUN: not clang-linker-wrapper 2>&1 | FileCheck --check-prefix=LINKER-PATH-NOT-PROVIDED %s
// LINKER-PATH-NOT-PROVIDED: linker path missing, must pass 'linker-path'

/// check for --device-lib-dir options for sycl-post-link.
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t5.o
//
// RUN: clang-linker-wrapper -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t5.o -sycl-device-libraries=libsycl-crt.new.o -sycl-device-library-location=%S/Inputs/SYCL/lib --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-DEVICE-LIB-DIR %s
// CHK-CMDS-DEVICE-LIB-DIR: spirv-to-ir-wrapper{{.*}} --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-DEVICE-LIB-DIR-NEXT: llvm-link{{.*}} --suppress-warnings
// CHK-CMDS-DEVICE-LIB-DIR-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings
// CHK-CMDS-DEVICE-LIB-DIR-NEXT: sycl-post-link{{.*}} --device-lib-dir={{.*}}/Inputs/SYCL/lib {{.*}} SYCL_POST_LINK_OPTIONS {{.*}}

/// check for libsycl-nativecpu_utils.bc getting linked in for Native CPU
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=native_cpu -fno-sycl-libspirv -c --offload-new-driver -o %t6.o
//
// RUN: clang-linker-wrapper "--host-triple=x86_64-unknown-linux-gnu" "-sycl-device-library-location=%S/Inputs/native_cpu" "--sycl-post-link-options=SYCL_POST_LINK_OPTIONS" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" %t6.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-NATIVE-CPU %s
// CHK-CMDS-NATIVE-CPU: spirv-to-ir-wrapper{{.*}} --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-CMDS-NATIVE-CPU-NEXT: llvm-link{{.*}} --suppress-warnings
// CHK-CMDS-NATIVE-CPU-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS
// CHK-CMDS-NATIVE-CPU-NEXT: clang{{.*}} --no-default-config -o [[OUT1:.*\.img]] -dumpdir a.out.native_cpu..img. --target=x86_64-unknown-linux-gnu -Wno-override-module -mllvm -sycl-native-cpu-backend -c {{.*}} -Xclang -mlink-bitcode-file -Xclang {{.*}}libsycl-nativecpu_utils.bc
// CHK-CMDS-NATIVE-CPU-NEXT:  offload-wrapper: output: [[OUT2:.*\.bc]], input: [[OUT1]]
// CHK-CMDS-NATIVE-CPU-NEXT: clang{{.*}} --target=x86_64-unknown-linux-gnu -c -o [[OUT3:.*\.o]] [[OUT2]]
// CHK-CMDS-NATIVE-CPU-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[OUT1]] [[OUT3]] {{.*\.o}}

// Verify that host linker is not called when --sycl-device-link is passed to clang-linker-wrapper
// RUN: clang-linker-wrapper --sycl-device-link -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-DEVLINK-CMDS %s
// CHK-DEVLINK-CMDS: spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-DEVLINK-CMDS-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-DEVLINK-CMDS-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-DEVLINK-CMDS-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-DEVLINK-CMDS-NEXT: llvm-spirv{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-DEVLINK-CMDS-NEXT: offload-wrapper: output: [[WRAPPEROUT:.*]].bc, input: {{.*}}
// CHK-DEVLINK-CMDS-NEXT: clang{{.*}} -c -o [[CLANGOUT:.*]] [[WRAPPEROUT]].bc
// CHK-DEVLINK-CMDS-NEXT: "{{.*cp|copy}}" [[CLANGOUT]] a.out
// CHK-DEVLINK-CMDS-NOT: "{{.*}}/ld"

// Verify list of commands when syclbin is used
// RUN: clang-linker-wrapper -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld -o a.out %t.o --dry-run -syclbin=executable 2>&1 | FileCheck --check-prefix CHK-SYCLBIN-CMDS %s
// CHK-SYCLBIN-CMDS:      spirv-to-ir-wrapper{{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts --spirv-preserve-auxdata --spirv-target-env=SPV-IR --spirv-builtin-format=global
// CHK-SYCLBIN-CMDS-NEXT: llvm-link{{.*}} --suppress-warnings [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc
// CHK-SYCLBIN-CMDS-NEXT: llvm-link{{.*}} -only-needed --suppress-warnings [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc
// CHK-SYCLBIN-CMDS-NEXT: sycl-post-link{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-SYCLBIN-CMDS-NEXT: llvm-spirv{{.*}} -o {{.*}}
// CHK-SYCLBIN-CMDS-NOT:  offload-wrapper: output
// CHK-SYCLBIN-CMDS-NOT:  clang
// CHK-SYCLBIN-CMDS-NEXT: "{{.*cp|copy}}" {{.*}}.syclbin a.out
// CHK-SYCLBIN-CMDS-NOT: "{{.*}}/ld"

// Check that clang-linker-wrapper errors out in case when it observes different compilation options encoded in images with the same triple and arch.
// This case tests the case of multi-step compilation.
// RUN: clang-offload-packager -o %t.packaged_1.fat "--image=file=%t.o,triple=spir64-unknown-unknown,arch=generic,kind=sycl,compile-opts=aaa,link-opts=bbb"
// RUN: clang-offload-packager -o %t.packaged_2.fat "--image=file=%t.o,triple=spir64-unknown-unknown,arch=generic,kind=sycl,compile-opts=ccc,link-opts=ddd"
//
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.embeded.o -fembed-offload-object=%t.packaged_1.fat -fembed-offload-object=%t.packaged_2.fat
//
// RUN: not clang-linker-wrapper -host-triple=x86_64-unknown-linux-gnu -sycl-device-libraries=%t.devicelib.o \
// RUN:                      %t.embeded.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s --check-prefix=COMPILE-LINK-OPTIONS-DO-NOT-MATCH

// COMPILE-LINK-OPTIONS-DO-NOT-MATCH: error: compile and link options passed to the backend of the target device compiler must be identical for device images of the same target. Mismatched options:
// COMPILE-LINK-OPTIONS-DO-NOT-MATCH-NEXT: Input[0]: compile_options: aaa, link_options: bbb
// COMPILE-LINK-OPTIONS-DO-NOT-MATCH-NEXT: Input[1]: compile_options: ccc, link_options: ddd

// Check that clang-linker-wrapper recognizes compile/link options and passes them
// over into offload wrapper and ocloc.
// RUN: clang-offload-packager -o %t.packaged_jit.fat "--image=file=%t.o,triple=spir64-unknown-unknown,arch=generic,kind=sycl,compile-opts=aaa aaa,link-opts=bbb bbb"
// RUN: clang-offload-packager -o %t.packaged_aot.fat "--image=file=%t.o,triple=spir64_gen-unknown-unknown,arch=pvc,kind=sycl,compile-opts=ccc ccc,link-opts=ddd ddd"
//
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.jit.o -fembed-offload-object=%t.packaged_jit.fat
// RUN: clang-linker-wrapper --verbose --dry-run -host-triple=x86_64-unknown-linux-gnu \
// RUN:                      -sycl-device-libraries=%t.devicelib.o \
// RUN:                      %t.jit.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s --check-prefix=CHECK-COMPILE-LINK-OPTS-JIT-WITH-ARGS
//
// CHECK-COMPILE-LINK-OPTS-JIT-WITH-ARGS: offload-wrapper: {{.*}} compile-opts: aaa aaa, link-opts: bbb bbb

// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.aot.o -fembed-offload-object=%t.packaged_aot.fat
// RUN: clang-linker-wrapper --verbose --dry-run -host-triple=x86_64-unknown-linux-gnu \
// RUN:                      -sycl-device-libraries=%t1.devicelib.o \
// RUN:                      %t.aot.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s --check-prefix=CHECK-COMPILE-LINK-OPTS-AOT
//
// Check that in AOT case backend options are passed to ocloc and are not passed to offload wrapper
// because SYCL Runtime can't make any use of it in AOT case.
// CHECK-COMPILE-LINK-OPTS-AOT: ocloc{{.*}} -device pvc ccc ccc -output
// CHECK-COMPILE-LINK-OPTS-AOT: offload-wrapper: {{.*}} compile-opts: , link-opts:

// TODO: The following check is turned off because clang-linker-wrapper doesn't support
//       processing of different targets at the moment.
// rUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.jit_and_aot.o -fembed-offload-object=%t.packaged_jit.fat -fembed-offload-object=%t.packaged_aot.fat
// Check that clang-linker-wrapper processes compilation and linking options independently for each input target.
// rUN: clang-linker-wrapper --verbose --dry-run -host-triple=x86_64-unknown-linux-gnu \
// rUN:                      -sycl-device-libraries=%t1.devicelib.o \
// rUN:                      %t.jit_and_aot.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s --check-prefix=CHECK-COMPILE-LINK-OPTS-JIT-AND-AOT

// cHECK-COMPILE-LINK-OPTS-JIT-AND-AOT: offload-wrapper: {{.*}} compile-opts: aaa aaa, link-opts: bbb bbb
// cHECK-COMPILE-LINK-OPTS-JIT-AND-AOT: ocloc{{.*}} -device pvc ccc ccc
// cHECK-COMPILE-LINK-OPTS-JIT-AND-AOT: offload-wrapper: {{.*}} compile-opts: , link-opts:

// Check that missed triple in image causes an error.
// RUN: clang-offload-packager -o %t.without_triple.fat "--image=file=%t.o,triple=,arch=pvc,kind=sycl"
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.without_triple.o -fembed-offload-object=%t.without_triple.fat
// RUN: not clang-linker-wrapper --verbose --dry-run -host-triple=x86_64-unknown-linux-gnu \
// RUN:                      -sycl-device-libraries=%t1.devicelib.o \
// RUN:                      %t.without_triple.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s --check-prefix=CHECK-ERROR-WITH-NO-TRIPLE

// CHECK-ERROR-WITH-NO-TRIPLE: linking is not supported
