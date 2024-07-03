// REQUIRES: system-linux

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
// CHK-CMDS: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}sycl-post-link"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-NEXT: "{{.*}}llvm-spirv"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]] [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// check for PIC for device wrap compilation when using -shared
// RUN: clang-linker-wrapper -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" -shared "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SHARED %s
// CHK-SHARED: "{{.*}}llc"{{.*}} -relocation-model=pic

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
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-libraries=%t1.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t1.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-GEN %s
// CHK-CMDS-AOT-GEN: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}sycl-post-link"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-spirv"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}ocloc"{{.*}} -output_no_suffix -spirv_input -device pvc {{.*}} -output {{.*}} -file {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
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
// CHK-CMDS-AOT-CPU: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}sycl-post-link"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-spirv"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}opencl-aot"{{.*}} --device=cpu -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

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
// CHK-CMDS-AOT-NV: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}sycl-post-link"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}clang"{{.*}} -o [[CLANGOUT:.*]] --target=nvptx64-nvidia-cuda -march={{.*}}
// CHK-CMDS-AOT-NV-NEXT: offload-wrapper: input: [[WRAPPERIN:.*]], output: [[WRAPPEROUT:.*]]
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]] [[WRAPPEROUT]]
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for AMD)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx803 -fgpu-rdc -nogpulib -fno-sycl-libspirv -c --offload-new-driver -o %t4.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t4.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-AMD %s
// CHK-CMDS-AOT-AMD: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}sycl-post-link"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[FIRSTLLVMLINKOUT]].bc
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}clang"{{.*}} -o [[CLANGOUT:.*]] --target=amdgcn-amd-amdhsa -mcpu={{.*}}
// CHK-CMDS-AOT-AMD-NEXT: offload-wrapper: input: [[WRAPPERIN:.*]], output: [[WRAPPEROUT:.*]]
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]] [[WRAPPEROUT]]
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]] HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o
