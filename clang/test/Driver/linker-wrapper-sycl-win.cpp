// REQUIRES: system-windows

/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 %s -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t.bc
// RUN: clang-offload-packager -o %t.out --image=file=%t.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
//
// Generate .o file as SYCL device library file.
//
// RUN: echo '' > %t.devicelib.cpp
// RUN: %clang -cc1 %t.devicelib.cpp -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t.devicelib.bc
// RUN: clang-offload-packager -o %t.devicelib.out --image=file=%t.devicelib.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t.devicelib.o \
// RUN:   -fembed-offload-object=%t.devicelib.out
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-library-location= -sycl-device-libraries=%t.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-NEXT: "{{.*}}llvm-spirv.exe"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for Intel GPU)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 %s -triple spir64_gen-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t1.bc
// RUN: clang-offload-packager -o %t1.out --image=file=%t1.bc,kind=sycl,triple=spir64_gen-unknown-unknown,arch=pvc
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t1.o \
// RUN:   -fembed-offload-object=%t1.out
//
// Generate .o file as SYCL device library file.
//
// RUN: echo '' > %t1.devicelib.cpp
// RUN: %clang -cc1 %t1.devicelib.cpp -triple spir64_gen-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t1.devicelib.bc
// RUN: clang-offload-packager -o %t1.devicelib.out --image=file=%t1.devicelib.bc,kind=sycl,triple=spir64_gen-unknown-unknown,arch=pvc
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t1.devicelib.o \
// RUN:   -fembed-offload-object=%t1.devicelib.out
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-library-location= -sycl-device-libraries=%t1.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t1.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-GEN %s
// CHK-CMDS-AOT-GEN: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-spirv.exe"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}ocloc{{.*}} -output_no_suffix -spirv_input -device pvc -output {{.*}} -file {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for Intel CPU)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 %s -triple spir64_x86_64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t2.bc
// RUN: clang-offload-packager -o %t2.out --image=file=%t2.bc,kind=sycl,triple=spir64_x86_64-unknown-unknown
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t2.o \
// RUN:   -fembed-offload-object=%t2.out
//
// Generate .o file as SYCL device library file.
//
// RUN: echo '' > %t2.devicelib.cpp
// RUN: %clang -cc1 %t2.devicelib.cpp -triple spir64_x86_64-unknown-unknown -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t2.devicelib.bc
// RUN: clang-offload-packager -o %t2.devicelib.out --image=file=%t2.devicelib.bc,kind=sycl,triple=spir64_x86_64-unknown-unknown
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t2.devicelib.o \
// RUN:   -fembed-offload-object=%t2.devicelib.out
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-library-location= -sycl-device-libraries=%t2.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t2.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-CPU %s
// CHK-CMDS-AOT-CPU: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-spirv.exe"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}opencl-aot.exe"{{.*}} --device=cpu -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for NVPTX)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t3.bc
// RUN: clang-offload-packager -o %t3.out --image=file=%t3.bc,kind=sycl,triple=nvptx64-nvidia-cuda
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t3.o \
// RUN:   -fembed-offload-object=%t3.out
//
// Generate .o file as SYCL device library file.
//
// RUN: echo '' > %t3.devicelib.cpp
// RUN: %clang -cc1 %t3.devicelib.cpp -triple nvptx64-nvidia-cuda -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t3.devicelib.bc
// RUN: clang-offload-packager -o %t3.devicelib.out --image=file=%t3.devicelib.bc,kind=sycl,triple=nvptx64-nvidia-cuda
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t3.devicelib.o \
// RUN:   -fembed-offload-object=%t3.devicelib.out
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-library-location= -sycl-device-libraries=%t3.devicelib.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t3.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-NV %s
// CHK-CMDS-AOT-NV: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}clang.exe"{{.*}} -o [[CLANGOUT:.*]] --target=nvptx64-nvidia-cuda -march={{.*}}
// CHK-CMDS-AOT-NV-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-NV-NEXT: "{{.*}}ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for AMD)
// -------
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -fsycl-is-device -emit-llvm-bc -o %t4.bc
// RUN: clang-offload-packager -o %t4.out --image=file=%t4.bc,kind=sycl,triple=amdgcn-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t4.o \
// RUN:   -fembed-offload-object=%t4.out
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper -sycl-device-library-location= -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %t4.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-AMD %s
// CHK-CMDS-AOT-AMD: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[FIRSTLLVMLINKOUT]].bc
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}clang.exe"{{.*}} -o [[CLANGOUT:.*]] --target=amdgcn-amd-amdhsa -mcpu={{.*}}
// CHK-CMDS-AOT-AMD-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-AMD-NEXT: "{{.*}}ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}.o
