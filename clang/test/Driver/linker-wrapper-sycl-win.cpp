// REQUIRES: system-windows

/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.obj,libsycl-complex.new.obj -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-NEXT: "{{.*}}llvm-spirv.exe"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
// CHK-CMDS-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for Intel GPU)
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.obj,libsycl-complex.new.obj -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl-aot-gen.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-GEN %s
// CHK-CMDS-AOT-GEN: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llvm-spirv.exe"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
/// -DFOO1 and -DFOO2 are backend options parsed from device image.
/// These options were specified when the input fat binary (test-sycl-aot-gen.o) was compiled.
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}ocloc{{.*}} -output_no_suffix -spirv_input -device pvc -device_options pvc -ze-intel-enable-auto-large-GRF-mode -DFOO1 -DFOO2 -output {{.*}} -file {{.*}}
// CHK-CMDS-AOT-GEN-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-GEN-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl-aot-gen.o

/// Check for list of commands for standalone clang-linker-wrapper run for sycl (AOT for Intel CPU)
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.obj,libsycl-complex.new.obj -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-pc-windows-msvc" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl-aot-cpu.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS-AOT-CPU %s
// CHK-CMDS-AOT-CPU: "{{.*}}spirv-to-ir-wrapper.exe" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-link.exe" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-link.exe" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}sycl-post-link.exe"{{.*}} SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llvm-spirv.exe"{{.*}} LLVM_SPIRV_OPTIONS -o {{.*}}
/// -DFOO1 and -DFOO2 are backend options parsed from device image.
/// These options were specified when the input fat binary (test-sycl-aot-cpu.o) was compiled.
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}opencl-aot.exe"{{.*}} --device=cpu -DFOO1 -DFOO2 -o {{.*}}
// CHK-CMDS-AOT-CPU-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}llc.exe" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-AOT-CPU-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl-aot-cpu.o
