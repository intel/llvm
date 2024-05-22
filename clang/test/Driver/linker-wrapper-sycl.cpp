// REQUIRES: system-linux

/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.o,libsycl-complex.new.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}sycl-post-link" SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// LLVM-SPIRV is not called in dry-run
// CHK-CMDS-NEXT: offload-wrapper: input: [[LLVMSPIRVOUT:.*]].table, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl.o
