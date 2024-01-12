// REQUIRES: system-linux

/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.o,libsycl-complex.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: "{{.*}}llvm-link" [[INPUT:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}clang-offload-bundler" -type=o -targets=sycl-spir64-unknown-unknown -input={{.*}}libsycl-crt.o -output=[[FIRSTUNBUNDLEDLIB:.*]].bc -unbundle -allow-missing-bundles
// CHK-CMDS-NEXT: "{{.*}}clang-offload-bundler" -type=o -targets=sycl-spir64-unknown-unknown -input={{.*}}libsycl-complex.o -output=[[SECONDUNBUNDLEDLIB:.*]].bc -unbundle -allow-missing-bundles
// CHK-CMDS-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc [[FIRSTUNBUNDLEDLIB]].bc [[SECONDUNBUNDLEDLIB]].bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}sycl-post-link" SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// LLVM-SPIRV is not called in dry-run
// CHK-CMDS-NEXT: offload-wrapper: input: [[LLVMSPIRVOUT:.*]].table, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl.o
