// REQUIRES: system-linux

/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.o,libsycl-complex.new.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-CMDS-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "{{.*}}sycl-post-link" SYCL_POST_LINK_OPTIONS {{.*}} -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc
// LLVM-SPIRV is not called in dry-run
// CHK-CMDS-NEXT: offload-wrapper: input: {{.*}}, output: [[WRAPPEROUT:.*]].bc
// CHK-CMDS-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl.o

// Check sycl-module-split-mode command line option
// RUN: clang-linker-wrapper -sycl-module-split-mode=auto -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.o,libsycl-complex.new.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SPLIT-CMDS %s
// CHK-SPLIT-CMDS: "{{.*}}spirv-to-ir-wrapper" {{.*}} -o [[FIRSTLLVMLINKIN:.*]].bc --llvm-spirv-opts=--spirv-preserve-auxdata --llvm-spirv-opts=--spirv-target-env=SPV-IR --llvm-spirv-opts=--spirv-builtin-format=global
// CHK-SPLIT-CMDS-NEXT: "{{.*}}llvm-link" [[FIRSTLLVMLINKIN]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-SPLIT-CMDS-NEXT: "{{.*}}llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}.bc {{.*}}.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-SPLIT-CMDS-NEXT: sycl-module-split: input: [[SECONDLLVMLINKOUT]].bc, output: [[SYCLMODULESPLITOUT:.*]].bc
// CHK-SPLIT-CMDS-NEXT: "{{.*}}llvm-spirv" LLVM_SPIRV_OPTIONS -o [[SPIRVOUT:.*]].spv [[SYCLMODULESPLITOUT]].bc
// LLVM-SPIRV is not called in dry-run
// CHK-SPLIT-CMDS-NEXT: offload-wrapper: input: [[SPIRVOUT]].spv, output: [[WRAPPEROUT:.*]].bc
// CHK-SPLIT-CMDS-NEXT: "{{.*}}llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-SPLIT-CMDS-NEXT: "{{.*}}/ld" -- HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl.o

/// check for PIC for device wrap compilation when using -shared
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.new.o,libsycl-complex.new.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" -shared "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-SHARED %s
// CHK-SHARED: "{{.*}}llc"{{.*}} -relocation-model=pic
