// REQUIRES: system-linux

/// ########################################################################

/// Check the phases for SYCL compilation using new offload model
// RUN: %clangxx -ccc-print-phases -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES:       0: input, "[[INPUT:.*]]", c++, (host-sycl)
// CHK-PHASES-NEXT:  1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NEXT:  2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NEXT:  3: compiler, {2}, ir, (host-sycl)
// CHK-PHASES-NEXT:  4: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-NEXT:  5: preprocessor, {4}, c++-cpp-output, (device-sycl)
// CHK-PHASES-NEXT:  6: compiler, {5}, ir, (device-sycl)
// CHK-PHASES-NEXT:  7: backend, {6}, assembler, (device-sycl)
// CHK-PHASES-NEXT:  8: assembler, {7}, object, (device-sycl)
// CHK-PHASES-NEXT:  9: offload, "device-sycl (spir64-unknown-unknown)" {8}, object
// CHK-PHASES-NEXT: 10: clang-offload-packager, {9}, image, (device-sycl)
// CHK-PHASES-NEXT: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {10}, ir
// CHK-PHASES-NEXT: 12: backend, {11}, assembler, (host-sycl)
// CHK-PHASES-NEXT: 13: assembler, {12}, object, (host-sycl)
// CHK-PHASES-NEXT: 14: clang-linker-wrapper, {13}, image, (host-sycl)

/// ########################################################################

/// Check the toolflow for SYCL compilation using new offload model
// RUN: %clangxx -### -fsycl --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-FLOW %s
// CHK-FLOW: "[[PATH:.*]]/clang-18" "-cc1" "-triple" "spir64-unknown-unknown" "-aux-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" {{.*}} "-fsycl-int-header=[[HEADER:.*]].h" "-fsycl-int-footer=[[FOOTER:.*]].h" {{.*}} "--offload-new-driver" {{.*}} "-o" "[[CC1DEVOUT:.*]]" "-x" "c++" "[[INPUT:.*]]"
// CHK-FLOW-NEXT: "[[PATH]]/clang-offload-packager" "-o" "[[PACKOUT:.*]]" "--image=file=[[CC1DEVOUT]],triple=spir64-unknown-unknown,arch=,kind=sycl"
// CHK-FLOW-NEXT: "[[PATH]]/append-file" "[[INPUT]]" "--append=[[FOOTER]].h" "--orig-filename=[[INPUT]]" "--output=[[APPENDOUT:.*]]" "--use-include"
// CHK-FLOW-NEXT: "[[PATH]]/clang-18" "-cc1" "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[HEADER]].h" "-dependency-filter" "[[HEADER]].h" {{.*}} "-fsycl-is-host"{{.*}} "-full-main-file-name" "[[INPUT]]" {{.*}} "--offload-new-driver" {{.*}} "-fembed-offload-object=[[PACKOUT]]" {{.*}} "-o" "[[CC1FINALOUT:.*]]" "-x" "c++" "[[APPENDOUT]]"
// CHK-FLOW-NEXT: "[[PATH]]/clang-linker-wrapper" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" {{.*}} "[[CC1FINALOUT]]"

/// Check for no crashes for SYCL compilation using new offload model
/// TODO(NOM4): Enable when driver support is available
// RUNXXX: %clangxx -fsycl --offload-new-driver %s

/// Check for list of commands for standalone clang-linker-wrapper run for sycl
// RUN: clang-linker-wrapper -sycl-device-library-location=%S/Inputs -sycl-device-libraries=libsycl-crt.o,libsycl-complex.o -sycl-post-link-options="SYCL_POST_LINK_OPTIONS" -llvm-spirv-options="LLVM_SPIRV_OPTIONS" "--host-triple=x86_64-unknown-linux-gnu" "--triple=spir64" "--linker-path=/usr/bin/ld" "--" HOST_LINKER_FLAGS "-dynamic-linker" HOST_DYN_LIB "-o" "a.out" HOST_LIB_PATH HOST_STAT_LIB %S/Inputs/test-sycl.o --dry-run 2>&1 | FileCheck -check-prefix=CHK-CMDS %s
// CHK-CMDS: "[[PATH:.*]]/llvm-link" [[INPUT:.*]].bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "[[PATH]]/clang-offload-bundler" -type=o -targets=sycl-spir64-unknown-unknown -input={{.*}}libsycl-crt.o -output=[[FIRSTUNBUNDLEDLIB:.*]].bc -unbundle -allow-missing-bundles
// CHK-CMDS-NEXT: "[[PATH]]/clang-offload-bundler" -type=o -targets=sycl-spir64-unknown-unknown -input={{.*}}libsycl-complex.o -output=[[SECONDUNBUNDLEDLIB:.*]].bc -unbundle -allow-missing-bundles
// CHK-CMDS-NEXT: "[[PATH]]/llvm-link" -only-needed [[FIRSTLLVMLINKOUT]].bc [[FIRSTUNBUNDLEDLIB]].bc [[SECONDUNBUNDLEDLIB]].bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CHK-CMDS-NEXT: "[[PATH]]/sycl-post-link" SYCL_POST_LINK_OPTIONS -o [[SYCLPOSTLINKOUT:.*]].table [[SECONDLLVMLINKOUT]].bc

/// LLVM-SPIRV is not called in dry-run
// CHK-CMDS-NEXT: "[[PATH]]/clang-offload-wrapper" -o=[[WRAPPEROUT:.*]].bc -host=x86_64-unknown-linux-gnu -target=spir64 -kind=sycl -batch [[LLVMSPIRVOUT:.*]].table
// CHK-CMDS-NEXT: "[[PATH]]/llc" -filetype=obj -o [[LLCOUT:.*]].o [[WRAPPEROUT]].bc
// CHK-CMDS-NEXT: "[[LOADER_PATH:.*]]/ld" HOST_LINKER_FLAGS -dynamic-linker HOST_DYN_LIB -o a.out [[LLCOUT]].o HOST_LIB_PATH HOST_STAT_LIB {{.*}}test-sycl.o