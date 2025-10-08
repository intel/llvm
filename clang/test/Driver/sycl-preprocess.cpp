// Test the behaviors when enabling SYCL offloading with preprocessed files.

/// Creating a preprocessed file is expected to do an integration header
/// creation step.
// RUN: %clangxx -fsycl --offload-new-driver -E -o %t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// RUN: %clang_cl -fsycl --offload-new-driver -P %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\.h]]"{{.*}} "-E"
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-host"{{.*}} "-include-internal-header" "[[INTHEADER]]"{{.*}} "-include-internal-footer" "[[INTFOOTER]]"{{.*}} "-o" "[[HOST_OUT:.+\.ii]]"

/// When compiling from preprocessed file, no integration header is expected
// RUN: touch %t.ii
// RUN: %clangxx -fsycl --offload-new-driver %t.ii -### 2>&1 | FileCheck -check-prefix PREPROC_IN %s
// PREPROC_IN-NOT: "-fsycl-int-header={{.*}}"
// PREPROC_IN: clang{{.*}} "-fsycl-is-host"

/// When generating preprocessed files, verify the compilation phases.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu --offload-new-driver -fsycl -E %s -o %t.ii -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s -check-prefix PREPROC_PHASES -DTARGET=x86_64-unknown-linux-gnu
// RUN: %clang_cl --target=x86_64-pc-windows-msvc --offload-new-driver -fsycl -P %s -Fi%t.ii -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s -check-prefix PREPROC_PHASES -DTARGET=x86_64-pc-windows-msvc
// PREPROC_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// PREPROC_PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// PREPROC_PHASES: 2: input, "[[INPUT]]", c++, (device-sycl)
// PREPROC_PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// PREPROC_PHASES: 4: compiler, {3}, none, (device-sycl)
// PREPROC_PHASES: 5: offload, "device-sycl (spir64-unknown-unknown)" {3}, c++-cpp-output
// PREPROC_PHASES: 6: llvm-offload-binary, {5, 1}, c++-cpp-output
// PREPROC_PHASES: 7: offload, "host-sycl ([[TARGET]])" {1}, "device-sycl (spir64-unknown-unknown)" {3}, "device-sycl (spir64-unknown-unknown)" {4}, " ([[TARGET]])" {6}, c++-cpp-output

/// When generating preprocessed files, verify the tools called and the expected
/// output file name.
// RUN: %clangxx --offload-new-driver -fsycl -E %s -o sycl-preprocess.ii -### 2>&1 \
// RUN: | FileCheck %s -check-prefix PREPROC_TOOLS
// RUN: %clang_cl --offload-new-driver -fsycl -P %s -Fisycl-preprocess.ii -### 2>&1 \
// RUN: | FileCheck %s -check-prefix PREPROC_TOOLS
// RUN: %clang_cl --offload-new-driver -fsycl -E %s -o sycl-preprocess.ii -### 2>&1 \
// RUN: | FileCheck %s -check-prefix PREPROC_TOOLS
// PREPROC_TOOLS: clang{{.*}} "-fsycl-is-device"
// PREPROC_TOOLS-SAME: "-o" "[[DEVICE_PP_FILE:.+\.ii]]
// PREPROC_TOOLS: clang{{.*}} "-fsycl-is-host"
// PREPROC_TOOLS-SAME: "-o" "[[HOST_PP_FILE:.+\.ii]]
// PREPROC_TOOLS: llvm-offload-binary{{.*}} "-o" "sycl-preprocess.ii"
// PREPROC_TOOLS-SAME: "--image=file=[[DEVICE_PP_FILE]],triple=spir64-unknown-unknown,arch=generic,kind=sycl{{.*}}" "--image=file=[[HOST_PP_FILE]],triple={{.*}},arch=generic,kind=host"
