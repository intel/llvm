/// Test preprocessing capabilities when using -fsycl
/// Creating a preprocessed file is expected to do an integration header
/// creation step.
// RUN: %clangxx -fsycl --offload-new-driver -E -o %t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// RUN: %clang_cl -fsycl --offload-new-driver -P -Fi%t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\.h]]"{{.*}} "-E"
// PREPROC_ONLY: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-include-footer" "[[INTFOOTER]]"{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_OUT:.+\.ii]]"

/// When compiling from preprocessed file, no integration header is expected
// RUN: touch %t.ii
// RUN: %clangxx -fsycl --offload-new-driver %t.ii -### 2>&1 | FileCheck -check-prefix PREPROC_IN %s
// PREPROC_IN-NOT: "-fsycl-int-header={{.*}}"
// PREPROC_IN: clang{{.*}} "-fsycl-is-host"

// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -E %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_PHASES %s
// PREPROC_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// PREPROC_PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// PREPROC_PHASES: 2: input, "[[INPUT]]", c++, (device-sycl)
// PREPROC_PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// PREPROC_PHASES: 4: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (spir64-unknown-unknown)" {3}, c++-cpp-output
