/// Test preprocessing capabilities when using -fsycl
/// Creating a preprocessed file is expected to do an integration header
/// creation step.
// RUN: %clangxx -fsycl --no-offload-new-driver -E -o %t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// RUN: %clang_cl -fsycl --no-offload-new-driver -P -Fi%t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-device"{{.*}} "-E"{{.*}} "-o" "[[DEVICE_OUT:.+\.ii]]"
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\.h]]"{{.*}} "-fsyntax-only"
// PREPROC_ONLY: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-include-footer" "[[INTFOOTER]]"{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_OUT:.+\.ii]]"

/// When compiling from preprocessed file, no integration header is expected
// RUN: touch %t.ii
// RUN: %clangxx -fsycl --no-offload-new-driver %t.ii -### 2>&1 | FileCheck -check-prefix PREPROC_IN %s
// PREPROC_IN: clang{{.*}} "-fsycl-is-device"
// PREPROC_IN-NOT: "-fsycl-int-header={{.*}}"
// PREPROC_IN: clang{{.*}} "-fsycl-is-host"

// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -E %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_PHASES %s
// PREPROC_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// PREPROC_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// PREPROC_PHASES: 2: offload, "device-sycl (spir64-unknown-unknown)" {1}, c++-cpp-output
// PREPROC_PHASES: 3: input, "[[INPUT]]", c++, (host-sycl)
// PREPROC_PHASES: 4: compiler, {1}, none, (device-sycl)
// PREPROC_PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown)" {4}, c++
// PREPROC_PHASES: 6: preprocessor, {5}, c++-cpp-output, (host-sycl)
// PREPROC_PHASES: 7: clang-offload-bundler, {2, 6}, c++-cpp-output, (host-sycl)
