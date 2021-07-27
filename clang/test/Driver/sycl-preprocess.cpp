/// Test preprocessing capabilities when using -fsycl
/// Creating a preprocessed file is expected to do an integration header
/// creation step.
// RUN: %clangxx -fsycl -E -o %t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// RUN: %clang_cl -fsycl -P -Fi%t_output.ii %s -### 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_ONLY %s
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-device"{{.*}} "-E"{{.*}} "-o" "[[DEVICE_OUT:.+\.ii]]"
// PREPROC_ONLY: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\.h]]"{{.*}} "-fsyntax-only"
// PREPROC_ONLY: append-file{{.*}} "--append=[[INTFOOTER]]"{{.*}} "--output=[[HOST_APPENDED:.+\.cpp]]"
// PREPROC_ONLY: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_OUT:.+\.ii]]"{{.*}} "[[HOST_APPENDED]]"
// PREPROC_ONLY: clang-offload-bundler{{.*}} "-type=ii"{{.*}} "-outputs={{.+_output.ii}}" "-inputs=[[DEVICE_OUT]],[[HOST_OUT]]"

/// When compiling from preprocessed file, no integration header is expected
// RUN: touch %t.ii
// RUN: %clangxx -fsycl %t.ii -### 2>&1 | FileCheck -check-prefix PREPROC_IN %s
// PREPROC_IN: clang{{.*}} "-fsycl-is-device"
// PREPROC_IN-NOT: "-fsycl-int-header={{.*}}"
// PREPROC_IN: clang{{.*}} "-fsycl-is-host"

// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -E %s -ccc-print-phases 2>&1 \
// RUN:  | FileCheck -check-prefix PREPROC_PHASES %s
// PREPROC_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (device-sycl)
// PREPROC_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// PREPROC_PHASES: 2: offload, "device-sycl (spir64-unknown-unknown-sycldevice)" {1}, c++-cpp-output
// PREPROC_PHASES: 3: input, "[[INPUT]]", c++, (host-sycl)
// PREPROC_PHASES: 4: compiler, {1}, none, (device-sycl)
// PREPROC_PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (spir64-unknown-unknown-sycldevice)" {4}, c++
// PREPROC_PHASES: 6: append-footer, {5}, c++, (host-sycl)
// PREPROC_PHASES: 7: preprocessor, {6}, c++-cpp-output, (host-sycl)
// PREPROC_PHASES: 8: clang-offload-bundler, {2, 7}, c++-cpp-output, (host-sycl)
