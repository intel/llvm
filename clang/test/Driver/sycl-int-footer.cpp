/// Check compilation tool steps when using the integration footer
// RUN:  %clangxx -fsycl --offload-new-driver -I cmdline/dir -include dummy.h %/s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER %s -DSRCDIR=%/S -DCMDDIR=cmdline/dir
// FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}"{{.*}} "-include" "dummy.h"
// FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"
// FOOTER-SAME: "-include-footer" "[[INTFOOTER]]"
// FOOTER-SAME: "-fsycl-is-host"{{.*}} "-main-file-name" "[[SRCFILE:.+\cpp]]" "-fsycl-use-main-file-name"{{.*}} "-include" "dummy.h"{{.*}} "-I" "cmdline/dir"
// FOOTER-NOT: "-include" "[[INTHEADER]]"

/// Preprocessed file creation with integration footer
// RUN: %clangxx -fsycl --offload-new-driver -E %/s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_GEN %s
// FOOTER_PREPROC_GEN: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}" "-o" "[[PREPROC_DEVICE:.+\.ii]]"
// FOOTER_PREPROC_GEN: clang{{.*}} "-include" "[[INTHEADER]]"
// FOOTER_PREPROC_GEN-SAME: "-include-footer" "[[INTFOOTER]]"
// FOOTER_PREPROC_GEN-SAME: "-fsycl-is-host"{{.*}} "-E"{{.*}} "-o" "-"

/// Preprocessed file use with integration footer
// RUN: touch %t.ii
// RUN:  %clangxx -fsycl --offload-new-driver %t.ii -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_USE %s
// FOOTER_PREPROC_USE: clang{{.*}} "-fsycl-is-host"

/// Check that integration footer can be disabled
// RUN:  %clangxx -fsycl --offload-new-driver -fno-sycl-use-footer %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix NO-FOOTER --implicit-check-not "-fsycl-int-footer" --implicit-check-not "-fsycl-use-main-file-name" %s
// NO-FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-sycl-std={{.*}}"
// NO-FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-main-file-name" "sycl-int-footer.cpp"{{.*}} "-o"

// Test that -fsycl-use-main-file-name is not passed if -fsycl --offload-new-driver is not passed.
// This test is located here, because -fsycl-use-main-file-name is tightly
// connected to the integration footer.
// RUN: %clangxx %s -### 2>&1 | FileCheck %s --check-prefix NO-FSYCL --implicit-check-not "-fsycl-use-main-file-name"
// NO-FSYCL: clang{{.*}} "-main-file-name" "sycl-int-footer.cpp"

/// Check phases without integration footer
// RUN: %clangxx -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fno-sycl-use-footer -target x86_64-unknown-linux-gnu %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix NO-FOOTER-PHASES -check-prefix COMMON-PHASES %s
// NO-FOOTER-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// NO-FOOTER-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// NO-FOOTER-PHASES: [[#HOST_IR:]]: compiler, {1}, ir, (host-sycl)
// NO-FOOTER-PHASES: 3: input, "{{.*}}", c++, (device-sycl)
// NO-FOOTER-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// NO-FOOTER-PHASES: [[#DEVICE_IR:]]: compiler, {4}, ir, (device-sycl)

/// Check phases with integration footer
// RUN: %clangxx -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER-PHASES -check-prefix COMMON-PHASES %s
// FOOTER-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// FOOTER-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// FOOTER-PHASES: [[#HOST_IR:]]: compiler, {1}, ir, (host-sycl)
// FOOTER-PHASES: 3: input, "{{.*}}", c++, (device-sycl)
// FOOTER-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// FOOTER-PHASES: [[#DEVICE_IR:]]: compiler, {4}, ir, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD:]]: backend, {[[#DEVICE_IR]]}, ir, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+1]]: offload, "device-sycl (spir64-unknown-unknown)" {[[#OFFLOAD]]}, ir
// COMMON-PHASES: [[#OFFLOAD+2]]: clang-offload-packager, {[[#OFFLOAD+1]]}, image, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+3]]: offload, "host-sycl (x86_64-unknown-linux-gnu)" {[[#HOST_IR]]}, "device-sycl (x86_64-unknown-linux-gnu)" {[[#OFFLOAD+2]]}, ir
// COMMON-PHASES: [[#OFFLOAD+4]]: backend, {[[#OFFLOAD+3]]}, assembler, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+5]]: assembler, {[[#OFFLOAD+4]]}, object, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+6]]: clang-linker-wrapper, {[[#OFFLOAD+5]]}, image, (host-sycl)

/// Test for -fsycl-footer-path=<dir>
// RUN:  %clangxx -fsycl --offload-new-driver -fsycl-footer-path=dummy_dir %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PATH %s
// FOOTER_PATH: clang{{.*}} "-fsycl-is-device"
// FOOTER_PATH-SAME: "-fsycl-int-footer=dummy_dir{{(/|\\\\)}}{{.*}}-footer-{{.*}}.h"
// FOOTER_PATH: clang{{.*}} "-include-footer" "dummy_dir{{(/|\\\\)}}{{.*}}-footer-{{.*}}.h"

/// Check behaviors for dependency generation
// RUN:  %clangxx -fsycl --offload-new-driver -MD -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix DEP_GEN %s
// DEP_GEN:  clang{{.*}} "-fsycl-is-device"
// DEP_GEN-SAME: "-dependency-file" "[[DEPFILE:.+\.d]]"
// DEP_GEN-SAME: "-MT"
// DEP_GEN-SAME: "-internal-isystem" "{{.*}}{{[/\\]+}}include{{[/\\]+}}sycl"
// DEP_GEN-SAME: "-x" "c++" "[[INPUTFILE:.+\.cpp]]"
// DEP_GEN: clang{{.*}} "-fsycl-is-host"
// DEP_GEN-SAME: "-dependency-file" "[[DEPFILE]]"

/// Dependency generation phases
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -MD -c %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix DEP_GEN_PHASES %s
// DEP_GEN_PHASES: 0: input, "[[INPUTFILE:.+\.cpp]]", c++, (host-sycl)
// DEP_GEN_PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// DEP_GEN_PHASES: 2: compiler, {1}, ir, (host-sycl)
// DEP_GEN_PHASES: 3: input, "[[INPUTFILE]]", c++, (device-sycl)
// DEP_GEN_PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// DEP_GEN_PHASES: 5: compiler, {4}, ir, (device-sycl)
// DEP_GEN_PHASES: 6: backend, {5}, ir, (device-sycl)
// DEP_GEN_PHASES: 7: offload, "device-sycl (spir64-unknown-unknown)" {6}, ir
// DEP_GEN_PHASES: 8: clang-offload-packager, {7}, image, (device-sycl)
// DEP_GEN_PHASES: 9: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (x86_64-unknown-linux-gnu)" {8}, ir
// DEP_GEN_PHASES: 10: backend, {9}, assembler, (host-sycl)
// DEP_GEN_PHASES: 11: assembler, {10}, object, (host-sycl)

/// Allow for -o and preprocessing
// RUN:  %clangxx -fsycl --offload-new-driver -MD -c %s -o dummy -### 2>&1 \
// RUN:   | FileCheck -check-prefix DEP_GEN_OUT_ERROR %s
// DEP_GEN_OUT_ERROR-NOT: cannot specify -o when generating multiple output files
