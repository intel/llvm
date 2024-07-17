/// Check compilation tool steps when using the integration footer
// RUN:  %clangxx -fsycl --no-offload-new-driver -I cmdline/dir -include dummy.h %/s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER %s -DSRCDIR=%/S -DCMDDIR=cmdline/dir
// FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}"{{.*}} "-include" "dummy.h"
// FOOTER: append-file{{.*}} "[[INPUTFILE:.+\.cpp]]" "--append=[[INTFOOTER]]" "--orig-filename=[[INPUTFILE]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
// FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-main-file-name" "[[SRCFILE:.+\cpp]]" "-fsycl-use-main-file-name{{.*}} "-include" "dummy.h"{{.*}} "-iquote" "[[SRCDIR]]" "-I" "cmdline/dir"
// FOOTER-NOT: "-include" "[[INTHEADER]]"

/// Preprocessed file creation with integration footer
// RUN: %clangxx -fsycl --no-offload-new-driver -E %/s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_GEN %s
// FOOTER_PREPROC_GEN: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}" "-o" "[[PREPROC_DEVICE:.+\.ii]]"
// FOOTER_PREPROC_GEN: append-file{{.*}} "[[INPUTFILE:.+\.cpp]]" "--append=[[INTFOOTER]]" "--orig-filename=[[INPUTFILE]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
// FOOTER_PREPROC_GEN: clang{{.*}} "-fsycl-is-host"{{.*}} "-E"{{.*}} "-o" "[[PREPROC_HOST:.+\.ii]]"{{.*}} "[[APPENDEDSRC]]"
// FOOTER_PREPROC_GEN: clang-offload-bundler{{.*}} "-input=[[PREPROC_DEVICE]]" "-input=[[PREPROC_HOST]]"

/// Preprocessed file use with integration footer
// RUN: touch %t.ii
// RUN:  %clangxx -fsycl --no-offload-new-driver %t.ii -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_USE %s
// FOOTER_PREPROC_USE: clang-offload-bundler{{.*}} "-output=[[HOST_PP:.+\.ii]]" "-output=[[DEVICE_PP:.+\.ii]]"
// FOOTER_PREPROC_USE: clang{{.*}} "-fsycl-is-device"{{.*}} "[[DEVICE_PP]]"
// FOOTER_PREPROC_USE: clang{{.*}} "-fsycl-is-host"{{.*}} "[[HOST_PP]]"

/// Check that integration footer can be disabled
// RUN:  %clangxx -fsycl --no-offload-new-driver -fno-sycl-use-footer %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix NO-FOOTER --implicit-check-not "-fsycl-int-footer" --implicit-check-not "-fsycl-use-main-file-name" %s
// NO-FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-sycl-std={{.*}}"
// NO-FOOTER-NOT: append-file
// NO-FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"

// Test that -fsycl-use-main-file-name is not passed if -fsycl --no-offload-new-driver is not passed.
// This test is located here, because -fsycl-use-main-file-name is tightly
// connected to the integration footer.
// RUN: %clangxx %s -### 2>&1 | FileCheck %s --check-prefix NO-FSYCL --implicit-check-not "-fsycl-use-main-file-name"
// NO-FSYCL: clang{{.*}} "-main-file-name" "sycl-int-footer-old-model.cpp"

/// Check phases without integration footer
// RUN: %clangxx -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fno-sycl-use-footer -target x86_64-unknown-linux-gnu %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix NO-FOOTER-PHASES -check-prefix COMMON-PHASES %s
// NO-FOOTER-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// NO-FOOTER-PHASES: [[#HOST_PREPROC:]]: preprocessor, {0}, c++-cpp-output, (host-sycl)
// NO-FOOTER-PHASES: 2: input, "{{.*}}", c++, (device-sycl)
// NO-FOOTER-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// NO-FOOTER-PHASES: [[#DEVICE_IR:]]: compiler, {3}, ir, (device-sycl)

/// Check phases with integration footer
// RUN: %clangxx -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER-PHASES -check-prefix COMMON-PHASES %s
// FOOTER-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// FOOTER-PHASES: 1: append-footer, {0}, c++, (host-sycl)
// FOOTER-PHASES: [[#HOST_PREPROC:]]: preprocessor, {1}, c++-cpp-output, (host-sycl)
// FOOTER-PHASES: 3: input, "{{.*}}", c++, (device-sycl)
// FOOTER-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// FOOTER-PHASES: [[#DEVICE_IR:]]: compiler, {4}, ir, (device-sycl)

// COMMON-PHASES: [[#OFFLOAD:]]: offload, "host-sycl (x86_64-{{.*}})" {[[#HOST_PREPROC]]}, "device-sycl (spir64-unknown-unknown)" {[[#DEVICE_IR]]}, c++-cpp-output
// COMMON-PHASES: [[#OFFLOAD+1]]: compiler, {[[#OFFLOAD]]}, ir, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+2]]: backend, {[[#OFFLOAD+1]]}, assembler, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+3]]: assembler, {[[#OFFLOAD+2]]}, object, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+4]]: linker, {[[#DEVICE_IR]]}, ir, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+5]]: sycl-post-link, {[[#OFFLOAD+4]]}, tempfiletable, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+6]]: file-table-tform, {[[#OFFLOAD+5]]}, tempfilelist, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+7]]: llvm-spirv, {[[#OFFLOAD+6]]}, tempfilelist, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+8]]: file-table-tform, {[[#OFFLOAD+5]], [[#OFFLOAD+7]]}, tempfiletable, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+9]]: clang-offload-wrapper, {[[#OFFLOAD+8]]}, object, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+10]]: offload, "device-sycl (spir64-unknown-unknown)" {[[#OFFLOAD+9]]}, object
// COMMON-PHASES: [[#OFFLOAD+11]]: linker, {[[#OFFLOAD+3]], [[#OFFLOAD+10]]}, image, (host-sycl)

/// Test for -fsycl-footer-path=<dir>
// RUN:  %clangxx -fsycl --no-offload-new-driver -fsycl-footer-path=dummy_dir %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PATH %s
// FOOTER_PATH: append-file{{.*}} "--append=dummy_dir{{(/|\\\\)}}{{.*}}-footer-{{.*}}.h"
// FOOTER_PATH-SAME: "--output=dummy_dir{{(/|\\\\)}}[[APPENDEDSRC:.+\.cpp]]"
// FOOTER_PATH: clang{{.*}} "-x" "c++" "dummy_dir{{(/|\\\\)}}[[APPENDEDSRC]]"

/// Check behaviors for dependency generation
// RUN:  %clangxx -fsycl --no-offload-new-driver -MD -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix DEP_GEN %s
// DEP_GEN:  clang{{.*}} "-fsycl-is-device"
// DEP_GEN-SAME: "-dependency-file"
// DEP_GEN-SAME: "-MT"
// DEP_GEN-SAME: "-internal-isystem" "{{.*}}{{[/\\]+}}include{{[/\\]+}}sycl"
// DEP_GEN-SAME: "-x" "c++" "[[INPUTFILE:.+\.cpp]]"
// DEP_GEN: append-file{{.*}} "[[INPUTFILE]]"
// DEP_GEN-NOT: clang{{.*}} "-dependency-file"

/// Dependency generation phases
// RUN:  %clangxx -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver -MD -c %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix DEP_GEN_PHASES %s
// DEP_GEN_PHASES: 0: input, "[[INPUTFILE:.+\.cpp]]", c++, (device-sycl)
// DEP_GEN_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// DEP_GEN_PHASES: 2: compiler, {1}, ir, (device-sycl)
// DEP_GEN_PHASES: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, ir
// DEP_GEN_PHASES: 4: input, "[[INPUTFILE]]", c++, (host-sycl)
// DEP_GEN_PHASES: 5: append-footer, {4}, c++, (host-sycl)
// DEP_GEN_PHASES: 6: preprocessor, {5}, c++-cpp-output, (host-sycl)
// DEP_GEN_PHASES: 7: offload, "host-sycl (x86_64-unknown-linux-gnu)" {6}, "device-sycl (spir64-unknown-unknown)" {2}, c++-cpp-output
// DEP_GEN_PHASES: 8: compiler, {7}, ir, (host-sycl)
// DEP_GEN_PHASES: 9: backend, {8}, assembler, (host-sycl)
// DEP_GEN_PHASES: 10: assembler, {9}, object, (host-sycl)
// DEP_GEN_PHASES: 11: clang-offload-bundler, {3, 10}, object, (host-sycl)

/// Allow for -o and preprocessing
// RUN:  %clangxx -fsycl --no-offload-new-driver -MD -c %s -o dummy -### 2>&1 \
// RUN:   | FileCheck -check-prefix DEP_GEN_OUT_ERROR %s
// DEP_GEN_OUT_ERROR-NOT: cannot specify -o when generating multiple output files
