/// Check compilation tool steps when using the integration footer
// RUN:  %clangxx -fsycl -include dummy.h %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER %s
// FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}"{{.*}} "-include" "dummy.h"
// FOOTER: append-file{{.*}} "[[INPUTFILE:.+\.cpp]]" "--append=[[INTFOOTER]]" "--orig-filename=[[INPUTFILE]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
// FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-include" "dummy.h"{{.*}}
// FOOTER-NOT: "-include" "[[INTHEADER]]"

/// Preprocessed file creation with integration footer
// RUN: %clangxx -fsycl -E %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_GEN %s
// FOOTER_PREPROC_GEN: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}" "-o" "[[PREPROC_DEVICE:.+\.ii]]"
// FOOTER_PREPROC_GEN: append-file{{.*}} "[[INPUTFILE:.+\.cpp]]" "--append=[[INTFOOTER]]" "--orig-filename=[[INPUTFILE]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
// FOOTER_PREPROC_GEN: clang{{.*}} "-fsycl-is-host"{{.*}} "-E"{{.*}} "-o" "[[PREPROC_HOST:.+\.ii]]"{{.*}} "[[APPENDEDSRC]]"
// FOOTER_PREPROC_GEN: clang-offload-bundler{{.*}} "-inputs=[[PREPROC_DEVICE]],[[PREPROC_HOST]]"

/// Preprocessed file use with integration footer
// RUN: touch %t.ii
// RUN:  %clangxx -fsycl %t.ii -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_USE %s
// FOOTER_PREPROC_USE: clang-offload-bundler{{.*}} "-outputs=[[HOST1:.+\.ii]],[[DEVICE_PP:.+\.ii]]"
// FOOTER_PREPROC_USE: clang{{.*}} "-fsycl-is-device"{{.*}} "[[DEVICE_PP]]"
// FOOTER_PREPROC_USE: clang-offload-bundler{{.*}} "-outputs=[[HOST_PP:.+\.ii]],[[DEVICE1:.+\.ii]]"
// FOOTER_PREPROC_USE: clang{{.*}} "-fsycl-is-host"{{.*}} "[[HOST_PP]]"

/// Check that integration footer can be disabled
// RUN:  %clangxx -fsycl -fno-sycl-use-footer %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix NO-FOOTER --implicit-check-not "-fsycl-int-footer" %s
// NO-FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-sycl-std={{.*}}"
// NO-FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-o"

/// Check phases without integration footer
// RUN: %clangxx -fsycl -fno-sycl-device-lib=all -fno-sycl-use-footer -target x86_64-unknown-linux-gnu %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix NO-FOOTER-PHASES -check-prefix COMMON-PHASES %s
// NO-FOOTER-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// NO-FOOTER-PHASES: [[#HOST_PREPROC:]]: preprocessor, {0}, c++-cpp-output, (host-sycl)
// NO-FOOTER-PHASES: 2: input, "{{.*}}", c++, (device-sycl)
// NO-FOOTER-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// NO-FOOTER-PHASES: [[#DEVICE_IR:]]: compiler, {3}, ir, (device-sycl)

/// Check phases with integration footer
// RUN: %clangxx -fsycl -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu %s -ccc-print-phases 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER-PHASES -check-prefix COMMON-PHASES %s
// FOOTER-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// FOOTER-PHASES: 1: append-footer, {0}, c++, (host-sycl)
// FOOTER-PHASES: [[#HOST_PREPROC:]]: preprocessor, {1}, c++-cpp-output, (host-sycl)
// FOOTER-PHASES: 3: input, "{{.*}}", c++, (device-sycl)
// FOOTER-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// FOOTER-PHASES: [[#DEVICE_IR:]]: compiler, {4}, ir, (device-sycl)

// COMMON-PHASES: [[#OFFLOAD:]]: offload, "host-sycl (x86_64-{{.*}})" {[[#HOST_PREPROC]]}, "device-sycl (spir64-unknown-unknown-sycldevice)" {[[#DEVICE_IR]]}, c++-cpp-output
// COMMON-PHASES: [[#OFFLOAD+1]]: compiler, {[[#OFFLOAD]]}, ir, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+2]]: backend, {[[#OFFLOAD+1]]}, assembler, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+3]]: assembler, {[[#OFFLOAD+2]]}, object, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+4]]: linker, {[[#OFFLOAD+3]]}, image, (host-sycl)
// COMMON-PHASES: [[#OFFLOAD+5]]: linker, {[[#DEVICE_IR]]}, ir, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+6]]: sycl-post-link, {[[#OFFLOAD+5]]}, tempfiletable, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+7]]: file-table-tform, {[[#OFFLOAD+6]]}, tempfilelist, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+8]]: llvm-spirv, {[[#OFFLOAD+7]]}, tempfilelist, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+9]]: file-table-tform, {[[#OFFLOAD+6]], [[#OFFLOAD+8]]}, tempfiletable, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+10]]: clang-offload-wrapper, {[[#OFFLOAD+9]]}, object, (device-sycl)
// COMMON-PHASES: [[#OFFLOAD+11]]: offload, "host-sycl (x86_64-{{.*}})" {[[#OFFLOAD+4]]}, "device-sycl (spir64-unknown-unknown-sycldevice)" {[[#OFFLOAD+10]]}, image
