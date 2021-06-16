/// Check compilation tool steps when using the integration footer
// RUN:  %clangxx -fsycl %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER %s
// FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}"
// FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-E"{{.*}} "-C"{{.*}} "-o" "[[PREPROC:.+\.ii]]"
// FOOTER: append-file{{.*}} "[[PREPROC]]" "--append=[[INTFOOTER]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
// FOOTER: clang{{.*}} "-fsycl-is-host"{{.*}} "[[APPENDEDSRC]]"
// FOOTER-NOT: "-include" "[[INTHEADER]]"

/// Preprocessed file creation with integration footer
// RUN: %clangxx -fsycl -E %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER_PREPROC_GEN %s
// FOOTER_PREPROC_GEN: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}" "-o" "[[PREPROC_DEVICE:.+\.ii]]"
// FOOTER_PREPROC_GEN: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-E"{{.*}} "-o" "[[PREPROC1:.+\.ii]]"
// FOOTER_PREPROC_GEN: append-file{{.*}} "[[PREPROC1]]" "--append=[[INTFOOTER]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
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
