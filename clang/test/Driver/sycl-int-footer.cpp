/// Check compilation tool steps when using the integrated footer
// RUN:  %clangxx -fsycl -fsycl-use-footer %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix FOOTER %s
// FOOTER: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[INTHEADER:.+\.h]]" "-fsycl-int-footer=[[INTFOOTER:.+\h]]" "-sycl-std={{.*}}"
// FOOTER: clang{{.*}} "-include" "[[INTHEADER]]"{{.*}} "-fsycl-is-host"{{.*}} "-E"{{.*}} "-o" "[[PREPROC:.+\.ii]]"
// FOOTER: append-file{{.*}} "[[PREPROC]]" "--append=[[INTFOOTER]]" "--output=[[APPENDEDSRC:.+\.cpp]]"
// FOOTER: clang{{.*}} "-fsycl-is-host"{{.*}} "[[APPENDEDSRC]]"
// FOOTER-NOT: "-include" "[[INTHEADER]]"
