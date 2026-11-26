// The real driver aliases can be omitted through CLANG_LINKS_TO_CREATE.
// RUN: not %dpclang 2>&1 | FileCheck %s --check-prefix=DIAG-C
// RUN: %if clang++ %{ not %dpclangxx 2>&1 | FileCheck %s --check-prefix=DIAG-CXX %}
// RUN: %if clang-cl %{ not %dpclang_cl 2>&1 | FileCheck %s --check-prefix=DIAG-CL %}
// RUN: %if clang-cpp %{ not %dpclang_cpp --dpclang-test-invalid-option %s 2>&1 | FileCheck %s --check-prefix=DIAG-CPP %}
// RUN: not %clang 2>&1 | FileCheck %s --check-prefix=DIAG-DIRECT

// DIAG-C: {{^}}dpclang: error: no input files
// DIAG-CXX: {{^}}dpclang++: error: no input files
// DIAG-CL: {{^}}dpclang-cl: error: no input files
// DIAG-CPP: {{^}}dpclang-cpp: error: unknown argument: '--dpclang-test-invalid-option'
// DIAG-DIRECT: {{^}}clang{{(-[0-9]+)?}}: error: no input files

// RUN: %dpclang --help | FileCheck %s --check-prefix=HELP-C --implicit-check-not=dpcpp-driver
// RUN: %if clang++ %{ %dpclangxx --help | FileCheck %s --check-prefix=HELP-CXX --implicit-check-not=dpcpp-driver %}
// RUN: %if clang-cl %{ %dpclang_cl -help | FileCheck %s --check-prefix=HELP-CL --implicit-check-not=dpcpp-driver %}
// RUN: %if clang-cpp %{ %dpclang_cpp --help | FileCheck %s --check-prefix=HELP-CPP --implicit-check-not=dpcpp-driver %}
// RUN: %clang --help | FileCheck %s --check-prefix=HELP-DIRECT --implicit-check-not=dpcpp-driver

// HELP-C: {{^}}USAGE: dpclang [options] file...
// HELP-CXX: {{^}}USAGE: dpclang++ [options] file...
// HELP-CL: {{^}}USAGE: dpclang-cl [options] file...
// HELP-CPP: {{^}}USAGE: dpclang-cpp [options] file...
// HELP-DIRECT: {{^}}USAGE: {{(clang(-[0-9]+)?|llvm)(\.exe)?}} [options] file...

// The wrapper marker must not interfere with response-file expansion or direct
// integrated-tool dispatch. A leading blank line exercises removal of the EOL
// sentinels used by clang-cl response files.
// RUN: %dpclang -cc1 -version 2>&1 | FileCheck %s --check-prefix=CC1-VERSION
// RUN: %if clang-cl %{ %dpclang_cl -cc1 -version 2>&1 | FileCheck %s --check-prefix=CC1-VERSION %}
// RUN: echo "" > %t.rsp
// RUN: echo "-cc1" >> %t.rsp
// RUN: echo "-version" >> %t.rsp
// RUN: %dpclang @%t.rsp 2>&1 | FileCheck %s --check-prefix=CC1-VERSION
// RUN: %if clang-cl %{ %dpclang_cl @%t.rsp 2>&1 | FileCheck %s --check-prefix=CC1-VERSION %}
// CC1-VERSION: LLVM version

// The dp-prefixed name must not change language or driver-mode selection.
// RUN: %dpclang -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MODE-C --implicit-check-not=dpcpp-driver
// RUN: %if clang++ %{ %dpclangxx -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MODE-CXX --implicit-check-not=dpcpp-driver %}
// RUN: %if clang-cl %{ %dpclang_cl -### /Zs -- %s 2>&1 | FileCheck %s --check-prefix=MODE-CL --implicit-check-not=dpcpp-driver %}
// RUN: %if clang-cpp %{ %dpclang_cpp -### %s 2>&1 | FileCheck %s --check-prefix=MODE-CPP --implicit-check-not=dpcpp-driver %}
// RUN: %if clang++ %{ %dpclangxx --driver-mode=gcc -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MODE-C --implicit-check-not=dpcpp-driver %}

// MODE-C: "-cc1"
// MODE-C-SAME: "-fsyntax-only"
// MODE-C-SAME: "-x" "c"
// MODE-CXX: "-cc1"
// MODE-CXX-SAME: "-fsyntax-only"
// MODE-CXX-SAME: "-x" "c++"
// MODE-CL: "-cc1"
// MODE-CL-SAME: "-fsyntax-only"
// MODE-CL-SAME: "-fdiagnostics-format" "msvc"
// MODE-CPP: "-cc1"
// MODE-CPP-SAME: "-E"
