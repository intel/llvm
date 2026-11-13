// RUN: not %dpclang 2>&1 | FileCheck %s --check-prefix=DIAG-C
// RUN: not %dpclangxx 2>&1 | FileCheck %s --check-prefix=DIAG-CXX
// RUN: not %dpclang_cl 2>&1 | FileCheck %s --check-prefix=DIAG-CL
// RUN: not %dpclang_cpp --dpclang-test-invalid-option %s 2>&1 | FileCheck %s --check-prefix=DIAG-CPP
// RUN: env DPCPP_DRIVER_NAME=stale not %dpclangxx 2>&1 | FileCheck %s --check-prefix=DIAG-CXX
// RUN: not %clang 2>&1 | FileCheck %s --check-prefix=DIAG-DIRECT
// RUN: env DPCPP_DRIVER_NAME= not %clang 2>&1 | FileCheck %s --check-prefix=DIAG-DIRECT

// DIAG-C: {{^}}dpclang: error: no input files
// DIAG-CXX: {{^}}dpclang++: error: no input files
// DIAG-CL: {{^}}dpclang-cl: error: no input files
// DIAG-CPP: {{^}}dpclang-cpp: error: unknown argument: '--dpclang-test-invalid-option'
// DIAG-DIRECT: {{^}}clang{{(-[0-9]+)?}}: error: no input files

// The public name must not change language or driver-mode selection.
// RUN: %dpclang -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MODE-C
// RUN: %dpclangxx -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MODE-CXX
// RUN: %dpclang_cl -### /Zs -- %s 2>&1 | FileCheck %s --check-prefix=MODE-CL
// RUN: %dpclang_cpp -### %s 2>&1 | FileCheck %s --check-prefix=MODE-CPP
// RUN: %dpclangxx --driver-mode=gcc -### -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MODE-C

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

// Exercise real subprocesses and builtin-header lookup without supplying a
// resource directory or relying on system headers.
// RUN: %dpclang -canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc %s
// RUN: %dpclang -no-canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc %s
// RUN: %dpclangxx -canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc -x c++ %s
// RUN: %dpclangxx -no-canonical-prefixes -fno-integrated-cc1 -fsyntax-only -nostdlibinc -x c++ %s

#include <stddef.h>
size_t size;
