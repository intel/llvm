// Tests for -foffload-include-binary error paths in loadLinkModules
// (SYCL -fno-sycl-rdc compile-step device embedding).

// Missing file: err_cannot_open_file
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary no-such-file.bc -emit-obj -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FILE %s
// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file.bc'

// Bad bitcode: err_fe_linking_module
// RUN: echo "not-a-bitcode-file" > %t.bad.bc
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.bad.bc -emit-obj -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BAD-BC %s
// CHECK-BAD-BC: fatal error: cannot link module '{{.*}}.bad.bc'