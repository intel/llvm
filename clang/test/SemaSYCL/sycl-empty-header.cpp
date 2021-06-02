// Test that at least an empty integration header file is created when
// compiling source files with no device sycl construct.

// RUN: mkdir -p %t_dir
// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h -save-temps=cwd %s
// RUN: ls %t.h
// RUN: rm -f %t.h
// RUN: touch %t.fail.h
// RUN: chmod 400 %t.fail.h
// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.fail.h %s 2>&1 | FileCheck %s --check-prefix=SYCL-BADFILE
// RUN: rm -f %t.fail.h
// SYCL-BADFILE: Error: {{[Pp]ermission}} denied when opening {{.*.fail.h}}
