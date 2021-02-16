// Test that at least an empty integration header file is created when
// compiling source files with no device sycl construct.

// RUN: mkdir -p %t_dir
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-int-header=%t.h -save-temps=cwd %s
// RUN: ls %t.h
// RUN: rm -f %t.h
// RUN: cp %s %t_dir/foo.cpp
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-int-header=%t.h -save-temps=cwd %t_dir/foo.cpp
// RUN: ls %t.h
// RUN: rm -rf %t.h %t_dir
