// RUN: not cgeist %s 2>&1 | FileCheck %s

// CHECK: error PREPROCESSOR ERROR
// CHECK: 1 error(s) generated

#error PREPROCESSOR ERROR
