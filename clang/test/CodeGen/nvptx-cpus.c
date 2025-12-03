// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_20 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_21 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_30 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_35 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_37 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_50 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_52 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_53 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_60 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_61 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_62 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_70 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_72 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_75 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_80 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_86 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_87 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_89 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_90 -O3 -o %t %s -emit-llvm
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_90a -O3 -o %t %s -emit-llvm

// Make sure clang accepts all supported architectures.

void foo(float* a,
         float* b) {
  a[0] = b[0];
}
