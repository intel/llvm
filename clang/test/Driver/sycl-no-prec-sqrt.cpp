// REQUIRES: clang-driver

// RUN: %clang -### -fsycl \
// RUN:   -fsycl-fp32-prec-sqrt %s 2>&1 | FileCheck %s

// RUN: %clang -### -fsycl -fsycl-targets=spir64_gen \
// RUN:   -fsycl-fp32-prec-sqrt %s 2>&1 | FileCheck %s
//
// RUN: %clang -### -fsycl -fsycl-targets=spir64_x86_64 \
// RUN:   -fsycl-fp32-prec-sqrt %s 2>&1 | FileCheck %s
//
// RUN: %clang -### -fsycl -fsycl-targets=spir64_fpga \
// RUN:   -fsycl-fp32-prec-sqrt %s 2>&1 | FileCheck %s

// CHECK: warning: argument unused during compilation: '-fsycl-fp32-prec-sqrt'

void func(){};
