// RUN: not cgeist -sycl-use-host-module=%S/host_module.mlir -O0 -w -o - %s 2>&1 | FileCheck %s

// CHECK: "-sycl-use-host-module" can only be used during SYCL device compilation

void do_nothing() {}
