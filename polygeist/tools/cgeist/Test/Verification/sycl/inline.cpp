// RUN: clang++ %s -O0 -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -mllvm -print-changed 2>&1 | FileCheck  %s

// COM: Ensure the 'always inline' pass trace is emitted.
// CHECK: *** IR Dump After AlwaysInlinerPass on [module] ***

SYCL_EXTERNAL extern "C" int __attribute__((always_inline)) callee() { return 10; }
SYCL_EXTERNAL extern "C" int caller() { return callee(); }
