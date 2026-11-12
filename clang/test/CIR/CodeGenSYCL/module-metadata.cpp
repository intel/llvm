// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-SPIRV
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-SPIRV
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,LLVM-SPIRV

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -DESIMD -fclangir -emit-cir %s -o %t-esimd.cir
// RUN: FileCheck --input-file=%t-esimd.cir %s -check-prefixes=CIR,CIR-ESIMD
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -DESIMD -fclangir -emit-llvm %s -o %t-esimd-cir.ll
// RUN: FileCheck --input-file=%t-esimd-cir.ll %s --check-prefixes=LLVM,LLVM-ESIMD
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -DESIMD -emit-llvm %s -o %t-esimd.ll
// RUN: FileCheck --input-file=%t-esimd.ll %s --check-prefixes=LLVM,LLVM-ESIMD

// Check the SYCL device module metadata: the "sycl-device" module flag, the
// SPIR version and the SPIR-V source language (OpenCL C++, or ESIMD when any
// explicit SIMD function is emitted).

#ifdef ESIMD
[[intel::sycl_explicit_simd]]
#endif
[[clang::sycl_external]] void foo() {}

// CIR: module {{.*}} attributes {
// CIR-SAME: cir.cl.spir.version = #cir.cl.version<1, 2>
// CIR-SPIRV-SAME: cir.spirv.source = array<i32: 4, 100000>
// CIR-ESIMD-SAME: cir.spirv.source = array<i32: 0, 100000>
// CIR-SAME: cir.sycl_device

// LLVM: !llvm.module.flags = !{[[FLAG:![0-9]+]]{{.*}}}
// LLVM: !opencl.spir.version = !{[[SPIRVER:![0-9]+]]}
// LLVM: !spirv.Source = !{[[SRC:![0-9]+]]}
// LLVM: [[FLAG]] = !{i32 1, !"sycl-device", i32 1}
// LLVM: [[SPIRVER]] = !{i32 1, i32 2}
// LLVM-SPIRV: [[SRC]] = !{i32 4, i32 100000}
// LLVM-ESIMD: [[SRC]] = !{i32 0, i32 100000}
