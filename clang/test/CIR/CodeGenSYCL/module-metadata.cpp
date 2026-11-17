// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR --implicit-check-not=sycl_explicit_simd
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fclangir -emit-llvm -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-SPIRV --implicit-check-not=sycl_explicit_simd
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,LLVM-SPIRV --implicit-check-not=sycl_explicit_simd

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -DESIMD -fclangir -emit-cir %s -o %t-esimd.cir
// RUN: FileCheck --input-file=%t-esimd.cir %s -check-prefixes=CIR,CIR-ESIMD
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -DESIMD -fclangir -emit-llvm -disable-llvm-passes %s -o %t-esimd-cir.ll
// RUN: FileCheck --input-file=%t-esimd-cir.ll %s --check-prefixes=LLVM,LLVM-ESIMD
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -DESIMD -emit-llvm -disable-llvm-passes %s -o %t-esimd.ll
// RUN: FileCheck --input-file=%t-esimd.ll %s --check-prefixes=LLVM,LLVM-ESIMD

// Resuming compilation from the serialized CIR must give the same result.
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -x cir -emit-llvm -disable-llvm-passes %t-esimd.cir -o %t-esimd-from-cir.ll
// RUN: FileCheck --input-file=%t-esimd-from-cir.ll %s --check-prefixes=LLVM,LLVM-ESIMD

// Check the SYCL device metadata: the "sycl-device" module flag, the SPIR
// version, the SPIR-V source language (OpenCL C++, or ESIMD when any explicit
// SIMD function is present) and the !sycl_explicit_simd function metadata.

#ifdef ESIMD
[[intel::sycl_explicit_simd]]
#endif
[[clang::sycl_external]] void foo() {}

// CIR: module {{.*}} attributes {
// CIR-SAME: cir.cl.spir.version = #cir.cl.version<1, 2>
// CIR-SAME: cir.sycl_device

// CIR-ESIMD: cir.func {{.*}}@_Z3foov() {{.*}} attributes {{{.*}}cir.sycl_explicit_simd{{.*}}} {

// LLVM-ESIMD: define {{.*}}void @_Z3foov() {{.*}}!sycl_explicit_simd [[ESIMD:![0-9]+]]

// LLVM: !llvm.module.flags = !{[[FLAG:![0-9]+]]{{.*}}}
// LLVM: !opencl.spir.version = !{[[SPIRVER:![0-9]+]]}
// LLVM: !spirv.Source = !{[[SRC:![0-9]+]]}
// LLVM: [[FLAG]] = !{i32 1, !"sycl-device", i32 1}
// LLVM: [[SPIRVER]] = !{i32 1, i32 2}
// LLVM-SPIRV: [[SRC]] = !{i32 4, i32 100000}
// LLVM-ESIMD: [[SRC]] = !{i32 0, i32 100000}
// LLVM-ESIMD: [[ESIMD]] = !{}
