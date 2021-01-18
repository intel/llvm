// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice -fsycl -fsycl-is-device -fsycl-explicit-simd -S -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-ESIMD

__attribute__((sycl_device)) void funcWithSpirvIntrin() {}
__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void standaloneCmFunc() { funcWithSpirvIntrin(); }

// CHECK-ESIMD-DAG: define {{.*}}spir_func void @{{.*}}funcWithSpirvIntrinv() #{{[0-9]+}} !sycl_explicit_simd !{{[0-9]+}} {
// CHECK-ESIMD-DAG: define {{.*}}spir_func void @{{.*}}standaloneCmFuncv() #{{[0-9]+}} !sycl_explicit_simd !{{[0-9]+}} {
