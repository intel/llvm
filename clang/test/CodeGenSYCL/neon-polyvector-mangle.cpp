// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple aarch64-pc-windows-msvc -target-feature +neon \
// RUN:   -emit-llvm -o - %s | FileCheck %s

// The device triple is SPIR64, but Neon polyvector types use the AArch64
// mangling selected by the auxiliary host triple.
typedef unsigned long long poly64_t;
typedef __attribute__((neon_polyvector_type(2))) poly64_t poly64x2_t;

// Mark the helper as a device function so its mangling is emitted.
__attribute__((sycl_device)) void takes_poly64(poly64x2_t) {}

// CHECK: define{{.*}} @_Z12takes_poly6412__Poly64x2_t(
