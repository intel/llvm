// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// When SYCL device code is compiled with a Windows-MSVC host, the device
// target (spir64) defines _M_X64 so that MSVC STL headers take the x86
// intrinsics path. The device target feature set must correspondingly carry
// sse/sse2 so that function-level __target__ attributes (e.g. VS2026
// <complex>'s [[gnu::target("fma")]] on _Sqr_error_x86_x64_fma) don't strip
// the baseline, which would break intrinsic calls like _mm_set_sd / _mm_store_sd.

#include "Inputs/sycl.hpp"

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) { h.single_task<class TestK>([=] {}); });
  return 0;
}

// CHECK: spir_kernel void @{{.*}}TestK{{.*}}() [[ATTRS:#[0-9]+]]
// CHECK: attributes [[ATTRS]] = {{.*}}"target-features"="+sse,+sse2"
