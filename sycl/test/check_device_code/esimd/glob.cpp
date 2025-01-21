// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that globals with register attribute are allowed in ESIMD
// mode, can be accessed in functions and correct LLVM IR is generated
// (including translation of the register attribute)

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

constexpr unsigned VL = 16;

ESIMD_PRIVATE ESIMD_REGISTER(17) simd<int, VL> vc;
// CHECK-DAG: @vc = {{.+}} <16 x i32> zeroinitializer, align 64 #0
// CHECK-DAG: attributes #0 = { {{.*}}"VCByteOffset"="17" "VCGlobalVariable"
// "VCVolatile"{{.*}} }

ESIMD_PRIVATE ESIMD_REGISTER(17 + VL) simd<int, VL> vc1;
// CHECK-DAG: @vc1 = {{.+}} <16 x i32> zeroinitializer, align 64 #1
// CHECK-DAG: attributes #1 = { {{.*}}"VCByteOffset"="33" "VCGlobalVariable"
// "VCVolatile"{{.*}} }

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL ESIMD_NOINLINE void init_vc(int x) {
  vc1 = vc + 1;
  vc = x;
}

void caller(int x) {
  kernel<class kernel_esimd>([=]() SYCL_ESIMD_KERNEL { init_vc(x); });
}
