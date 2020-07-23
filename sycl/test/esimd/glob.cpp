// RUN: %clangxx -fsycl -fsycl-explicit-simd -c -fsycl-device-only -Xclang -emit-llvm %s -o - | \
// RUN:  FileCheck %s

// This test checks that globals with register attribute are allowed in ESIMD
// mode, can be accessed in functions and correct LLVM IR is generated
// (including translation of the register attribute)

#include <CL/sycl.hpp>
#include <CL/sycl/intel/esimd.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace sycl::intel::gpu;

constexpr unsigned VL = 16;

ESIMD_PRIVATE ESIMD_REGISTER(17) simd<int, VL> vc;
// CHECK-DAG: @vc = {{.+}} <16 x i32> zeroinitializer, align 64 #0
// CHECK-DAG: attributes #0 = { {{.*}}"VCByteOffset"="17" "VCGlobalVariable" "VCVolatile"{{.*}} }

ESIMD_PRIVATE ESIMD_REGISTER(17 + VL) simd<int, VL> vc1;
// CHECK-DAG: @vc1 = {{.+}} <16 x i32> zeroinitializer, align 64 #1
// CHECK-DAG: attributes #1 = { {{.*}}"VCByteOffset"="33" "VCGlobalVariable" "VCVolatile"{{.*}} }

SYCL_EXTERNAL ESIMD_NOINLINE void init_vc(int x) {
  vc1 = vc + 1;
  vc = x;
}
