// RUN: %clangxx -fsycl -fsycl-device-only -flegacy-pass-manager -S -emit-llvm -x c++ %s -o %t-lgcy
// RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %t-lgcy -o %t-lgcy.table
// RUN: FileCheck %s -input-file=%t-lgcy_esimd_0.ll

// RUN: %clangxx -fsycl -fsycl-device-only -fno-legacy-pass-manager -S -emit-llvm -x c++ %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that unreferenced SYCL_EXTERNAL functions are not dropped
// from the module and go through sycl-post-link. This test also checks that
// ESIMD lowering happens for such functions as well.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

constexpr unsigned VL = 8;
using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;
extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void vmult2(simd<float, VL> a) {
  int i = __spirv_GlobalInvocationId_x();
  a *= i;
}

// CHECK: define dso_local spir_func void @vmult2
// CHECK:   call <3 x i32> @llvm.genx.local.id.v3i32()
// CHECK:   call <3 x i32> @llvm.genx.local.size.v3i32()
// CHECK:   call i32 @llvm.genx.group.id.x()
// CHECK:   ret void
// CHECK: }
