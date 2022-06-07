// RUN: %clangxx -DESIMD_XE_HPC -O0 -fsycl -c -Xclang -emit-llvm %s -o %t
// RUN: %clangxx -DESIMD_XE_HPC -O0 -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void foo();

class EsimdFunctor {
public:
  void operator()() __attribute__((sycl_explicit_simd)) { foo(); }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  EsimdFunctor esimdf;
  kernel<class kernel_esimd>(esimdf);
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void foo() {
  simd<short, 16> A_ACC = 7;
  simd<int, 128> A_ISRC1 = 0;
  simd<int, 8> A_ISRC2 = 0;
  simd<float, 16> A_DST =
      dpas<argument_type::BF16, argument_type::BF16, float, 8, 1>(
          A_ACC, A_ISRC1, A_ISRC2);
  // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16i16.v128i32.v8i32(<16 x i16> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 9, i32 9, i32 8, i32 1, i32 1, i32 1)

  simd<float, 16> B_ACC = 7;
  simd<int, 128> B_ISRC1 = 0;
  simd<int, 8> B_ISRC2 = 0;
  simd<float, 16> B_DST = dpas<argument_type::BF16, argument_type::BF16, 8, 1>(
      B_ACC, B_ISRC1, B_ISRC2);
  // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 9, i32 9, i32 8, i32 1, i32 1, i32 1)

  simd<int, 128> C_ISRC1 = 0;
  simd<int, 8> C_ISRC2 = 0;
  simd<float, 16> C_DST =
      dpas<argument_type::BF16, argument_type::BF16, 8, 1, float, int, int, 16>(
          C_ISRC1, C_ISRC2);
  // CHECK: call <16 x float> @llvm.genx.dpas.nosrc0.v16f32.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 {{[^,]+}})

  simd<float, 16> D_ACC = 7;
  simd<int, 128> D_ISRC1 = 0;
  simd<int, 8> D_ISRC2 = 0;
  simd<float, 16> D_DST = dpasw<argument_type::BF16, argument_type::BF16, 8, 1>(
      D_ACC, D_ISRC1, D_ISRC2);
  // CHECK: call <16 x float> @llvm.genx.dpasw.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 {{[^,]+}})

  simd<int, 128> E_ISRC1 = 0;
  simd<int, 8> E_ISRC2 = 0;
  simd<float, 16> E_DST = dpasw2<argument_type::BF16, argument_type::BF16, 8, 1,
                                 float, int, int, 16>(E_ISRC1, E_ISRC2);
  // CHECK: call <16 x float> @llvm.genx.dpasw.nosrc0.v16f32.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 {{[^,]+}})
}
