// The contents of this test checks for the following PVC_XT+ features:
//     * stochastic rounding (srnd)

// Checks host+device compilation
// RUN: %clangxx -fsycl -fsyntax-only %s

// Checks that lowerESIMD pass builds proper vc-intrinsics
// RUN: %clangxx -O0 -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// -O0 lowering, requires `-force-disable-esimd-opt` to disable all
// optimizations.
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -force-disable-esimd-opt -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_0.esimd.ll

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> foo();

// Note: lines #21-37 are needed to avoid sycl-post-link throwing away "foo"
// (SYCL_EXTERNAL) function.
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

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> foo() {
  simd<float, 16> F_srnd1 = 0;
  simd<float, 16> F_srnd2 = 0;
  simd<uint16_t, 16> rnd1 = 0;
  simd<uint8_t, 16> rnd2 = 0;

  simd<sycl::half, 16> HF_srnd = 0;
  simd<sycl::ext::intel::experimental::esimd::bf8, 16> BF8_srnd_out;
// for now, it seems that we support *half* type only for device target
#if defined(__SYCL_DEVICE_ONLY__)
  simd<_Float16, 16> HF_srnd1 = 0;
  simd<_Float16, 16> HF_srnd2 = 0;
  // fp32 (float) -> half
  // CHECK: call <16 x half> @llvm.genx.srnd.v16f16.v16f32.v16f32(<16 x float> {{[^,]+}}, <16 x float> {{[^)]+}})
  simd<_Float16, 16> F32_srnd_out1 =
      srnd<sycl::ext::intel::esimd::xmx::dpas_argument_type::fp16>(F_srnd1,
                                                                   F_srnd2);
  // half -> bf8
  // CHECK: call <16 x i8> @llvm.genx.srnd.v16i8.v16f16.v16f16(<16 x half> {{[^,]+}}, <16 x half> {{[^,]+}})
  simd<uint8_t, 16> BF8_srnd_out2 =
      srnd<sycl::ext::intel::esimd::xmx::dpas_argument_type::bf8>(HF_srnd1,
                                                                  HF_srnd2);
  // fp32 -> bf8 (emulated sequence, fp32 converted to half)
  // CHECK: call <16 x i8> @llvm.genx.srnd.v16i8.v16f16.v16f16(<16 x half> {{[^,]+}}, <16 x half> {{[^,]+}})
  simd<uint8_t, 16> BF8_srnd_out3 =
      srnd<sycl::ext::intel::esimd::xmx::dpas_argument_type::bf8>(F_srnd1,
                                                                  F_srnd2);
  // half -> bf8
  // CHECK: call <16 x i8> @llvm.genx.srnd.v16i8.v16f16.v16i8(<16 x half> {{[^,]+}}, <16 x i8> {{[^,]+}})
  BF8_srnd_out = srnd(HF_srnd, rnd2);

  // float -> bf8
  // CHECK: call <16 x i8> @llvm.genx.srnd.v16i8.v16f16.v16i8(<16 x half> {{[^,]+}}, <16 x i8> {{[^,]+}})
  BF8_srnd_out = srnd(F_srnd1, rnd2);

#endif
  return simd<float, 16>();
}

