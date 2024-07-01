// RUN: %clangxx -fsycl -c -Xclang -emit-llvm %s -o %t
// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

namespace xmx = sycl::ext::intel::esimd::xmx;

using bfloat16 = sycl::ext::oneapi::bfloat16;
using half = sycl::half;

constexpr auto bf16 = xmx::dpas_argument_type::bf16;
constexpr auto fp16 = xmx::dpas_argument_type::fp16;
constexpr auto s2 = xmx::dpas_argument_type::s2;
constexpr auto s8 = xmx::dpas_argument_type::s8;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void xmx_func();

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void xmx_func_end();

class EsimdFunctor {
public:
  void operator()() __attribute__((sycl_explicit_simd)) { xmx_func(); }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  EsimdFunctor esimdf;
  kernel<class kernel_esimd>(esimdf);
}

template <typename... T> SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void zoo(T... A);

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void xmx_func() {
  // DPAS: Result(M x N) = A(M x K) * B(K x N)
  // where:
  //   M = RepeatCount;
  //   K = SystolicDepth * OpsPerChannel;
  //   N = ExecutionSize, must be 16 on PVC and 8 on DG2.
  constexpr int M_one = 1;
  constexpr int K_half = 8 * 2;
  constexpr int K_bf16 = 8 * 2;
  constexpr int K_int8x2 = 8 * 4;
  constexpr int K_tf32 = 8 * 1;
  constexpr int N_pvc = 16;
  constexpr int N_dg2 = 8;

  // CHECK-LABEL: define dso_local spir_func void @_Z8xmx_funcv()

  { // ======= DPAS BF16 =======================================================
    simd<bfloat16, M_one * N_pvc> R_bf = 0;
    simd<float, M_one * N_pvc> R_f = 0;

    simd<bfloat16, M_one * N_pvc> C_bf = 0;
    simd<float, M_one * N_pvc> C_f = 0;

    simd<bfloat16, K_bf16 * N_pvc> B_bf = 0;
    simd<bfloat16, M_one * K_bf16> A_bf = 0;

    R_f = xmx::dpas<8, 1, float>(C_f, B_bf, A_bf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 9, i32 9, i32 8, i32 1, i32 1, i32 1)

    R_f = xmx::dpas<8, 1, float>(C_bf, B_bf, A_bf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16i16.v128i32.v8i32(<16 x i16> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 9, i32 9, i32 8, i32 1, i32 1, i32 0)

    R_bf = xmx::dpas<8, 1, bfloat16>(C_f, B_bf, A_bf);
    zoo(R_bf);
    // CHECK: call <16 x i16> @llvm.genx.dpas2.v16i16.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 9, i32 9, i32 8, i32 1, i32 0, i32 1)

    R_bf = xmx::dpas<8, 1, bfloat16>(C_bf, B_bf, A_bf);
    zoo(R_bf);
    // CHECK: call <16 x i16> @llvm.genx.dpas2.v16i16.v16i16.v128i32.v8i32(<16 x i16> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 9, i32 9, i32 8, i32 1, i32 0, i32 0)

    R_f = xmx::dpas<8, 1, float>(B_bf, A_bf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas.nosrc0.v16f32.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 17303817)

    R_bf = xmx::dpas<8, 1, bfloat16>(B_bf, A_bf);
    zoo(R_bf);
    // CHECK: call <16 x i16> @llvm.genx.dpas.nosrc0.v16i16.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 17303817)
  }

  { // ======= DPAS FP16 =======================================================
    simd<half, M_one * N_pvc> R_hf = 0;
    simd<float, M_one * N_pvc> R_f = 0;

    simd<half, M_one * N_pvc> C_hf = 0;
    simd<float, M_one * N_pvc> C_f = 0;

    simd<half, K_half * N_pvc> B_hf = 0;
    simd<half, M_one * K_half> A_hf = 0;

    // ------------------- FP16: WITH ACC OPERAND -----------------------
    R_f = xmx::dpas<8, 1, float>(C_f, B_hf, A_hf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 10, i32 10, i32 8, i32 1, i32 1, i32 1)

    R_f = xmx::dpas<8, 1, float>(C_hf, B_hf, A_hf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16f16.v128i32.v8i32(<16 x half> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 10, i32 10, i32 8, i32 1, i32 1, i32 0)

    R_hf = xmx::dpas<8, 1, half>(C_f, B_hf, A_hf);
    zoo(R_hf);
    // CHECK: call <16 x half> @llvm.genx.dpas2.v16f16.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 10, i32 10, i32 8, i32 1, i32 0, i32 1)

    R_hf = xmx::dpas<8, 1, half>(C_hf, B_hf, A_hf);
    zoo(R_hf);
    // CHECK: call <16 x half> @llvm.genx.dpas2.v16f16.v16f16.v128i32.v8i32(<16 x half> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 10, i32 10, i32 8, i32 1, i32 0, i32 0)

    // ------------------- FP16: NO ACC OPERAND -----------------------
    R_f = xmx::dpas<8, 1, float>(B_hf, A_hf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas.nosrc0.v16f32.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 17304074)

    R_hf = xmx::dpas<8, 1, half>(B_hf, A_hf);
    zoo(R_hf);
    // CHECK: call <16 x half> @llvm.genx.dpas.nosrc0.v16f16.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 17304074)
  }

  { // ======= DPAS 8-BIT x 2-BIT INT ==========================================
    simd<int, M_one * N_pvc> R_d = 0;
    simd<int, M_one * N_pvc> C_d = 0;
    simd<int, K_int8x2 * N_pvc / 16> B_int2 = 0; // 16 2-bit integers per int32
    simd<signed char, M_one * K_int8x2> A_int8 = 0;

    // ------------ DPAS s8 x s2: WITH THE ACCUMULATOR OPERAND -----------------
    R_d = xmx::dpas<8, 1, int, int, int, signed char, s2, s8>(C_d, B_int2,
                                                              A_int8);
    zoo(R_d);
    // CHECK: call <16 x i32> @llvm.genx.dpas2.v16i32.v16i32.v32i32.v8i32(<16 x i32> {{[^,]+}}, <32 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 4, i32 8, i32 8, i32 1, i32 1, i32 1)

    // ------------ DPAS s8 x s2: WITHOUT THE ACCUMULATOR OPERAND --------------
    R_d = xmx::dpas<8, 1, int, int, signed char, s2, s8>(B_int2, A_int8);
    zoo(R_d);
    // CHECK: call <16 x i32> @llvm.genx.dpas.nosrc0.v16i32.v32i32.v8i32(<32 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 17303556)
  }

  { // ======= DPASW BF16 ======================================================
    simd<float, M_one * N_dg2> R_f = 0;
    simd<float, M_one * N_dg2> C_f = 0;

    simd<bfloat16, K_bf16 * N_dg2> B_bf = 0;
    simd<bfloat16, M_one * K_bf16 / 2> A_bf = 0;

    // ------------ DPASW BF16: WITH THE ACCUMULATOR OPERAND -------------------
    R_f = xmx::dpasw<8, 1, float>(C_f, B_bf, A_bf);
    zoo(R_f);
    // CHECK: call <8 x float> @llvm.genx.dpasw.v8f32.v64i32.v4i32(<8 x float> {{[^,]+}}, <64 x i32> {{[^,]+}}, <4 x i32> {{[^,]+}}, i32 17303817)

    // ------------ DPASW BF16: WITHOUT ACC OPERAND ----------------------------
    R_f = xmx::dpasw<8, 1, float>(B_bf, A_bf);
    zoo(R_f);
    // CHECK: call <8 x float> @llvm.genx.dpasw.nosrc0.v8f32.v64i32.v4i32(<64 x i32> {{[^,]+}}, <4 x i32> {{[^,]+}}, i32 17303817)
  }

  { // ======= DPASW FP16 ======================================================
    simd<float, M_one * N_dg2> R_f = 0;
    simd<float, M_one * N_dg2> C_f = 0;

    simd<half, K_half * N_dg2> B_hf = 0;
    simd<half, M_one * K_half / 2> A_hf = 0;

    // ------------ DPASW FP16: WITH THE ACCUMULATOR OPERAND -------------------
    R_f = xmx::dpasw<8, 1, float>(C_f, B_hf, A_hf);
    zoo(R_f);
    // CHECK: call <8 x float> @llvm.genx.dpasw.v8f32.v64i32.v4i32(<8 x float> {{[^,]+}}, <64 x i32> {{[^,]+}}, <4 x i32> {{[^,]+}}, i32 17304074)

    // ------------ DPASW FP16: WITHOUT ACC OPERAND ----------------------------
    R_f = xmx::dpasw<8, 1, float>(B_hf, A_hf);
    zoo(R_f);
    // CHECK: call <8 x float> @llvm.genx.dpasw.nosrc0.v8f32.v64i32.v4i32(<64 x i32> {{[^,]+}}, <4 x i32> {{[^,]+}}, i32 17304074)
  }

  { // ======= DPAS TFLOAT32 ===================================================
    simd<float, M_one * N_pvc> R_f = 0;
    simd<float, M_one * N_pvc> C_f = 0;

    simd<sycl::ext::intel::experimental::esimd::tfloat32, K_tf32 * N_pvc> B_tf =
        0;
    simd<sycl::ext::intel::experimental::esimd::tfloat32, M_one * K_tf32> A_tf =
        0;

    // ------------------- TFLOAT32: WITH ACC OPERAND --------------------------
    R_f = xmx::dpas<8, 1, float>(C_f, B_tf, A_tf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas2.v16f32.v16f32.v128i32.v8i32(<16 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 12, i32 12, i32 8, i32 1, i32 1, i32 1)

    // ------------------- TFLOAT32: NO ACC OPERAND ----------------------------
    R_f = xmx::dpas<8, 1, float>(B_tf, A_tf);
    zoo(R_f);
    // CHECK: call <16 x float> @llvm.genx.dpas.nosrc0.v16f32.v128i32.v8i32(<128 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, i32 17304588)
  }

  xmx_func_end();
  // CHECK-LABEL: call spir_func void @_Z12xmx_func_endv()
}
