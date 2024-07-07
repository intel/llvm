// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

// Tests invoke_simd support in the compiler/headers
/* This program is basically an extension of the standard vector addition
 * program (i.e., call_vadd_1d). The only difference is that instead of adding 2
 * vectors (i.e., va + vb), we add 14 vectors. There are 2 cases in the test:
 * 1. The extra vectors beyond va and vb are zero vectors
 * 2. The extra vectors beyond va and vb are non-zero vectors
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

/* Subgroup size attribute is optional
 * In case it is absent compiler decides what subgroup size to use
 */
#ifdef IMPL_SUBGROUP
#define SUBGROUP_ATTR
#else
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(VL)]]
#endif

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

template <int VL>
__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE_doVadd(esimd::simd<float, VL> va, esimd::simd<float, VL> vb,
                    esimd::simd<float, VL> va1, esimd::simd<float, VL> vb1,
                    esimd::simd<float, VL> va2, esimd::simd<float, VL> vb2,
                    esimd::simd<float, VL> va3, esimd::simd<float, VL> vb3,
                    esimd::simd<float, VL> va4, esimd::simd<float, VL> vb4,
                    esimd::simd<float, VL> va5, esimd::simd<float, VL> vb5,
                    esimd::simd<float, VL> va6,
                    esimd::simd<float, VL> vb6) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> vc = va + vb + va1 + vb1 + va2 + vb2 + va3 + vb3 +
                              va4 + vb4 + va5 + vb5 + va6 + vb6;
  return vc;
}

template <int VL>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_doVadd(
        simd<float, VL> va, simd<float, VL> vb, simd<float, VL> va1,
        simd<float, VL> vb1, simd<float, VL> va2, simd<float, VL> vb2,
        simd<float, VL> va3, simd<float, VL> vb3, simd<float, VL> va4,
        simd<float, VL> vb4, simd<float, VL> va5, simd<float, VL> vb5,
        simd<float, VL> va6, simd<float, VL> vb6) SYCL_ESIMD_FUNCTION;

float SPMD_CALLEE_doVadd(float va, float vb, float vp, float vq, float vr,
                         float vx, float vy, float vz, float v1, float v2,
                         float v3, float v4, float v5, float v6) {
  return va + vb + vp + vq + vr + vx + vy + vz + v1 + v2 + v3 + v4 + v5 + v6;
}

using namespace sycl;

template <int, int, bool, int> class TestID;

template <int Size, int VL, bool use_invoke_simd, int CaseNum, class QueueTY>
bool test(QueueTY q, float *A, float *B, float *C, float *P, float *Q, float *R,
          float *X, float *Y, float *Z) {
  std::cout << "Case #" << CaseNum << '\n';

  try {
    buffer<float, 1> bufa(A, range<1>(Size));
    buffer<float, 1> bufb(B, range<1>(Size));
    buffer<float, 1> bufc(C, range<1>(Size));
    buffer<float, 1> bufp(P, range<1>(Size));
    buffer<float, 1> bufq(Q, range<1>(Size));
    buffer<float, 1> bufr(R, range<1>(Size));
    buffer<float, 1> bufx(X, range<1>(Size));
    buffer<float, 1> bufy(Y, range<1>(Size));
    buffer<float, 1> bufz(Z, range<1>(Size));

    sycl::range<1> GlobalRange{Size};
    sycl::range<1> LocalRange{VL};

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PP = bufp.get_access<access::mode::read>(cgh);
      auto PQ = bufq.get_access<access::mode::read>(cgh);
      auto PR = bufr.get_access<access::mode::read>(cgh);
      auto PX = bufx.get_access<access::mode::read>(cgh);
      auto PY = bufy.get_access<access::mode::read>(cgh);
      auto PZ = bufz.get_access<access::mode::read>(cgh);

      auto PC = bufc.get_access<access::mode::write>(cgh);

      cgh.parallel_for<TestID<Size, VL, use_invoke_simd, CaseNum>>(
          nd_range<1>(GlobalRange, LocalRange),
          [=](nd_item<1> item) SUBGROUP_ATTR {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            unsigned int offset = g.get_group_id() * g.get_local_range() +
                                  sg.get_group_id() * sg.get_max_local_range();
            float va = sg.load(
                PA.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vb = sg.load(
                PB.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vp = sg.load(
                PP.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vq = sg.load(
                PQ.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vr = sg.load(
                PR.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vx = sg.load(
                PX.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vy = sg.load(
                PY.get_multi_ptr<access::decorated::yes>().get() + offset);
            float vz = sg.load(
                PZ.get_multi_ptr<access::decorated::yes>().get() + offset);

            float vc;

            if constexpr (use_invoke_simd) {
              vc = invoke_simd(sg, SIMD_CALLEE_doVadd<VL>, va, vb, vx, vy, vx,
                               vy, vx, vy, vx, vy, vp, vq, vr, vz);
            } else {
              vc = SPMD_CALLEE_doVadd(va, vb, vx, vy, vx, vy, vx, vy, vx, vy,
                                      vp, vq, vr, vz);
            }
            sg.store(PC.get_multi_ptr<access::decorated::yes>().get() + offset,
                     vc);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] + X[i] + Y[i] + X[i] + Y[i] + X[i] + Y[i] + X[i] + Y[i] +
            P[i] + Q[i] + R[i] + Z[i] !=
        C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << " + " << X[i] << " + " << Y[i] << " + "
                  << X[i] << " + " << Y[i] << " + " << X[i] << " + " << Y[i]
                  << " + " << X[i] << " + " << Y[i] << " + " << P[i] << " + "
                  << Q[i] << " + " << R[i] << " + " << Z[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt == 0;
}

int main() {
  constexpr int VL = 16;
  constexpr int Size = 1024;
  constexpr bool use_invoke_simd = true;

  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;

  float *A = new float[Size];
  float *B = new float[Size];
  float *C = new float[Size];
  float *P = new float[Size];
  float *Q = new float[Size];
  float *R = new float[Size];
  float *X = new float[Size];
  float *Y = new float[Size];
  float *Z = new float[Size];

  // Case #1 - zero vectors except A and B
  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    // Basically, we can ignore these extra zero vectors.
    P[i] = Q[i] = R[i] = X[i] = Y[i] = Z[i] = 0;
    C[i] = -1.0f;
  }
  passed &= test<Size, VL, use_invoke_simd, 1>(q, A, B, C, P, Q, R, X, Y, Z);

  // Case #2 - all nonzero vectors
  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    // In this non-zero version, these extra vectors are ones vectors.
    P[i] = Q[i] = R[i] = X[i] = Y[i] = Z[i] = 1.0f;
    C[i] = -1.0f;
  }
  passed &= test<Size, VL, use_invoke_simd, 2>(q, A, B, C, P, Q, R, X, Y, Z);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] P;
  delete[] Q;
  delete[] R;
  delete[] X;
  delete[] Y;
  delete[] Z;

  return passed ? 0 : 1;
}

template <int VL>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_doVadd(
        simd<float, VL> va, simd<float, VL> vb, simd<float, VL> va1,
        simd<float, VL> vb1, simd<float, VL> va2, simd<float, VL> vb2,
        simd<float, VL> va3, simd<float, VL> vb3, simd<float, VL> va4,
        simd<float, VL> vb4, simd<float, VL> va5, simd<float, VL> vb5,
        simd<float, VL> va6, simd<float, VL> vb6) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> vc = ESIMD_CALLEE_doVadd<VL>(
      va, vb, va1, vb1, va2, vb2, va3, vb3, va4, vb4, va5, vb5, va6, vb6);
  return vc;
}
