//===---joint_matrix_hip_small_float.hpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Helpers for testing joint_matrix with the OCP FP8 (fp8_e4m3, fp8_e5m2) and
// FP4 (fp4_e2m1) multiplicand types on AMD gfx950.
//
// The device code only moves the raw encodings through joint_matrix_load /
// fill / copy / mad; all conversions happen on the host. References are
// computed from an independent decoder of the OCP formats, which is also
// checked against the host conversions of the SYCL types.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::experimental::fp4_e2m1;
using sycl::ext::oneapi::experimental::fp8_e4m3;
using sycl::ext::oneapi::experimental::fp8_e5m2;

template <typename T> constexpr int num_encodings() {
  return std::is_same_v<T, fp4_e2m1> ? 16 : 256;
}

template <typename T> const char *type_name() {
  if constexpr (std::is_same_v<T, fp8_e4m3>)
    return "fp8_e4m3";
  else if constexpr (std::is_same_v<T, fp8_e5m2>)
    return "fp8_e5m2";
  else
    return "fp4_e2m1";
}

template <typename T> T from_bits(uint8_t Bits) {
  T V;
  V.vals[0] = Bits;
  return V;
}

// Reference decoder of the OCP E4M3, E5M2 and E2M1 encodings.
template <typename T> double ocp_decode(uint8_t Bits) {
  const double Sign =
      (Bits & (std::is_same_v<T, fp4_e2m1> ? 0x8 : 0x80)) ? -1.0 : 1.0;
  if constexpr (std::is_same_v<T, fp8_e4m3>) {
    const int E = (Bits >> 3) & 0xF, Mant = Bits & 0x7;
    if (E == 0xF && Mant == 0x7)
      return std::numeric_limits<double>::quiet_NaN();
    if (E == 0)
      return Sign * std::ldexp(Mant / 8.0, -6);
    return Sign * std::ldexp(1.0 + Mant / 8.0, E - 7);
  } else if constexpr (std::is_same_v<T, fp8_e5m2>) {
    const int E = (Bits >> 2) & 0x1F, Mant = Bits & 0x3;
    if (E == 0x1F)
      return Mant == 0 ? Sign * std::numeric_limits<double>::infinity()
                       : std::numeric_limits<double>::quiet_NaN();
    if (E == 0)
      return Sign * std::ldexp(Mant / 4.0, -14);
    return Sign * std::ldexp(1.0 + Mant / 4.0, E - 15);
  } else {
    const int E = (Bits >> 1) & 0x3, Mant = Bits & 0x1;
    if (E == 0)
      return Sign * Mant * 0.5;
    return Sign * std::ldexp(1.0 + Mant / 2.0, E - 1);
  }
}

template <typename T> bool is_finite_encoding(uint8_t Bits) {
  return std::isfinite(ocp_decode<T>(Bits));
}

template <typename T> uint8_t ocp_encode_exact(double V) {
  for (int B = 0; B < num_encodings<T>(); ++B)
    if (ocp_decode<T>(B) == V && !(V == 0.0 && std::signbit(ocp_decode<T>(B))))
      return static_cast<uint8_t>(B);
  assert(false && "value is not exactly representable");
  return 0;
}

// The SYCL host conversions must agree with the reference decoder for every
// finite encoding, otherwise the references below would not be meaningful.
template <typename T> void check_host_decoding() {
  for (int B = 0; B < num_encodings<T>(); ++B) {
    const double Ref = ocp_decode<T>(B);
    const float Host = static_cast<float>(from_bits<T>(B));
    if (std::isnan(Ref))
      assert(std::isnan(Host));
    else
      assert(static_cast<double>(Host) == Ref && "host decoding mismatch");
  }
}

// Computes D = C + A * B with joint_matrix, where A is M x (K * KX) and B is
// (K * KX) x N, both given as row-major logical encodings and stored in memory
// following AL/BL. C and the returned D are row-major logical M x N, while D
// is stored by the device following CL.
template <typename TA, typename TB, size_t M, size_t N, size_t K, size_t KX,
          layout AL, layout BL, layout CL>
std::vector<float> hip_small_float_gemm(const std::vector<uint8_t> &ABits,
                                        const std::vector<uint8_t> &BBits,
                                        const std::vector<float> &C) {
  constexpr size_t KT = K * KX;
  std::vector<TA> AMem(M * KT);
  std::vector<TB> BMem(KT * N);
  for (size_t m = 0; m < M; ++m)
    for (size_t k = 0; k < KT; ++k)
      AMem[AL == layout::row_major ? m * KT + k : k * M + m] =
          from_bits<TA>(ABits[m * KT + k]);
  for (size_t k = 0; k < KT; ++k)
    for (size_t n = 0; n < N; ++n)
      BMem[BL == layout::row_major ? k * N + n : n * KT + k] =
          from_bits<TB>(BBits[k * N + n]);
  std::vector<float> CMem(C), DMem(M * N, 0.f);

  {
    queue Q;
    buffer<TA> BufA(AMem.data(), range{M * KT});
    buffer<TB> BufB(BMem.data(), range{KT * N});
    buffer<float> BufC(CMem.data(), range{M * N});
    buffer<float> BufD(DMem.data(), range{M * N});
    Q.submit([&](handler &CGH) {
       accessor AccA{BufA, CGH, read_only};
       accessor AccB{BufB, CGH, read_only};
       accessor AccC{BufC, CGH, read_only};
       accessor AccD{BufD, CGH, write_only};
       CGH.parallel_for(nd_range<2>{{4, 16}, {4, 16}}, [=](nd_item<2> It) {
         auto SG = It.get_sub_group();
         joint_matrix<sub_group, float, use::accumulator, M, N> SubC;
         joint_matrix<sub_group, TA, use::a, M, K, AL> SubA;
         joint_matrix<sub_group, TB, use::b, K, N, BL> SubB;
         joint_matrix_load(
             SG, SubC, AccC.template get_multi_ptr<access::decorated::yes>(), N,
             layout::row_major);
         for (size_t KX0 = 0; KX0 < KX; ++KX0) {
           const size_t AOff = AL == layout::row_major ? KX0 * K : KX0 * K * M;
           const size_t ALd = AL == layout::row_major ? KT : M;
           const size_t BOff = BL == layout::row_major ? KX0 * K * N : KX0 * K;
           const size_t BLd = BL == layout::row_major ? N : KT;
           joint_matrix_load(
               SG, SubA,
               AccA.template get_multi_ptr<access::decorated::yes>() + AOff,
               ALd);
           joint_matrix_load(
               SG, SubB,
               AccB.template get_multi_ptr<access::decorated::yes>() + BOff,
               BLd);
           joint_matrix_mad(SG, SubC, SubA, SubB, SubC);
         }
         joint_matrix_store(
             SG, SubC, AccD.template get_multi_ptr<access::decorated::yes>(), N,
             CL);
       });
     }).wait();
  }

  std::vector<float> D(M * N);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n)
      D[m * N + n] = DMem[CL == layout::row_major ? m * N + n : n * M + m];
  return D;
}

// Exact check: every multiplicand is in {-2, -1, -0.5, 0, 0.5, 1, 2}, which is
// representable in all three formats, and C holds small integers, so every
// partial sum is exactly representable in fp32 and D must match bit-for-bit.
template <typename TA, typename TB, size_t M, size_t N, size_t K, size_t KX,
          layout AL, layout BL, layout CL>
void hip_small_float_mfma_exact() {
  constexpr size_t KT = K * KX;
  constexpr double Vals[] = {-2, -1, -0.5, 0, 0.5, 1, 2};
  std::mt19937 Gen(M * 1000 + K * 10 + KX);
  std::uniform_int_distribution<int> Pick(0, 6), PickC(-8, 8);
  std::vector<double> A(M * KT), B(KT * N);
  std::vector<uint8_t> ABits(M * KT), BBits(KT * N);
  std::vector<float> C(M * N);
  for (size_t i = 0; i < M * KT; ++i) {
    A[i] = Vals[Pick(Gen)];
    ABits[i] = ocp_encode_exact<TA>(A[i]);
  }
  for (size_t i = 0; i < KT * N; ++i) {
    B[i] = Vals[Pick(Gen)];
    BBits[i] = ocp_encode_exact<TB>(B[i]);
  }
  for (auto &V : C)
    V = static_cast<float>(PickC(Gen));

  auto D =
      hip_small_float_gemm<TA, TB, M, N, K, KX, AL, BL, CL>(ABits, BBits, C);
  int Errors = 0;
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      double Ref = C[m * N + n];
      for (size_t k = 0; k < KT; ++k)
        Ref += A[m * KT + k] * B[k * N + n];
      if (D[m * N + n] != static_cast<float>(Ref) && Errors++ < 5)
        printf("exact %s x %s %zux%zux%zu KX=%zu (%d,%d,%d): D[%zu][%zu] = %g, "
               "expected %g\n",
               type_name<TA>(), type_name<TB>(), M, N, K, KX, int(AL), int(BL),
               int(CL), m, n, D[m * N + n], Ref);
    }
  assert(Errors == 0 && "exact small-float joint_matrix_mad mismatch");
}

// Full-range check: random finite encodings spanning every exponent, including
// subnormals and the format maxima. The reference is computed in double and
// compared with a tolerance relative to the magnitude of the partial products.
template <typename TA, typename TB, size_t M, size_t N, size_t K, size_t KX>
void hip_small_float_mfma_full_range() {
  constexpr size_t KT = K * KX;
  std::mt19937 Gen(7 + M + K);
  std::uniform_int_distribution<int> PickA(0, num_encodings<TA>() - 1),
      PickB(0, num_encodings<TB>() - 1);
  std::uniform_real_distribution<float> PickC(-100.f, 100.f);
  std::vector<uint8_t> ABits(M * KT), BBits(KT * N);
  std::vector<float> C(M * N);
  for (auto &Bits : ABits)
    do
      Bits = PickA(Gen);
    while (!is_finite_encoding<TA>(Bits));
  for (auto &Bits : BBits)
    do
      Bits = PickB(Gen);
    while (!is_finite_encoding<TB>(Bits));
  for (auto &V : C)
    V = PickC(Gen);

  auto D = hip_small_float_gemm<TA, TB, M, N, K, KX, layout::row_major,
                                layout::row_major, layout::row_major>(ABits,
                                                                      BBits, C);
  int Errors = 0;
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      double Ref = C[m * N + n], Mag = std::fabs(C[m * N + n]);
      for (size_t k = 0; k < KT; ++k) {
        const double P = ocp_decode<TA>(ABits[m * KT + k]) *
                         ocp_decode<TB>(BBits[k * N + n]);
        Ref += P;
        Mag += std::fabs(P);
      }
      // The MFMA accumulates the products with somewhat less precision than
      // sequential IEEE fp32 additions (errors around 2^-13 of the magnitude
      // are observed), so the bound is relative to the sum of magnitudes.
      if (std::fabs(D[m * N + n] - Ref) > 0x1p-11 * Mag && Errors++ < 5)
        printf("full-range %s x %s %zux%zux%zu: D[%zu][%zu] = %.9g, expected "
               "%.9g (sum of magnitudes %.9g)\n",
               type_name<TA>(), type_name<TB>(), M, N, K, m, n, D[m * N + n],
               Ref, Mag);
    }
  assert(Errors == 0 && "full-range small-float joint_matrix_mad mismatch");
}

// Format check: multiplies every encoding of T by an exact one-hot operand of
// the other type, so each output element is the hardware's decoding of a
// single encoding. Finite encodings must decode to their OCP value; non-finite
// ones (NaN, and Inf * 0) must turn their whole output row/column into NaN.
// FNUZ-style decoding (as on gfx94x) would fail both checks, e.g. 0x7F decodes
// to 240 in E4M3FNUZ and 0x80 is NaN.
template <typename T, typename TOther, bool TIsA, size_t M, size_t N, size_t K>
void hip_small_float_format() {
  static_assert(M == N);
  // Number of one-hot columns of the identity-like operand.
  constexpr size_t P = N < K ? N : K;
  const uint8_t OneBits = ocp_encode_exact<TOther>(1.0);

  // Encoding of the value at output (i, j), j < P: one encoding per output
  // element, cycling through all the encodings of T.
  auto EncodingAt = [](size_t i, size_t j, size_t Pass) {
    return static_cast<uint8_t>((Pass * M * P + i * P + j) %
                                num_encodings<T>());
  };
  const size_t Passes = (num_encodings<T>() + M * P - 1) / (M * P);
  // The one-hot block is moved along K so that every K position of both
  // operands is exercised, which checks that A and B agree on the K layout.
  for (size_t KOff = 0; KOff < K; KOff += P)
    for (size_t Pass = 0; Pass < Passes; ++Pass) {
      std::vector<uint8_t> ABits(M * K, 0), BBits(K * N, 0);
      std::vector<bool> NonFinite(M, false);
      for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < P; ++j) {
          const uint8_t E = EncodingAt(i, j, Pass);
          // A: D[i][j] = A[i][j] when B is one-hot, so a non-finite A[i][k]
          // poisons row i of D.
          // B: D[j][i] = B[j][i] when A is one-hot, so a non-finite B[k][i]
          // poisons column i of D.
          if (TIsA)
            ABits[i * K + KOff + j] = E;
          else
            BBits[(KOff + j) * N + i] = E;
          NonFinite[i] = NonFinite[i] || !is_finite_encoding<T>(E);
        }
      for (size_t d = 0; d < P; ++d) {
        if (TIsA)
          BBits[(KOff + d) * N + d] = OneBits;
        else
          ABits[d * K + KOff + d] = OneBits;
      }
      std::vector<float> C(M * N, 0.f);
      std::vector<float> D;
      if constexpr (TIsA)
        D = hip_small_float_gemm<T, TOther, M, N, K, 1, layout::row_major,
                                 layout::row_major, layout::row_major>(
            ABits, BBits, C);
      else
        D = hip_small_float_gemm<TOther, T, M, N, K, 1, layout::row_major,
                                 layout::row_major, layout::row_major>(
            ABits, BBits, C);
      int Errors = 0;
      for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < P; ++j) {
          const float Got = TIsA ? D[i * N + j] : D[j * N + i];
          const uint8_t E = EncodingAt(i, j, Pass);
          bool Ok;
          if (NonFinite[i])
            Ok = std::isnan(Got);
          else
            Ok = static_cast<double>(Got) == ocp_decode<T>(E);
          if (!Ok && Errors++ < 5)
            printf("format %s as %s %zux%zux%zu: encoding 0x%02x gave %g, "
                   "expected %g (k offset %zu)\n",
                   type_name<T>(), TIsA ? "A" : "B", M, N, K, E, Got,
                   NonFinite[i] ? std::numeric_limits<double>::quiet_NaN()
                                : ocp_decode<T>(E),
                   KOff);
        }
      assert(Errors == 0 && "small-float encoding decoded incorrectly");
    }
}

// joint_matrix_fill and joint_matrix_copy on small-float multiplicands, using
// values constructed on the host.
template <typename TA, typename TB, size_t M, size_t N, size_t K>
void hip_small_float_fill_copy() {
  const TA VA = from_bits<TA>(ocp_encode_exact<TA>(-0.5));
  const TB VB = from_bits<TB>(ocp_encode_exact<TB>(2.0));
  std::vector<float> DMem(M * N, 0.f);
  {
    queue Q;
    buffer<float> BufD(DMem.data(), range{M * N});
    Q.submit([&](handler &CGH) {
       accessor AccD{BufD, CGH, write_only};
       CGH.parallel_for(nd_range<2>{{4, 16}, {4, 16}}, [=](nd_item<2> It) {
         auto SG = It.get_sub_group();
         joint_matrix<sub_group, float, use::accumulator, M, N> SubC;
         joint_matrix<sub_group, TA, use::a, M, K, layout::row_major> SubA,
             SubACopy;
         joint_matrix<sub_group, TB, use::b, K, N, layout::row_major> SubB,
             SubBCopy;
         joint_matrix_fill(SG, SubC, 1.f);
         joint_matrix_fill(SG, SubA, VA);
         joint_matrix_fill(SG, SubB, VB);
         joint_matrix_copy(SG, SubA, SubACopy);
         joint_matrix_copy(SG, SubB, SubBCopy);
         joint_matrix_mad(SG, SubC, SubACopy, SubBCopy, SubC);
         joint_matrix_store(
             SG, SubC, AccD.template get_multi_ptr<access::decorated::yes>(), N,
             layout::row_major);
       });
     }).wait();
  }
  const float Expected = 1.f - static_cast<float>(K);
  for (auto V : DMem)
    assert(V == Expected && "small-float fill/copy mismatch");
}

template <typename TA, typename TB, size_t M, size_t N, size_t K>
void hip_small_float_all_layouts() {
  hip_small_float_mfma_exact<TA, TB, M, N, K, 2, layout::row_major,
                             layout::row_major, layout::row_major>();
  hip_small_float_mfma_exact<TA, TB, M, N, K, 2, layout::row_major,
                             layout::col_major, layout::col_major>();
  hip_small_float_mfma_exact<TA, TB, M, N, K, 2, layout::col_major,
                             layout::row_major, layout::row_major>();
  hip_small_float_mfma_exact<TA, TB, M, N, K, 2, layout::col_major,
                             layout::col_major, layout::col_major>();
  hip_small_float_mfma_exact<TA, TB, M, N, K, 1, layout::row_major,
                             layout::row_major, layout::row_major>();
  hip_small_float_mfma_exact<TA, TB, M, N, K, 4, layout::row_major,
                             layout::row_major, layout::row_major>();
}

// Runs every check for an (A type, B type) pair at one MFMA shape.
template <typename TA, typename TB, size_t M, size_t N, size_t K>
void hip_small_float_mfma_suite() {
  hip_small_float_all_layouts<TA, TB, M, N, K>();
  hip_small_float_mfma_full_range<TA, TB, M, N, K, 1>();
  hip_small_float_mfma_full_range<TA, TB, M, N, K, 3>();
  hip_small_float_fill_copy<TA, TB, M, N, K>();
}
