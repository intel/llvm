//===---joint_matrix_hip_rocwmma.hpp - DPC++ joint_matrix------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// joint_matrix tests for AMD GPUs modeled on the rocWMMA test suite
// (https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma/test):
//
//  - hip_rocwmma_gemm: the rocWMMA GEMM tests. D = alpha * A * B + beta * C on
//    rocWMMA's validation problem sizes, with several sub-groups per
//    work-group (rocWMMA thread blocks), several output tiles per sub-group
//    (rocWMMA blocks per wave), all A/B layouts, and optionally staging the
//    multiplicands and the result through local memory (rocWMMA's LDS
//    GEMMs).
//  - hip_rocwmma_contamination: the rocWMMA load/store contamination tests.
//    A, B and C are loaded at an offset inside padded buffers whose padding
//    holds poison values, and D is stored at an offset inside a padded buffer
//    whose padding must be left untouched.
//
// Multiplicand values are small multiples of 1/2 (integers for int8) that are
// exact in every multiplicand type, so the references are exact.

#include "joint_matrix_hip_small_float.hpp"

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/group_barrier.hpp>

#include <algorithm>
#include <memory>
#include <tuple>

using sycl::ext::oneapi::bfloat16;

template <typename T>
constexpr bool is_rocwmma_small_float_v =
    std::is_same_v<T, fp8_e4m3> || std::is_same_v<T, fp8_e5m2> ||
    std::is_same_v<T, fp4_e2m1>;

template <typename T> T rocwmma_to_storage(double V) {
  if constexpr (is_rocwmma_small_float_v<T>)
    return from_bits<T>(ocp_encode_exact<T>(V));
  else
    return static_cast<T>(static_cast<float>(V));
}

template <typename T> double rocwmma_pick(std::mt19937 &Gen) {
  // Integers in [-2, 2] for int8, multiples of 1/2 in [-2, 2] otherwise.
  if constexpr (std::is_integral_v<T>)
    return std::uniform_int_distribution<int>(-2, 2)(Gen);
  else
    return std::uniform_int_distribution<int>(-4, 4)(Gen) * 0.5;
}

// Offset of logical element (r, c) of a matrix stored with layout L and
// leading dimension Ld.
inline size_t rocwmma_offset(size_t R, size_t C, layout L, size_t Ld) {
  return L == layout::row_major ? R * Ld + C : C * Ld + R;
}

// The FP8/FP4 MFMAs accumulate with slightly less precision than IEEE fp32,
// so results are compared against a bound relative to the sum of magnitudes.
template <typename TA, typename TB>
bool rocwmma_close(double Got, double Ref, double Mag) {
  if constexpr (is_rocwmma_small_float_v<TA> || is_rocwmma_small_float_v<TB>)
    return std::fabs(Got - Ref) <= 0x1p-11 * Mag;
  else
    return Got == Ref;
}

template <typename T> const char *rocwmma_type_name() {
  if constexpr (is_rocwmma_small_float_v<T>)
    return type_name<T>();
  else if constexpr (std::is_same_v<T, int8_t>)
    return "int8";
  else if constexpr (std::is_same_v<T, half>)
    return "half";
  else if constexpr (std::is_same_v<T, bfloat16>)
    return "bfloat16";
  else if constexpr (std::is_same_v<T, float>)
    return "float";
  else
    return "double";
}

inline const char *rocwmma_layout_name(layout L) {
  return L == layout::row_major ? "T" : "N";
}

// Logical row-major problem data and the reference result.
template <typename TA, typename TB, typename TC> struct rocwmma_problem {
  size_t PM, PN, PK;
  TC Alpha, Beta;
  std::vector<double> A, B, C, Ref, Mag;
  std::vector<TA> AS;
  std::vector<TB> BS;

  rocwmma_problem(size_t PM, size_t PN, size_t PK, TC Alpha, TC Beta,
                  unsigned Seed)
      : PM(PM), PN(PN), PK(PK), Alpha(Alpha), Beta(Beta), A(PM * PK),
        B(PK * PN), C(PM * PN), Ref(PM * PN, 0.0), Mag(PM * PN, 0.0),
        AS(PM * PK), BS(PK * PN) {
    std::mt19937 Gen(Seed);
    for (auto &V : A)
      V = rocwmma_pick<TA>(Gen);
    for (auto &V : B)
      V = rocwmma_pick<TB>(Gen);
    for (auto &V : C)
      V = std::uniform_int_distribution<int>(-8, 8)(Gen);
    // Every value is a multiple of 1/2 in [-2, 2], so convert each one once.
    std::vector<TA> AVals(9);
    std::vector<TB> BVals(9);
    for (int i = 0; i < 9; ++i) {
      if (!std::is_integral_v<TA> || i % 2 == 0)
        AVals[i] = rocwmma_to_storage<TA>((i - 4) * 0.5);
      if (!std::is_integral_v<TB> || i % 2 == 0)
        BVals[i] = rocwmma_to_storage<TB>((i - 4) * 0.5);
    }
    for (size_t i = 0; i < A.size(); ++i)
      AS[i] = AVals[static_cast<int>(A[i] * 2) + 4];
    for (size_t i = 0; i < B.size(); ++i)
      BS[i] = BVals[static_cast<int>(B[i] * 2) + 4];
    for (size_t m = 0; m < PM; ++m)
      for (size_t k = 0; k < PK; ++k) {
        const double AV = A[m * PK + k];
        for (size_t n = 0; n < PN; ++n) {
          Ref[m * PN + n] += AV * B[k * PN + n];
          Mag[m * PN + n] += std::fabs(AV * B[k * PN + n]);
        }
      }
    for (size_t i = 0; i < PM * PN; ++i) {
      Mag[i] = std::fabs(static_cast<double>(Alpha)) * Mag[i] +
               std::fabs(static_cast<double>(Beta) * C[i]);
      Ref[i] = static_cast<double>(Alpha) * Ref[i] +
               static_cast<double>(Beta) * C[i];
    }
  }
};

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          size_t BlocksM, size_t BlocksN, layout AL, layout BL, layout CL,
          bool UseLocal>
class rocwmma_gemm_kernel;

// Runs one GEMM with WavesM x WavesN sub-groups per work-group, each one
// computing BlocksM x BlocksN tiles of M x N. Returns false if the problem
// size is not a multiple of the work-group tile.
template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          size_t BlocksM, size_t BlocksN, layout AL, layout BL, layout CL,
          bool UseLocal>
bool hip_rocwmma_gemm_run(queue &Q, const rocwmma_problem<TA, TB, TC> &P,
                          size_t WavesM, size_t WavesN) {
  constexpr size_t Wave = 64;
  const size_t MacroM = WavesM * BlocksM * M, MacroN = WavesN * BlocksN * N;
  const size_t PM = P.PM, PN = P.PN, PK = P.PK;
  if (PM % MacroM || PN % MacroN || PK % K)
    return false;

  const size_t LdA = AL == layout::row_major ? PK : PM;
  const size_t LdB = BL == layout::row_major ? PN : PK;
  const size_t LdC = CL == layout::row_major ? PN : PM;
  std::vector<TA> AMem(PM * PK);
  std::vector<TB> BMem(PK * PN);
  std::vector<TC> CMem(PM * PN), DMem(PM * PN, TC(0));
  for (size_t m = 0; m < PM; ++m)
    for (size_t k = 0; k < PK; ++k)
      AMem[rocwmma_offset(m, k, AL, LdA)] = P.AS[m * PK + k];
  for (size_t k = 0; k < PK; ++k)
    for (size_t n = 0; n < PN; ++n)
      BMem[rocwmma_offset(k, n, BL, LdB)] = P.BS[k * PN + n];
  for (size_t m = 0; m < PM; ++m)
    for (size_t n = 0; n < PN; ++n)
      CMem[rocwmma_offset(m, n, CL, LdC)] = static_cast<TC>(P.C[m * PN + n]);

  {
    buffer<TA> BufA(AMem.data(), range{AMem.size()});
    buffer<TB> BufB(BMem.data(), range{BMem.size()});
    buffer<TC> BufC(CMem.data(), range{CMem.size()});
    buffer<TC> BufD(DMem.data(), range{DMem.size()});
    Q.submit([&](handler &CGH) {
       accessor AccA{BufA, CGH, read_only};
       accessor AccB{BufB, CGH, read_only};
       accessor AccC{BufC, CGH, read_only};
       accessor AccD{BufD, CGH, write_only};
       // Local tiles of the work-group: A is MacroM x K, B is K x MacroN and D
       // is MacroM x MacroN, each one stored with the layout of its global
       // matrix.
       const size_t LocASize = UseLocal ? MacroM * K : 1;
       const size_t LocBSize = UseLocal ? K * MacroN : 1;
       const size_t LocDSize = UseLocal ? MacroM * MacroN : 1;
       local_accessor<TA, 1> LocA{range{LocASize}, CGH};
       local_accessor<TB, 1> LocB{range{LocBSize}, CGH};
       local_accessor<TC, 1> LocD{range{LocDSize}, CGH};
       const TC Alpha = P.Alpha, Beta = P.Beta;
       const range<2> Local{WavesM, WavesN * Wave};
       const range<2> Global{PM / MacroM * WavesM, PN / MacroN * WavesN * Wave};
       CGH.parallel_for<rocwmma_gemm_kernel<TA, TB, TC, M, N, K, BlocksM,
                                            BlocksN, AL, BL, CL, UseLocal>>(
           nd_range<2>{Global, Local}, [=](nd_item<2> It) {
             auto SG = It.get_sub_group();
             const size_t WM = It.get_local_id(0);
             const size_t WN = It.get_local_id(1) / Wave;
             const size_t TileM0 = It.get_group(0) * MacroM;
             const size_t TileN0 = It.get_group(1) * MacroN;
             const size_t WaveM0 = WM * BlocksM * M, WaveN0 = WN * BlocksN * N;
             const size_t LId = It.get_local_linear_id();
             const size_t WGSize = It.get_local_range().size();
             const size_t LdLA = AL == layout::row_major ? K : MacroM;
             const size_t LdLB = BL == layout::row_major ? MacroN : K;
             const size_t LdLD = CL == layout::row_major ? MacroN : MacroM;
             auto PA = AccA.template get_multi_ptr<access::decorated::yes>();
             auto PB = AccB.template get_multi_ptr<access::decorated::yes>();
             auto PC = AccC.template get_multi_ptr<access::decorated::yes>();
             auto PD = AccD.template get_multi_ptr<access::decorated::yes>();

             joint_matrix<sub_group, TC, use::accumulator, M, N> Acc[BlocksM]
                                                                    [BlocksN];
             for (size_t bm = 0; bm < BlocksM; ++bm)
               for (size_t bn = 0; bn < BlocksN; ++bn)
                 joint_matrix_fill(SG, Acc[bm][bn], TC(0));

             for (size_t k0 = 0; k0 < PK; k0 += K) {
               joint_matrix<sub_group, TA, use::a, M, K, AL> SubA[BlocksM];
               joint_matrix<sub_group, TB, use::b, K, N, BL> SubB[BlocksN];
               if constexpr (UseLocal) {
                 for (size_t E = LId; E < MacroM * K; E += WGSize) {
                   const size_t R = E / K, C = E % K;
                   LocA[rocwmma_offset(R, C, AL, LdLA)] =
                       AccA[rocwmma_offset(TileM0 + R, k0 + C, AL, LdA)];
                 }
                 for (size_t E = LId; E < K * MacroN; E += WGSize) {
                   const size_t R = E / MacroN, C = E % MacroN;
                   LocB[rocwmma_offset(R, C, BL, LdLB)] =
                       AccB[rocwmma_offset(k0 + R, TileN0 + C, BL, LdB)];
                 }
                 group_barrier(It.get_group());
                 auto LA =
                     LocA.template get_multi_ptr<access::decorated::yes>();
                 auto LB =
                     LocB.template get_multi_ptr<access::decorated::yes>();
                 for (size_t bm = 0; bm < BlocksM; ++bm)
                   joint_matrix_load(
                       SG, SubA[bm],
                       LA + rocwmma_offset(WaveM0 + bm * M, 0, AL, LdLA), LdLA);
                 for (size_t bn = 0; bn < BlocksN; ++bn)
                   joint_matrix_load(
                       SG, SubB[bn],
                       LB + rocwmma_offset(0, WaveN0 + bn * N, BL, LdLB), LdLB);
               } else {
                 for (size_t bm = 0; bm < BlocksM; ++bm)
                   joint_matrix_load(
                       SG, SubA[bm],
                       PA + rocwmma_offset(TileM0 + WaveM0 + bm * M, k0, AL,
                                           LdA),
                       LdA);
                 for (size_t bn = 0; bn < BlocksN; ++bn)
                   joint_matrix_load(
                       SG, SubB[bn],
                       PB + rocwmma_offset(k0, TileN0 + WaveN0 + bn * N, BL,
                                           LdB),
                       LdB);
               }
               for (size_t bm = 0; bm < BlocksM; ++bm)
                 for (size_t bn = 0; bn < BlocksN; ++bn)
                   joint_matrix_mad(SG, Acc[bm][bn], SubA[bm], SubB[bn],
                                    Acc[bm][bn]);
               if constexpr (UseLocal)
                 group_barrier(It.get_group());
             }

             auto LD = LocD.template get_multi_ptr<access::decorated::yes>();
             for (size_t bm = 0; bm < BlocksM; ++bm)
               for (size_t bn = 0; bn < BlocksN; ++bn) {
                 const size_t R = TileM0 + WaveM0 + bm * M;
                 const size_t C = TileN0 + WaveN0 + bn * N;
                 joint_matrix<sub_group, TC, use::accumulator, M, N> SubC;
                 joint_matrix_load(SG, SubC, PC + rocwmma_offset(R, C, CL, LdC),
                                   LdC, CL);
                 joint_matrix_apply(
                     SG, Acc[bm][bn], SubC,
                     [=](TC &D, const TC &CV) { D = Alpha * D + Beta * CV; });
                 if constexpr (UseLocal)
                   joint_matrix_store(SG, Acc[bm][bn],
                                      LD + rocwmma_offset(WaveM0 + bm * M,
                                                          WaveN0 + bn * N, CL,
                                                          LdLD),
                                      LdLD, CL);
                 else
                   joint_matrix_store(SG, Acc[bm][bn],
                                      PD + rocwmma_offset(R, C, CL, LdC), LdC,
                                      CL);
               }
             if constexpr (UseLocal) {
               group_barrier(It.get_group());
               for (size_t E = LId; E < MacroM * MacroN; E += WGSize) {
                 const size_t R = E / MacroN, C = E % MacroN;
                 AccD[rocwmma_offset(TileM0 + R, TileN0 + C, CL, LdC)] =
                     LocD[rocwmma_offset(R, C, CL, LdLD)];
               }
             }
           });
     }).wait();
  }

  int Errors = 0;
  for (size_t m = 0; m < PM; ++m)
    for (size_t n = 0; n < PN; ++n) {
      const double Got =
          static_cast<double>(DMem[rocwmma_offset(m, n, CL, LdC)]);
      if (!rocwmma_close<TA, TB>(Got, P.Ref[m * PN + n], P.Mag[m * PN + n]) &&
          Errors++ < 5)
        printf("gemm %s x %s %zux%zux%zu %s%s%s blocks %zux%zu waves %zux%zu%s"
               " problem %zux%zux%zu: D[%zu][%zu] = %.9g, expected %.9g\n",
               rocwmma_type_name<TA>(), rocwmma_type_name<TB>(), M, N, K,
               rocwmma_layout_name(AL), rocwmma_layout_name(BL),
               rocwmma_layout_name(CL), BlocksM, BlocksN, WavesM, WavesN,
               UseLocal ? " local" : "", PM, PN, PK, m, n, Got,
               P.Ref[m * PN + n]);
    }
  assert(Errors == 0 && "rocWMMA-style GEMM mismatch");
  return true;
}

// rocWMMA validation problem sizes (M, N, K), each one paired with the
// rocWMMA thread blocks it runs with, given as sub-groups along M x sub-groups
// along N. All thread blocks run on the first problem size, while the other
// sizes use one thread block each to keep the run time reasonable.
struct rocwmma_config {
  size_t PM, PN, PK;
  std::vector<std::pair<size_t, size_t>> Waves;
};

inline const std::vector<rocwmma_config> &rocwmma_configs() {
  static const std::vector<rocwmma_config> Configs = {
      {64, 64, 1024, {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}}},
      {32, 64, 1024, {{1, 2}}},
      {64, 32, 1024, {{2, 1}}},
      {256, 256, 1024, {{2, 2}}},
      {2048, 64, 1024, {{4, 1}}},
      {64, 2048, 1024, {{1, 4}}},
      {512, 512, 512, {{2, 2}}}};
  return Configs;
}

// The problems only depend on the types, so they are shared by all shapes.
template <typename TA, typename TB, typename TC>
const rocwmma_problem<TA, TB, TC> &rocwmma_get_problem(size_t Index, TC Alpha,
                                                       TC Beta) {
  static std::vector<std::unique_ptr<rocwmma_problem<TA, TB, TC>>> Problems(
      rocwmma_configs().size());
  auto &P = Problems[Index];
  if (!P || P->Alpha != Alpha || P->Beta != Beta) {
    const auto &C = rocwmma_configs()[Index];
    P = std::make_unique<rocwmma_problem<TA, TB, TC>>(
        C.PM, C.PN, C.PK, Alpha, Beta, static_cast<unsigned>(Index + 1));
  }
  return *P;
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          layout AL, layout BL>
void hip_rocwmma_gemm_layout(
    queue &Q, const rocwmma_problem<TA, TB, TC> &P,
    const std::vector<std::pair<size_t, size_t>> &Waves, size_t &Runs) {
  for (auto [WavesM, WavesN] : Waves) {
    // C and D are col_major as in rocWMMA, plus row_major for one blocking.
    Runs +=
        hip_rocwmma_gemm_run<TA, TB, TC, M, N, K, 1, 1, AL, BL,
                             layout::col_major, false>(Q, P, WavesM, WavesN);
    Runs +=
        hip_rocwmma_gemm_run<TA, TB, TC, M, N, K, 2, 2, AL, BL,
                             layout::col_major, false>(Q, P, WavesM, WavesN);
    Runs +=
        hip_rocwmma_gemm_run<TA, TB, TC, M, N, K, 1, 2, AL, BL,
                             layout::row_major, false>(Q, P, WavesM, WavesN);
    Runs += hip_rocwmma_gemm_run<TA, TB, TC, M, N, K, 1, 1, AL, BL,
                                 layout::col_major, true>(Q, P, WavesM, WavesN);
  }
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K>
void hip_rocwmma_gemm(TC Alpha = TC(2), TC Beta = TC(-1)) {
  queue Q;
  size_t Runs = 0;
  for (size_t i = 0; i < rocwmma_configs().size(); ++i) {
    const auto &P = rocwmma_get_problem<TA, TB, TC>(i, Alpha, Beta);
    const auto &W = rocwmma_configs()[i].Waves;
    hip_rocwmma_gemm_layout<TA, TB, TC, M, N, K, layout::col_major,
                            layout::col_major>(Q, P, W, Runs);
    hip_rocwmma_gemm_layout<TA, TB, TC, M, N, K, layout::col_major,
                            layout::row_major>(Q, P, W, Runs);
    hip_rocwmma_gemm_layout<TA, TB, TC, M, N, K, layout::row_major,
                            layout::col_major>(Q, P, W, Runs);
    hip_rocwmma_gemm_layout<TA, TB, TC, M, N, K, layout::row_major,
                            layout::row_major>(Q, P, W, Runs);
  }
  assert(Runs > 0);
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          layout AL, layout BL, layout CL, layout DL>
class rocwmma_contamination_kernel;

// rocWMMA's contamination padding: 4 rows and 8 columns on each side.
constexpr size_t RocwmmaPadR = 4, RocwmmaPadC = 8;

// A padded R x C matrix with layout L whose logical element (r, c) lives at
// (r + PadR, c + PadC) of a (R + 2 PadR) x (C + 2 PadC) matrix.
struct rocwmma_padded {
  size_t R, C;
  layout L;
  size_t ld() const {
    return L == layout::row_major ? C + 2 * RocwmmaPadC : R + 2 * RocwmmaPadR;
  }
  size_t size() const { return (R + 2 * RocwmmaPadR) * (C + 2 * RocwmmaPadC); }
  size_t origin() const {
    return rocwmma_offset(RocwmmaPadR, RocwmmaPadC, L, ld());
  }
  size_t at(size_t r, size_t c) const {
    return origin() + rocwmma_offset(r, c, L, ld());
  }
};

// Poison values for the padding of the multiplicands: NaN for the floating
// point formats that have it, and values outside the data range otherwise.
template <typename T> T rocwmma_poison() {
  if constexpr (std::is_same_v<T, fp8_e4m3> || std::is_same_v<T, fp8_e5m2>)
    return from_bits<T>(0x7F);
  else if constexpr (std::is_same_v<T, fp4_e2m1>)
    return from_bits<T>(0x7); // 6.0, the largest FP4 value.
  else if constexpr (std::is_integral_v<T>)
    return T(100);
  else
    return static_cast<T>(std::numeric_limits<float>::quiet_NaN());
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          layout AL, layout BL, layout CL, layout DL>
void hip_rocwmma_contamination_run(queue &Q) {
  // B has no zero so that any element read from the padding of A changes D.
  std::mt19937 Gen(M + N + K);
  std::vector<double> A(M * K), B(K * N), C(M * N), Ref(M * N), Mag(M * N);
  for (auto &V : A)
    V = rocwmma_pick<TA>(Gen);
  for (auto &V : B)
    do
      V = rocwmma_pick<TB>(Gen);
    while (V == 0.0);
  for (auto &V : C)
    V = std::uniform_int_distribution<int>(-8, 8)(Gen);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      Ref[m * N + n] = C[m * N + n];
      Mag[m * N + n] = std::fabs(C[m * N + n]);
      for (size_t k = 0; k < K; ++k) {
        Ref[m * N + n] += A[m * K + k] * B[k * N + n];
        Mag[m * N + n] += std::fabs(A[m * K + k] * B[k * N + n]);
      }
    }

  const rocwmma_padded PA{M, K, AL}, PB{K, N, BL}, PC{M, N, CL}, PD{M, N, DL};
  const TC CPoison =
      std::is_integral_v<TC>
          ? TC(1 << 20)
          : static_cast<TC>(std::numeric_limits<float>::quiet_NaN());
  const TC Canary = TC(-77);
  std::vector<TA> AMem(PA.size(), rocwmma_poison<TA>());
  std::vector<TB> BMem(PB.size(), rocwmma_poison<TB>());
  std::vector<TC> CMem(PC.size(), CPoison), DMem(PD.size(), Canary);
  for (size_t m = 0; m < M; ++m)
    for (size_t k = 0; k < K; ++k)
      AMem[PA.at(m, k)] = rocwmma_to_storage<TA>(A[m * K + k]);
  for (size_t k = 0; k < K; ++k)
    for (size_t n = 0; n < N; ++n)
      BMem[PB.at(k, n)] = rocwmma_to_storage<TB>(B[k * N + n]);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n)
      CMem[PC.at(m, n)] = static_cast<TC>(C[m * N + n]);

  {
    buffer<TA> BufA(AMem.data(), range{AMem.size()});
    buffer<TB> BufB(BMem.data(), range{BMem.size()});
    buffer<TC> BufC(CMem.data(), range{CMem.size()});
    buffer<TC> BufD(DMem.data(), range{DMem.size()});
    Q.submit([&](handler &CGH) {
       accessor AccA{BufA, CGH, read_only};
       accessor AccB{BufB, CGH, read_only};
       accessor AccC{BufC, CGH, read_only};
       accessor AccD{BufD, CGH, read_write};
       CGH.parallel_for<
           rocwmma_contamination_kernel<TA, TB, TC, M, N, K, AL, BL, CL, DL>>(
           nd_range<2>{{1, 64}, {1, 64}}, [=](nd_item<2> It) {
             auto SG = It.get_sub_group();
             joint_matrix<sub_group, TA, use::a, M, K, AL> SubA;
             joint_matrix<sub_group, TB, use::b, K, N, BL> SubB;
             joint_matrix<sub_group, TC, use::accumulator, M, N> SubC;
             joint_matrix_load(
                 SG, SubA,
                 AccA.template get_multi_ptr<access::decorated::yes>() +
                     PA.origin(),
                 PA.ld());
             joint_matrix_load(
                 SG, SubB,
                 AccB.template get_multi_ptr<access::decorated::yes>() +
                     PB.origin(),
                 PB.ld());
             joint_matrix_load(
                 SG, SubC,
                 AccC.template get_multi_ptr<access::decorated::yes>() +
                     PC.origin(),
                 PC.ld(), CL);
             joint_matrix_mad(SG, SubC, SubA, SubB, SubC);
             joint_matrix_store(
                 SG, SubC,
                 AccD.template get_multi_ptr<access::decorated::yes>() +
                     PD.origin(),
                 PD.ld(), DL);
           });
     }).wait();
  }

  int Errors = 0;
  std::vector<bool> InTile(PD.size(), false);
  for (size_t m = 0; m < M; ++m)
    for (size_t n = 0; n < N; ++n) {
      InTile[PD.at(m, n)] = true;
      const double Got = static_cast<double>(DMem[PD.at(m, n)]);
      if (!rocwmma_close<TA, TB>(Got, Ref[m * N + n], Mag[m * N + n]) &&
          Errors++ < 5)
        printf("contamination %s x %s %zux%zux%zu %s%s%s%s: D[%zu][%zu] = "
               "%.9g, expected %.9g\n",
               rocwmma_type_name<TA>(), rocwmma_type_name<TB>(), M, N, K,
               rocwmma_layout_name(AL), rocwmma_layout_name(BL),
               rocwmma_layout_name(CL), rocwmma_layout_name(DL), m, n, Got,
               Ref[m * N + n]);
    }
  for (size_t i = 0; i < PD.size(); ++i)
    if (!InTile[i] && DMem[i] != Canary && Errors++ < 5)
      printf("contamination %s x %s %zux%zux%zu %s%s%s%s: store wrote padding "
             "element %zu\n",
             rocwmma_type_name<TA>(), rocwmma_type_name<TB>(), M, N, K,
             rocwmma_layout_name(AL), rocwmma_layout_name(BL),
             rocwmma_layout_name(CL), rocwmma_layout_name(DL), i);
  assert(Errors == 0 && "rocWMMA-style contamination test failed");
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          layout AL, layout BL>
void hip_rocwmma_contamination_ab(queue &Q) {
  hip_rocwmma_contamination_run<TA, TB, TC, M, N, K, AL, BL, layout::row_major,
                                layout::row_major>(Q);
  hip_rocwmma_contamination_run<TA, TB, TC, M, N, K, AL, BL, layout::row_major,
                                layout::col_major>(Q);
  hip_rocwmma_contamination_run<TA, TB, TC, M, N, K, AL, BL, layout::col_major,
                                layout::row_major>(Q);
  hip_rocwmma_contamination_run<TA, TB, TC, M, N, K, AL, BL, layout::col_major,
                                layout::col_major>(Q);
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K>
void hip_rocwmma_contamination() {
  queue Q;
  hip_rocwmma_contamination_ab<TA, TB, TC, M, N, K, layout::row_major,
                               layout::row_major>(Q);
  hip_rocwmma_contamination_ab<TA, TB, TC, M, N, K, layout::row_major,
                               layout::col_major>(Q);
  hip_rocwmma_contamination_ab<TA, TB, TC, M, N, K, layout::col_major,
                               layout::row_major>(Q);
  hip_rocwmma_contamination_ab<TA, TB, TC, M, N, K, layout::col_major,
                               layout::col_major>(Q);
}
