//==---------------- dpas_common.hpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains the common utility/helper functions for the tests
// verifying DPAS functionality.

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;

constexpr dpas_argument_type s2 = dpas_argument_type::s2;
constexpr dpas_argument_type u2 = dpas_argument_type::u2;
constexpr dpas_argument_type s4 = dpas_argument_type::s4;
constexpr dpas_argument_type u4 = dpas_argument_type::u4;
constexpr dpas_argument_type s8 = dpas_argument_type::s8;
constexpr dpas_argument_type u8 = dpas_argument_type::u8;

constexpr dpas_argument_type fp16 = dpas_argument_type::fp16;
constexpr dpas_argument_type bf16 = dpas_argument_type::bf16;
constexpr dpas_argument_type tf32 = dpas_argument_type::tf32;

std::string toString(dpas_argument_type T) {
  switch (T) {
  case dpas_argument_type::s2:
    return "s2";
  case dpas_argument_type::u2:
    return "u2";
  case dpas_argument_type::s4:
    return "s4";
  case dpas_argument_type::u4:
    return "u4";
  case dpas_argument_type::s8:
    return "s8";
  case dpas_argument_type::u8:
    return "u8";
  case dpas_argument_type::fp16:
    return "fp16";
  case dpas_argument_type::bf16:
    return "bf16";
  case dpas_argument_type::tf32:
    return "tf32";
  case dpas_argument_type::s1:
  case dpas_argument_type::u1:
  case dpas_argument_type::Invalid:
    return "UNSUPPORTED";
  }
  return "UNRECOGNIZED";
}

template <dpas_argument_type T> struct DpasPrintType {
  static constexpr bool is_sint = T == dpas_argument_type::s2 ||
                                  T == dpas_argument_type::s4 ||
                                  T == dpas_argument_type::s8;
  static constexpr bool is_uint = T == dpas_argument_type::u2 ||
                                  T == dpas_argument_type::u4 ||
                                  T == dpas_argument_type::u8;
  static constexpr bool is_fp = T == dpas_argument_type::fp16 ||
                                T == dpas_argument_type::bf16 ||
                                T == dpas_argument_type::tf32;

  using type = std::conditional_t<
      is_fp, float,
      std::conditional_t<is_sint, int,
                         std::conditional_t<is_uint, unsigned int, void>>>;
};

template <int Size> struct getIntTypeWithSize {
  using type =
      std::conditional_t<Size == 4, int,
                         std::conditional_t<Size == 2, short, signed char>>;
};

template <dpas_argument_type T> struct DpasNaturalOperandType {
  static constexpr bool is_sint = T == dpas_argument_type::s2 ||
                                  T == dpas_argument_type::s4 ||
                                  T == dpas_argument_type::s8;
  static constexpr bool is_uint = T == dpas_argument_type::u2 ||
                                  T == dpas_argument_type::u4 ||
                                  T == dpas_argument_type::u8;

  static constexpr bool is_fp16 = T == dpas_argument_type::fp16;
  static constexpr bool is_bf16 = T == dpas_argument_type::bf16;
  static constexpr bool is_tf32 = T == dpas_argument_type::tf32;

  // TODO: support tf32 here.
  using type = std::conditional_t<
      is_sint, signed char,
      std::conditional_t<
          is_uint, unsigned char,
          std::conditional_t<
              is_fp16, sycl::half,
              std::conditional_t<
                  is_bf16, sycl::ext::oneapi::bfloat16,
                  std::conditional_t<
                      is_tf32, sycl::ext::intel::experimental::esimd::tfloat32,
                      void>>>>>;
};

template <dpas_argument_type T> constexpr int getBitSize() {
  switch (T) {
  case dpas_argument_type::s2:
  case dpas_argument_type::u2:
    return 2;

  case dpas_argument_type::s4:
  case dpas_argument_type::u4:
    return 4;

  case dpas_argument_type::s8:
  case dpas_argument_type::u8:
    return 8;
  case dpas_argument_type::fp16:
  case dpas_argument_type::bf16:
    return 16;

  case dpas_argument_type::tf32:
    return 32;

  case dpas_argument_type::Invalid:
  case dpas_argument_type::s1:
  case dpas_argument_type::u1:
    break;
  }
  return 0;
}

std::string toString(dpas_argument_type T1, dpas_argument_type T2) {
  return std::string("{") + toString(T1) + ", " + toString(T2) + "}";
}

template <int NumRows, int NumCols, dpas_argument_type ArgPrecision,
          typename ElemT>
void writeToHorizontallyPackedMatrix(void *VVec, int Row, int Col,
                                     ElemT Value) {
  constexpr int ElemBitSize = getBitSize<ArgPrecision>();

  ElemT *Vec = reinterpret_cast<ElemT *>(VVec);

  // 1. Find and read the target 'unsigned int' element.
  // THe unpacked matrix has dimensions: NumRows*NumCols
  constexpr int ElemsInElemT = sizeof(ElemT) * 8 / ElemBitSize;
  int UnpackedLinearIndex = Row * NumCols + Col;
  int PackedLinearIndex = UnpackedLinearIndex / ElemsInElemT;

  // 2. Update the corresponding bits of the target element.
  if constexpr (ElemBitSize == sizeof(ElemT) * 8) {
    Vec[PackedLinearIndex] = Value;
  } else {
    ElemT TargetElem = Vec[PackedLinearIndex];
    // TargetElem has 2 or more elements in it. Need to extract one.
    // TODO: for now assume that is the case only for 2 or 4-bit integers.
    assert((ElemBitSize == 2 || ElemBitSize == 4) && "Unexpected element type");

    unsigned int Offset = (UnpackedLinearIndex % ElemsInElemT) * ElemBitSize;
    unsigned int Mask = (1 << ElemBitSize) - 1;

    Value = (Value & Mask) << Offset;
    Mask = Mask << Offset;
    TargetElem = (TargetElem & ~Mask) | Value;
    Vec[PackedLinearIndex] = TargetElem;
  }
}

template <int NumRows, int NumCols, dpas_argument_type ArgPrecision,
          typename ReadT>
ReadT readFromHorizontallyPackedMatrix(void *VVec, int Row, int Col) {
  constexpr int ElemBitSize = ArgPrecision == dpas_argument_type::Invalid
                                  ? (sizeof(ReadT) * 8)
                                  : getBitSize<ArgPrecision>();
  using ElemT =
      std::conditional_t<ArgPrecision == dpas_argument_type::Invalid, ReadT,
                         typename DpasNaturalOperandType<ArgPrecision>::type>;
  ElemT *Vec = reinterpret_cast<ElemT *>(VVec);

  // 1. Find and read the target 'unsigned int' element.
  // The unpacked matrix has dimensions: NumRows*NumCols
  constexpr int ElemsInElemT = sizeof(ElemT) * 8 / ElemBitSize;
  int UnpackedLinearIndex = Row * NumCols + Col;
  int PackedLinearIndex = UnpackedLinearIndex / ElemsInElemT;
  ElemT TargetElem = Vec[PackedLinearIndex];

  // 2. Extract, add sign and return the value.
  if constexpr (ElemBitSize == sizeof(ElemT) * 8) {
    return static_cast<ReadT>(TargetElem);
  } else {
    // TargetElem has 2 or more elements in it. Need to extract one.
    // TODO: for now assume that is the case only for 2 or 4-bit integers.
    assert((ElemBitSize == 2 || ElemBitSize == 4) && "Unexpected element type");
    unsigned int Offset = (UnpackedLinearIndex % ElemsInElemT) * ElemBitSize;
    unsigned int Mask = (static_cast<uint64_t>(1) << ElemBitSize) - 1;
    ElemT Value = (TargetElem >> Offset) & Mask;
    if constexpr (std::is_signed_v<ElemT>) {
      Value <<= ((sizeof(ElemT) * 8) - ElemBitSize);
      Value >>= ((sizeof(ElemT) * 8) - ElemBitSize);
    }
    return Value;
  }
}

template <int NumRows, int NumCols, dpas_argument_type ArgPrecision,
          typename ElemT>
void writeToVerticallyPackedMatrix(void *VVec, int Row, int Col, ElemT Value) {
  int *Vec = reinterpret_cast<int *>(VVec);
  constexpr int ElemBitSize = getBitSize<ArgPrecision>();

  // 1. Find and read the target 'int' element.
  // The unpacked matrix has dimensions: NumRows*NumCols.
  constexpr int ElemsInInt = 32 / ElemBitSize;
  int PackedRow = Row / ElemsInInt;
  int PackedLinearIndex = PackedRow * NumCols + Col;
  int TargetElem = Vec[PackedLinearIndex];

  // Insert sub-element 'Value' into 32-bit int and write back to matrix.
  int ElemBitOffset = (Row % ElemsInInt) * ElemBitSize;
  int Mask = (static_cast<uint64_t>(1) << ElemBitSize) - 1;
  using IType = typename getIntTypeWithSize<sizeof(ElemT)>::type;
  int IValue = sycl::bit_cast<IType>(Value);
  IValue = (IValue & Mask) << ElemBitOffset;
  Mask = Mask << ElemBitOffset;
  TargetElem = (TargetElem & ~Mask) | IValue;
  Vec[PackedLinearIndex] = TargetElem;
}

template <int NumRows, int NumCols, dpas_argument_type ArgPrecision,
          typename ReadT>
ReadT readFromVerticallyPackedMatrix(void *VVec, int Row, int Col) {
  constexpr int ElemBitSize = getBitSize<ArgPrecision>();
  using ElemT = typename DpasNaturalOperandType<ArgPrecision>::type;
  int *Vec = reinterpret_cast<int *>(VVec);

  // 1. Find and read the target 'int' element.
  // The unpacked matrix has dimensions: NumRows*NumCols.
  constexpr int ElemsInInt = 32 / ElemBitSize;

  int PackedRow = Row / ElemsInInt;
  int TargetElem = Vec[PackedRow * NumCols + Col];

  // 2. Extract the queried sub-elem from 32-bit int, bit-cast to ReadT and
  // return.
  int ElemBitOffset = (Row % ElemsInInt) * ElemBitSize;
  unsigned int Mask = (static_cast<uint64_t>(1) << ElemBitSize) - 1;
  int Value = (TargetElem >> ElemBitOffset) & Mask;
  if constexpr (std::is_signed_v<ElemT> && std::is_integral_v<ElemT>) {
    Value <<= (32 - ElemBitSize);
    Value >>= (32 - ElemBitSize);
    return Value;
  } else {
    using IType = typename getIntTypeWithSize<sizeof(ElemT)>::type;
    IType IValue = static_cast<IType>(Value);
    return sycl::bit_cast<ElemT>(IValue);
  }
}

template <int M, int N, dpas_argument_type ArgPrecision, typename ReadT,
          bool IsHorizontalPack>
void printMatrix(void *Vec, std::string Msg) {
  std::cout << Msg << "(" << M << "x" << N
            << "), element precision = " << toString(ArgPrecision) << std::endl;
  for (int I = 0; I < M; I++) {
    for (int J = 0; J < N; J++) {

      ReadT Value;
      if constexpr (IsHorizontalPack)
        Value = readFromHorizontallyPackedMatrix<M, N, ArgPrecision, ReadT>(
            Vec, I, J);
      else
        Value = readFromVerticallyPackedMatrix<M, N, ArgPrecision, ReadT>(Vec,
                                                                          I, J);

      if constexpr (std::is_integral_v<ReadT>)
        printf("%3d", Value);
      else
        std::cout << (float)Value;
      if (J + 1 < N)
        std::cout << ",";
    }
    std::cout << std::endl;
  }
}

template <int SystolicDepth, int RepeatCount, dpas_argument_type BPrec,
          dpas_argument_type APrec, bool UseSrc0, int ExecSize,
          bool LetDeduceArgs>
bool test(queue &Q, bool Print) {
  constexpr unsigned Size = 128;
  constexpr unsigned VL = 16;

  constexpr int AElemBitSize = getBitSize<APrec>();
  constexpr int BElemBitSize = getBitSize<BPrec>();
  constexpr int OpsPerChannel =
      std::min(32 / std::max(AElemBitSize, BElemBitSize), 8);

  using BPrintT = typename DpasPrintType<BPrec>::type;
  using APrintT = typename DpasPrintType<APrec>::type;
  using ABPrintT = decltype(std::declval<APrintT>() * std::declval<BPrintT>());

  // A(_Mx_K) * B(_Kx_N) + C(_Mx_N)
  // where:
  constexpr int M = RepeatCount;
  constexpr int K = SystolicDepth * OpsPerChannel;
  constexpr int N = ExecSize; // 16 for PVC, 8 for DG2.

  std::cout << "dpas.8x" << RepeatCount << ": (ExecSize = " << ExecSize
            << "): " << toString(BPrec, APrec) << ", UseSrc0 = " << UseSrc0
            << ", LetDeduceArgs = " << LetDeduceArgs << std::endl;

  using ANaturalType = typename DpasNaturalOperandType<APrec>::type;
  using BNaturalType = typename DpasNaturalOperandType<BPrec>::type;
  using ResNaturalType = ABPrintT;
  constexpr int APackedSize = M * K * AElemBitSize / (sizeof(ANaturalType) * 8);
  constexpr int BPackedSize = K * N * BElemBitSize / (sizeof(BNaturalType) * 8);

  auto APacked = aligned_alloc_shared<ANaturalType>(128, APackedSize, Q);
  auto BPacked = aligned_alloc_shared<BNaturalType>(128, BPackedSize, Q);
  auto Res = aligned_alloc_shared<ResNaturalType>(128, M * N, Q);
  // Init APacked;
  float Value = 1.2;
  for (int II = 0; II < M; II++) {
    for (int JJ = 0; JJ < K; JJ++) {
      Value += 1.1;
      writeToHorizontallyPackedMatrix<M, K, APrec>(
          APacked, II, JJ, static_cast<ANaturalType>(Value));
    }
  }
  if (Print)
    printMatrix<M, K, APrec, APrintT, true /*horizontal-pack*/>(APacked, "A");

  // Init BPacked;
  for (int II = 0; II < K; II++) {
    for (int JJ = 0; JJ < N; JJ++) {
      int Value = (II + JJ % 4) == 0 ? 1 : (2 + II + JJ) % 3;
      writeToVerticallyPackedMatrix<K, N, BPrec>(
          BPacked, II, JJ, static_cast<BNaturalType>(Value));
      assert(Value == (int)(static_cast<BNaturalType>(Value)) && "ERROR");
    }
  }
  if (Print)
    printMatrix<K, N, BPrec, BPrintT, false /*vertical-pack*/>(BPacked, "B");

  Q.single_task([=]() SYCL_ESIMD_KERNEL {
     simd<ANaturalType, APackedSize> A(APacked, overaligned_tag<16>{});
     simd<BNaturalType, BPackedSize> B(BPacked, overaligned_tag<16>{});
     simd<ResNaturalType, M * N> C;

     if constexpr (LetDeduceArgs) {
       if constexpr (UseSrc0) {
         // Compute C = C + AxB;
         C = 1;
         C = dpas<8, RepeatCount, ResNaturalType>(C, B, A);
       } else {
         // Compute C = AxB;
         C = dpas<8, RepeatCount, ResNaturalType>(B, A);
       }

     } else {
       if constexpr (UseSrc0) {
         // Compute C = C + AxB;
         C = 1;
         C = dpas<8, RepeatCount, ResNaturalType, ResNaturalType, BNaturalType,
                  ANaturalType, BPrec, APrec>(C, B, A);
       } else {
         // Compute C = AxB;
         C = dpas<8, RepeatCount, ResNaturalType, BNaturalType, ANaturalType,
                  BPrec, APrec>(B, A);
       }
     }

     C.copy_to(Res);
   }).wait();

  if (Print)
    printMatrix<M, N, dpas_argument_type::Invalid, ABPrintT,
                true /*horizontal-pack*/>(Res, "C");

  int NErrors = 0;
  auto A = APacked;
  auto B = BPacked;
  for (int II = 0; II < M && NErrors < 10; II++) {
    for (int JJ = 0; JJ < N && NErrors < 10; JJ++) {
      ABPrintT GoldRes = 0;
      if constexpr (UseSrc0)
        GoldRes = 1;

      // Res(i,j) = C(i,j) = A(i,*)*B(*,j))
      for (int KK = 0; KK < K; KK++) {
        APrintT AVal =
            readFromHorizontallyPackedMatrix<M, K, APrec, APrintT>(A, II, KK);
        BPrintT BVal =
            readFromVerticallyPackedMatrix<K, N, BPrec, BPrintT>(B, KK, JJ);
        GoldRes += AVal * BVal;
      }
      // Res(i,j) is Res[N*i + j]
      if (Res[N * II + JJ] != GoldRes) {
        NErrors++;
        std::cerr << "Res[" << II << ", " << JJ << "] = (" << Res[M * II + JJ]
                  << ") != expected (" << GoldRes << ")" << std::endl;
      }
    } // end for JJ
  }   // end for II

  free(Res, Q);
  free(APacked, Q);
  free(BPacked, Q);
  return NErrors == 0;
}

template <int SystolicDepth, int RepeatCount, dpas_argument_type T1,
          dpas_argument_type T2, bool LetDeduceArgs = false,
          bool ExecSize16Only = false>
bool tests(queue Q, bool Print) {
  bool Passed = true;
  constexpr bool UseSrc0 = true;
  auto Dev = Q.get_device();

  // Detect the execution size.
  int ExecSize;
  try {
    ExecSize = Dev.get_info<ext::intel::info::device::gpu_eu_simd_width>();
  } catch (sycl::exception e) {
    std::cerr << "Could not determine ExecSize, FAIL" << std::endl;
    return false;
  }
  assert(((ExecSize == 8 || ExecSize == 16)) &&
         "Execution size must be 8 or 16");

  if (ExecSize == 16) {
    Passed &=
        test<SystolicDepth, RepeatCount, T1, T2, UseSrc0, 16, LetDeduceArgs>(
            Q, Print);
    Passed &=
        test<SystolicDepth, RepeatCount, T1, T2, !UseSrc0, 16, LetDeduceArgs>(
            Q, Print);
  }
  if (ExecSize == 8) {
    if constexpr (!ExecSize16Only) {
      Passed &=
          test<SystolicDepth, RepeatCount, T1, T2, UseSrc0, 8, LetDeduceArgs>(
              Q, Print);
      Passed &=
          test<SystolicDepth, RepeatCount, T1, T2, !UseSrc0, 8, LetDeduceArgs>(
              Q, Print);
    }
  }

  return Passed;
}
