//===--- OffloadArch.h - Definition of offloading architectures --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OFFLOADARCH_H
#define LLVM_CLANG_BASIC_OFFLOADARCH_H

namespace llvm {
class StringRef;
template <typename T, typename R> class StringSwitch;
} // namespace llvm

namespace clang {

enum class OffloadArch {
  UNUSED,
  UNKNOWN,
  // TODO: Deprecate and remove GPU architectures older than sm_52.
  SM_20,
  SM_21,
  SM_30,
  // This has a name conflict with sys/mac.h on AIX, rename it as a workaround.
  SM_32_,
  SM_35,
  SM_37,
  SM_50,
  SM_52,
  SM_53,
  SM_60,
  SM_61,
  SM_62,
  SM_70,
  SM_72,
  SM_75,
  SM_80,
  SM_86,
  SM_87,
  SM_89,
  SM_90,
  SM_90a,
  SM_100,
  SM_100a,
  SM_101,
  SM_101a,
  SM_103,
  SM_103a,
  SM_120,
  SM_120a,
  SM_121,
  SM_121a,
  GFX600,
  GFX601,
  GFX602,
  GFX700,
  GFX701,
  GFX702,
  GFX703,
  GFX704,
  GFX705,
  GFX801,
  GFX802,
  GFX803,
  GFX805,
  GFX810,
  GFX9_GENERIC,
  GFX900,
  GFX902,
  GFX904,
  GFX906,
  GFX908,
  GFX909,
  GFX90a,
  GFX90c,
  GFX9_4_GENERIC,
  GFX942,
  GFX950,
  GFX10_1_GENERIC,
  GFX1010,
  GFX1011,
  GFX1012,
  GFX1013,
  GFX10_3_GENERIC,
  GFX1030,
  GFX1031,
  GFX1032,
  GFX1033,
  GFX1034,
  GFX1035,
  GFX1036,
  GFX11_GENERIC,
  GFX1100,
  GFX1101,
  GFX1102,
  GFX1103,
  GFX1150,
  GFX1151,
  GFX1152,
  GFX1153,
  GFX12_GENERIC,
  GFX1200,
  GFX1201,
  GFX1250,
  GFX1251,
  AMDGCNSPIRV,
  Generic, // A processor model named 'generic' if the target backend defines a
           // public one.
  // Intel CPUs
  SKYLAKEAVX512,
  COREAVX2,
  COREI7AVX,
  COREI7,
  WESTMERE,
  SANDYBRIDGE,
  IVYBRIDGE,
  BROADWELL,
  COFFEELAKE,
  ALDERLAKE,
  SKYLAKE,
  SKX,
  CASCADELAKE,
  ICELAKECLIENT,
  ICELAKESERVER,
  SAPPHIRERAPIDS,
  GRANITERAPIDS,
  // Intel GPUs
  BDW,
  SKL,
  KBL,
  CFL,
  APL,
  BXT,
  GLK,
  WHL,
  AML,
  CML,
  ICLLP,
  ICL,
  EHL,
  JSL,
  TGLLP,
  TGL,
  RKL,
  ADL_S,
  RPL_S,
  ADL_P,
  ADL_N,
  DG1,
  ACM_G10,
  DG2_G10,
  ACM_G11,
  DG2_G11,
  ACM_G12,
  DG2_G12,
  PVC,
  PVC_VG,
  MTL_U,
  MTL_S,
  ARL_U,
  ARL_S,
  MTL_H,
  ARL_H,
  BMG_G21,
  LNL_M,
  LAST,

  CudaDefault = OffloadArch::SM_52,
  HIPDefault = OffloadArch::GFX906,
};

static inline bool IsNVIDIAOffloadArch(OffloadArch A) {
  return A >= OffloadArch::SM_20 && A < OffloadArch::GFX600;
}

static inline bool IsAMDOffloadArch(OffloadArch A) {
  // Generic processor model is for testing only.
  return A >= OffloadArch::GFX600 && A < OffloadArch::Generic;
}

static inline bool IsIntelCPUOffloadArch(OffloadArch Arch) {
  return Arch >= OffloadArch::SKYLAKEAVX512 &&
         Arch <= OffloadArch::GRANITERAPIDS;
}

static inline bool IsIntelGPUOffloadArch(OffloadArch Arch) {
  return Arch >= OffloadArch::BDW && Arch < OffloadArch::LAST;
}

static inline bool IsIntelOffloadArch(OffloadArch Arch) {
  return IsIntelCPUOffloadArch(Arch) || IsIntelGPUOffloadArch(Arch);
}

// Check if the given Arch value is a Generic AMD GPU.
// Currently GFX*_GENERIC AMD GPUs do not support SYCL offloading.
// This list is used to filter out GFX*_GENERIC AMD GPUs in
// `IsSYCLSupportedAMDGPUArch`.
static inline bool IsAMDGenericGPUArch(OffloadArch Arch) {
  return Arch == OffloadArch::GFX9_GENERIC ||
         Arch == OffloadArch::GFX10_1_GENERIC ||
         Arch == OffloadArch::GFX10_3_GENERIC ||
         Arch == OffloadArch::GFX11_GENERIC ||
         Arch == OffloadArch::GFX12_GENERIC;
}

// Check if the given Arch value is a valid SYCL supported AMD GPU.
static inline bool IsSYCLSupportedAMDGPUArch(OffloadArch Arch) {
  return Arch >= OffloadArch::GFX700 && Arch < OffloadArch::AMDGCNSPIRV &&
         !IsAMDGenericGPUArch(Arch);
}

// Check if the given Arch value is a valid SYCL supported NVidia GPU.
static inline bool IsSYCLSupportedNVidiaGPUArch(OffloadArch Arch) {
  return Arch >= OffloadArch::SM_50 && Arch <= OffloadArch::SM_90a;
}

const char *OffloadArchToString(OffloadArch A);
const char *OffloadArchToVirtualArchString(OffloadArch A);

// Convert a string to an OffloadArch enum value. Returns
// OffloadArch::UNKNOWN if the string is not recognized.
OffloadArch StringToOffloadArch(llvm::StringRef S);

} // namespace clang

#endif // LLVM_CLANG_BASIC_OFFLOADARCH_H
