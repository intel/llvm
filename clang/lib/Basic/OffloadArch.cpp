//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/OffloadArch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {

OffloadArch OffloadArch::CudaDefault() {
  return getNVPTX(llvm::NVPTX::parseArch("sm_75"));
}

OffloadArch OffloadArch::HIPDefault() {
  return getAMDGPU(llvm::AMDGPU::parseArchAMDGCN("gfx906"));
}

namespace {
struct IntelArchNameMap {
  OffloadArch::TargetArch V;
  OffloadArch::IntelArch Arch;
  const char *Name;
};
} // namespace

#define INTEL_CPU(id, name)                                                    \
  {OffloadArch::TargetArch::IntelCPU, OffloadArch::IntelArch::id, name}
#define INTEL_GPU(id, name)                                                    \
  {OffloadArch::TargetArch::IntelGPU, OffloadArch::IntelArch::id, name}
static const IntelArchNameMap IntelArchNames[] = {
    // clang-format off
    INTEL_CPU(SKYLAKEAVX512, "skylake-avx512"),
    INTEL_CPU(COREAVX2, "core-avx2"),
    INTEL_CPU(COREI7AVX, "corei7-avx"),
    INTEL_CPU(COREI7, "corei7"),
    INTEL_CPU(WESTMERE, "westmere"),
    INTEL_CPU(SANDYBRIDGE, "sandybridge"),
    INTEL_CPU(IVYBRIDGE, "ivybridge"),
    INTEL_CPU(BROADWELL, "broadwell"),
    INTEL_CPU(COFFEELAKE, "coffeelake"),
    INTEL_CPU(ALDERLAKE, "alderlake"),
    INTEL_CPU(SKYLAKE, "skylake"),
    INTEL_CPU(SKX, "skx"),
    INTEL_CPU(CASCADELAKE, "cascadelake"),
    INTEL_CPU(ICELAKECLIENT, "icelake-client"),
    INTEL_CPU(ICELAKESERVER, "icelake-server"),
    INTEL_CPU(SAPPHIRERAPIDS, "sapphirerapids"),
    INTEL_CPU(GRANITERAPIDS, "graniterapids"),
    INTEL_GPU(BDW, "bdw"),
    INTEL_GPU(SKL, "skl"),
    INTEL_GPU(KBL, "kbl"),
    INTEL_GPU(CFL, "cfl"),
    INTEL_GPU(APL, "apl"),
    INTEL_GPU(BXT, "bxt"),
    INTEL_GPU(GLK, "glk"),
    INTEL_GPU(WHL, "whl"),
    INTEL_GPU(AML, "aml"),
    INTEL_GPU(CML, "cml"),
    INTEL_GPU(ICLLP, "icllp"),
    INTEL_GPU(ICL, "icl"),
    INTEL_GPU(EHL, "ehl"),
    INTEL_GPU(JSL, "jsl"),
    INTEL_GPU(TGLLP, "tgllp"),
    INTEL_GPU(TGL, "tgl"),
    INTEL_GPU(RKL, "rkl"),
    INTEL_GPU(ADL_S, "adl_s"),
    INTEL_GPU(RPL_S, "rpl_s"),
    INTEL_GPU(ADL_P, "adl_p"),
    INTEL_GPU(ADL_N, "adl_n"),
    INTEL_GPU(DG1, "dg1"),
    INTEL_GPU(DG2, "dg2"),
    INTEL_GPU(ACM_G10, "acm_g10"),
    INTEL_GPU(DG2_G10, "dg2_g10"),
    INTEL_GPU(ACM_G11, "acm_g11"),
    INTEL_GPU(DG2_G11, "dg2_g11"),
    INTEL_GPU(ACM_G12, "acm_g12"),
    INTEL_GPU(DG2_G12, "dg2_g12"),
    INTEL_GPU(PVC, "pvc"),
    INTEL_GPU(PVC_VG, "pvc_vg"),
    INTEL_GPU(MTL, "mtl"),
    INTEL_GPU(MTL_U, "mtl_u"),
    INTEL_GPU(MTL_S, "mtl_s"),
    INTEL_GPU(ARL_U, "arl_u"),
    INTEL_GPU(ARL_S, "arl_s"),
    INTEL_GPU(MTL_H, "mtl_h"),
    INTEL_GPU(ARL_H, "arl_h"),
    INTEL_GPU(BMG, "bmg"),
    INTEL_GPU(BMG_G21, "bmg_g21"),
    INTEL_GPU(PTL, "ptl"),
    INTEL_GPU(LNL_M, "lnl_m"),
    // clang-format on
};
#undef INTEL_CPU
#undef INTEL_GPU

static const IntelArchNameMap *lookupIntelArch(OffloadArch::TargetArch V,
                                               OffloadArch::IntelArch Arch) {
  for (const IntelArchNameMap &Entry : IntelArchNames)
    if (Entry.V == V && Entry.Arch == Arch)
      return &Entry;
  return nullptr;
}

static const IntelArchNameMap *lookupIntelArch(llvm::StringRef Name) {
  for (const IntelArchNameMap &Entry : IntelArchNames)
    if (Name == Entry.Name)
      return &Entry;
  return nullptr;
}

const char *OffloadArchToString(OffloadArch A) {
  switch (A.targetArch()) {
  case OffloadArch::TargetArch::Unused:
    return "";
  case OffloadArch::TargetArch::Unknown:
    return "unknown";
  case OffloadArch::TargetArch::NVPTX:
    return llvm::NVPTX::getArchName(A.nvptxKind()).data();
  case OffloadArch::TargetArch::AMDGPU:
    return llvm::AMDGPU::getArchNameAMDGCN(A.amdgpuKind()).data();
  case OffloadArch::TargetArch::SPIRV:
    return "amdgcnspirv";
  case OffloadArch::TargetArch::IntelCPU:
  case OffloadArch::TargetArch::IntelGPU: {
    const IntelArchNameMap *Entry =
        lookupIntelArch(A.targetArch(), A.intelKind());
    return Entry ? Entry->Name : "unknown";
  }
  case OffloadArch::TargetArch::Generic:
    return "generic";
  }
  return "unknown";
}

const char *OffloadArchToVirtualArchString(OffloadArch A) {
  switch (A.targetArch()) {
  case OffloadArch::TargetArch::NVPTX:
    return llvm::NVPTX::getVirtualArch(A.nvptxKind()).data();
  case OffloadArch::TargetArch::AMDGPU:
  case OffloadArch::TargetArch::SPIRV:
    return "compute_amdgcn";
  case OffloadArch::TargetArch::Unknown:
    return "unknown";
  case OffloadArch::TargetArch::Unused:
  case OffloadArch::TargetArch::IntelCPU:
  case OffloadArch::TargetArch::IntelGPU:
  case OffloadArch::TargetArch::Generic:
    return "";
  }
  return "unknown";
}

OffloadArch StringToOffloadArch(llvm::StringRef S) {
  // The empty string denotes the "unused" architecture.
  if (S.empty())
    return OffloadArch::getUnused();

  // Non-GPU-table pseudo/sentinel architectures.
  if (S == "amdgcnspirv")
    return OffloadArch::getSPIRV();
  if (S == "generic")
    return OffloadArch::getGeneric();
  if (const IntelArchNameMap *Entry = lookupIntelArch(S))
    return OffloadArch::getIntel(Entry->V, Entry->Arch);

  // Otherwise defer to the vendor TargetParser GPU lists.
  if (llvm::NVPTX::GPUKind NV = llvm::NVPTX::parseArch(S))
    return OffloadArch::getNVPTX(NV);
  if (llvm::AMDGPU::GPUKind AK = llvm::AMDGPU::parseArchAMDGCN(S))
    return OffloadArch::getAMDGPU(AK);
  return OffloadArch::getUnknown();
}

void fillValidOffloadArchList(llvm::SmallVectorImpl<llvm::StringRef> &Values) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  Values.push_back(NAME);
#include "llvm/TargetParser/NVPTXTargetParser.def"
  llvm::AMDGPU::fillValidArchListAMDGCN(Values, llvm::Triple::NoSubArch);
}

llvm::Triple OffloadArchToTriple(const llvm::Triple &DefaultToolchainTriple,
                                 OffloadArch ID) {
  if (ID.isSPIRV())
    return llvm::Triple(llvm::Triple::spirv64, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);

  if (ID.isNVPTX()) {
    llvm::Triple::ArchType Arch = DefaultToolchainTriple.isArch64Bit()
                                      ? llvm::Triple::nvptx64
                                      : llvm::Triple::nvptx;
    return llvm::Triple(Arch, llvm::Triple::NoSubArch, llvm::Triple::NVIDIA,
                        llvm::Triple::CUDA);
  }

  if (ID.isAMDGPU())
    return llvm::Triple("amdgcn-amd-amdhsa");

  if (ID.isIntelCPU())
    return llvm::Triple("spir64_x86_64-unknown-unknown");

  if (ID.isIntelGPU())
    return llvm::Triple("spir64_gen-unknown-unknown");

  return {};
}

bool IsAMDGenericGPUArch(OffloadArch Arch) {
  return Arch.isAMDGPU() && llvm::AMDGPU::isPseudoTarget(Arch.amdgpuKind());
}

bool IsSYCLSupportedAMDGPUArch(OffloadArch Arch) {
  return Arch.isAMDGPU() && !IsAMDGenericGPUArch(Arch);
}

bool IsSYCLSupportedNVidiaGPUArch(OffloadArch Arch) {
  if (!Arch.isNVPTX())
    return false;
  unsigned SmVersion = llvm::NVPTX::getSmVersion(Arch.nvptxKind());
  return SmVersion >= 500 && SmVersion <= 900;
}

} // namespace clang
