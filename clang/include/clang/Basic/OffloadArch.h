//===--- OffloadArch.h - Definition of offloading architectures --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OFFLOADARCH_H
#define LLVM_CLANG_BASIC_OFFLOADARCH_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <tuple>

namespace llvm {
template <typename T> class SmallVectorImpl;
namespace NVPTX {
enum GPUKind : uint8_t;
}
namespace AMDGPU {
enum GPUKind : uint8_t;
}
namespace IntelGPU {
enum GPUKind : uint8_t;
}
} // namespace llvm

namespace clang {

/// A processor an offloading action can target. This is a tagged handle pairing
/// a TargetArch with the matching TargetParser GPU kind; it does not enumerate
/// individual GPUs, so new NVPTX/AMDGPU targets are added in the TargetParser
/// data alone.
class OffloadArch {
public:
  enum class TargetArch : uint8_t {
    Unused,      // Default-constructed; no architecture bound.
    Unknown,     // A name that matched no known architecture.
    NVPTX,       // Kind is an llvm::NVPTX::GPUKind.
    AMDGPU,      // Kind is an llvm::AMDGPU::GPUKind.
    AMDGCNSPIRV, // The 'amdgcnspirv' pseudo target.
    IntelCPU,    // Kind is an IntelArch, or unused for getIntelCPU().
    IntelGPU,    // Kind is an llvm::IntelGPU::GPUKind or an IntelArch.
    Generic,     // The 'generic' processor model.
  };

  // Intel architectures named by their SYCL spellings (e.g. "pvc", "bmg_g21",
  // "skylake-avx512"), which the TargetParser Intel GPU list does not cover.
  enum class IntelArch : uint32_t {
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
    DG2,
    ACM_G10,
    DG2_G10,
    ACM_G11,
    DG2_G11,
    ACM_G12,
    DG2_G12,
    PVC,
    PVC_VG,
    MTL,
    MTL_U,
    MTL_S,
    ARL_U,
    ARL_S,
    MTL_H,
    ARL_H,
    BMG,
    BMG_G21,
    PTL,
    LNL_M,
  };

  // IntelArch kinds are stored offset by this base so they never collide with
  // an llvm::IntelGPU::GPUKind (a uint8_t) or the kind-less getIntelCPU().
  static constexpr uint32_t IntelArchKindBase = 0x100;

private:
  // Interpreted according to V; unused for the tagless TargetArch values.
  uint32_t Kind = 0;
  TargetArch V = TargetArch::Unused;

  constexpr OffloadArch(TargetArch V, uint32_t Kind) : Kind(Kind), V(V) {}

public:
  constexpr OffloadArch() = default;

  static OffloadArch getNVPTX(llvm::NVPTX::GPUKind K) {
    return {TargetArch::NVPTX, static_cast<uint32_t>(K)};
  }
  static OffloadArch getAMDGPU(llvm::AMDGPU::GPUKind K) {
    return {TargetArch::AMDGPU, static_cast<uint32_t>(K)};
  }
  /// An Intel GPU, named by its entry in the TargetParser device list.
  static OffloadArch getIntelGPU(llvm::IntelGPU::GPUKind K) {
    return {TargetArch::IntelGPU, static_cast<uint32_t>(K)};
  }
  /// Intel CPU offload, which has a single target, so no kind is needed.
  static constexpr OffloadArch getIntelCPU() {
    return {TargetArch::IntelCPU, 0};
  }
  /// An Intel CPU or GPU named by one of the IntelArch spellings.
  static constexpr OffloadArch getIntel(TargetArch V, IntelArch A) {
    return {V, IntelArchKindBase + static_cast<uint32_t>(A)};
  }
  static constexpr OffloadArch getUnused() { return {TargetArch::Unused, 0}; }
  static constexpr OffloadArch getUnknown() { return {TargetArch::Unknown, 0}; }
  static constexpr OffloadArch getAMDGCNSPIRV() {
    return {TargetArch::AMDGCNSPIRV, 0};
  }
  static constexpr OffloadArch getGeneric() { return {TargetArch::Generic, 0}; }

  /// Default architectures used when the user does not specify one.
  static OffloadArch CudaDefault();
  static OffloadArch HIPDefault();

  TargetArch targetArch() const { return V; }

  bool isNVPTX() const { return V == TargetArch::NVPTX; }
  bool isAMDGPU() const { return V == TargetArch::AMDGPU; }
  bool isAMDGCNSPIRV() const { return V == TargetArch::AMDGCNSPIRV; }
  bool isIntelCPU() const { return V == TargetArch::IntelCPU; }
  bool isIntelGPU() const { return V == TargetArch::IntelGPU; }
  bool isIntel() const { return isIntelCPU() || isIntelGPU(); }
  bool isGeneric() const { return V == TargetArch::Generic; }
  bool isUnused() const { return V == TargetArch::Unused; }
  bool isUnknown() const { return V == TargetArch::Unknown; }

  // Only valid when isNVPTX() / isAMDGPU() respectively.
  llvm::NVPTX::GPUKind nvptxKind() const {
    return static_cast<llvm::NVPTX::GPUKind>(Kind);
  }
  llvm::AMDGPU::GPUKind amdgpuKind() const {
    return static_cast<llvm::AMDGPU::GPUKind>(Kind);
  }
  /// Which Intel GPU this is; only meaningful when isIntelGPU() and
  /// !isIntelArchKind().
  llvm::IntelGPU::GPUKind intelGPUKind() const {
    return static_cast<llvm::IntelGPU::GPUKind>(Kind);
  }
  // True when this Intel arch was named by one of the IntelArch spellings.
  bool isIntelArchKind() const {
    return isIntel() && Kind >= IntelArchKindBase;
  }
  // Only valid when isIntelArchKind().
  IntelArch intelKind() const {
    return static_cast<IntelArch>(Kind - IntelArchKindBase);
  }

  bool operator==(const OffloadArch &Other) const {
    return V == Other.V && Kind == Other.Kind;
  }
  bool operator!=(const OffloadArch &Other) const { return !(*this == Other); }

  bool operator<(const OffloadArch &Other) const {
    return std::tie(V, Kind) < std::tie(Other.V, Other.Kind);
  }
};

// Check if the given Arch value is a Generic AMD GPU.
// Currently GFX*_GENERIC AMD GPUs do not support SYCL offloading.
// This is used to filter out GFX*_GENERIC AMD GPUs in
// `IsSYCLSupportedAMDGPUArch`.
bool IsAMDGenericGPUArch(OffloadArch Arch);

// Check if the given Arch value is a valid SYCL supported AMD GPU.
bool IsSYCLSupportedAMDGPUArch(OffloadArch Arch);

// Check if the given Arch value is a valid SYCL supported NVidia GPU.
bool IsSYCLSupportedNVidiaGPUArch(OffloadArch Arch);

const char *OffloadArchToString(OffloadArch A);
const char *OffloadArchToVirtualArchString(OffloadArch A);

// Convert a string to an OffloadArch. Returns an Unknown OffloadArch if the
// string is not recognized.
OffloadArch StringToOffloadArch(llvm::StringRef S);

OffloadArch getSubArchOffloadArch(llvm::Triple::SubArchType SubArch);
llvm::Triple::SubArchType getOffloadArchSubArch(OffloadArch ID);

llvm::Triple OffloadArchToTriple(const llvm::Triple &DefaultToolchainTriple,
                                 OffloadArch ID);

/// Represents a bound architecture for offload / multiple architecture
/// compilation.
struct BoundArch {
  llvm::StringRef ArchName;

  /// The parsed offload architecture.
  /// Will be an Unknown OffloadArch if ArchName is not recognized.
  OffloadArch Arch = OffloadArch::getUnused();

  BoundArch() = default;
  explicit BoundArch(llvm::StringRef Name)
      : ArchName(Name), Arch(Name.empty() ? OffloadArch::getUnknown()
                                          : StringToOffloadArch(Name)) {}

  BoundArch(llvm::StringRef Name, OffloadArch A) : ArchName(Name), Arch(A) {}

  bool empty() const { return ArchName.empty(); }
  explicit operator bool() const { return !Arch.isUnused(); }

  bool operator==(const BoundArch &Other) const {
    return Arch == Other.Arch && ArchName == Other.ArchName;
  }

  bool operator<(const BoundArch &Other) const {
    return std::tie(Arch, ArchName) < std::tie(Other.Arch, Other.ArchName);
  }
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_OFFLOADARCH_H
