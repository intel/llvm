//=- VectorComputeUtil.h - vector compute utilities declarations -*- C++ -*-=//
//
//                     The LLVM/SPIR-V Translator
//
//
//
//===----------------------------------------------------------------------===//
//
// This file declares translation of VectorComputeUtil float control bits,
// and VC kernel metadata
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_VCUTIL_H
#define SPIRV_VCUTIL_H

#include "SPIRVInternal.h"
#include "SPIRVUtil.h"
#include "spirv.hpp"
#include "llvm/IR/Module.h"

namespace VectorComputeUtil {

///////////////////////////////////////////////////////////////////////////////
//
// Types
//
///////////////////////////////////////////////////////////////////////////////

enum VCRoundMode {
  RTE, // Round to nearest or even
  RTP, // Round towards +ve inf
  RTN, // Round towards -ve inf
  RTZ, // Round towards zero
};
enum VCDenormMode {
  FlushToZero,
  Preserve,
};
enum VCFloatMode {
  IEEE, // Single precision float IEEE mode
  ALT,  // Single precision float ALT mode
};
enum VCFloatType {
  Double,
  Float,
  Half,
};

VCRoundMode getVCRoundMode(unsigned FloatControl) noexcept;
VCDenormMode getVCDenormPreserve(unsigned FloatControl,
                                 VCFloatType FloatType) noexcept;
VCFloatMode getVCFloatMode(unsigned FloatControl) noexcept;

unsigned getVCFloatControl(VCRoundMode RoundMode) noexcept;
unsigned getVCFloatControl(VCFloatMode FloatMode) noexcept;
unsigned getVCFloatControl(VCDenormMode DenormMode,
                           VCFloatType FloatType) noexcept;

typedef SPIRV::SPIRVMap<VCRoundMode, spv::ExecutionMode> VCRoundModeExecModeMap;
typedef SPIRV::SPIRVMap<VCFloatMode, spv::ExecutionMode> VCFloatModeExecModeMap;
typedef SPIRV::SPIRVMap<VCDenormMode, spv::ExecutionMode>
    VCDenormModeExecModeMap;
typedef SPIRV::SPIRVMap<VCFloatType, unsigned> VCFloatTypeSizeMap;

///////////////////////////////////////////////////////////////////////////////
//
// Functions
//
///////////////////////////////////////////////////////////////////////////////

SPIRVStorageClassKind
getVCGlobalVarStorageClass(SPIRAddressSpace AddressSpace) noexcept;
SPIRAddressSpace
getVCGlobalVarAddressSpace(SPIRVStorageClassKind StorageClass) noexcept;

} // namespace VectorComputeUtil

///////////////////////////////////////////////////////////////////////////////
//
// Constants
//
///////////////////////////////////////////////////////////////////////////////

namespace kVCMetadata {
const static char VCFunction[] = "VCFunction";
const static char VCStackCall[] = "VCStackCall";
const static char VCArgumentIOKind[] = "VCArgumentIOKind";
const static char VCFloatControl[] = "VCFloatControl";
const static char VCSLMSize[] = "VCSLMSize";
const static char VCGlobalVariable[] = "VCGlobalVariable";
const static char VCVolatile[] = "VCVolatile";
const static char VCByteOffset[] = "VCByteOffset";
} // namespace kVCMetadata

///////////////////////////////////////////////////////////////////////////////
//
// Map definitions
//
///////////////////////////////////////////////////////////////////////////////

namespace SPIRV {
template <>
inline void
SPIRVMap<VectorComputeUtil::VCRoundMode, spv::ExecutionMode>::init() {
  add(VectorComputeUtil::RTE, spv::ExecutionModeRoundingModeRTE);
  add(VectorComputeUtil::RTZ, spv::ExecutionModeRoundingModeRTZ);
  add(VectorComputeUtil::RTP, spv::ExecutionModeRoundingModeRTPINTEL);
  add(VectorComputeUtil::RTN, spv::ExecutionModeRoundingModeRTNINTEL);
}
template <>
inline void
SPIRVMap<VectorComputeUtil::VCDenormMode, spv::ExecutionMode>::init() {
  add(VectorComputeUtil::FlushToZero, spv::ExecutionModeDenormFlushToZero);
  add(VectorComputeUtil::Preserve, spv::ExecutionModeDenormPreserve);
}
template <>
inline void
SPIRVMap<VectorComputeUtil::VCFloatMode, spv::ExecutionMode>::init() {
  add(VectorComputeUtil::IEEE, spv::ExecutionModeFloatingPointModeIEEEINTEL);
  add(VectorComputeUtil::ALT, spv::ExecutionModeFloatingPointModeALTINTEL);
}
template <>
inline void SPIRVMap<VectorComputeUtil::VCFloatType, unsigned>::init() {
  add(VectorComputeUtil::Double, 64);
  add(VectorComputeUtil::Float, 32);
  add(VectorComputeUtil::Half, 16);
}
} // namespace SPIRV

#endif // SPIRV_VCUTIL_H
