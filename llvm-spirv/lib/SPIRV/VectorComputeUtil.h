//=- VectorComputeUtil.h - vector compute utilities declarations -*- C++ -*-=//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2020 Intel Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Intel Corporation, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
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
#include "spirv/unified1/spirv.hpp"

namespace VectorComputeUtil {

///////////////////////////////////////////////////////////////////////////////
//
// Types
//
///////////////////////////////////////////////////////////////////////////////

enum VCFloatType {
  Double,
  Float,
  Half,
};

FPRoundingMode getFPRoundingMode(unsigned FloatControl) noexcept;
FPDenormMode getFPDenormMode(unsigned FloatControl,
                             VCFloatType FloatType) noexcept;
FPOperationMode getFPOperationMode(unsigned FloatControl) noexcept;

unsigned getVCFloatControl(FPRoundingMode RoundMode) noexcept;
unsigned getVCFloatControl(FPOperationMode FloatMode) noexcept;
unsigned getVCFloatControl(FPDenormMode DenormMode,
                           VCFloatType FloatType) noexcept;

typedef SPIRV::SPIRVMap<FPRoundingMode, spv::ExecutionMode>
    FPRoundingModeExecModeMap;
typedef SPIRV::SPIRVMap<FPOperationMode, spv::ExecutionMode>
    FPOperationModeExecModeMap;
typedef SPIRV::SPIRVMap<FPDenormMode, spv::ExecutionMode>
    FPDenormModeExecModeMap;
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

std::string getVCBufferSurfaceName();
std::string getVCBufferSurfaceName(SPIRVAccessQualifierKind Access);

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
const static char VCSIMTCall[] = "VCSIMTCall";
const static char VCArgumentKind[] = "VCArgumentKind";
const static char VCArgumentDesc[] = "VCArgumentDesc";
const static char VCCallable[] = "VCCallable";
const static char VCSingleElementVector[] = "VCSingleElementVector";
const static char VCFCEntry[] = "VCFCEntry";
} // namespace kVCMetadata

namespace kVCType {
const static char VCBufferSurface[] = "intel.buffer";
}

///////////////////////////////////////////////////////////////////////////////
//
// Map definitions
//
///////////////////////////////////////////////////////////////////////////////

namespace SPIRV {
template <>
inline void SPIRVMap<spv::FPRoundingMode, spv::ExecutionMode>::init() {
  add(spv::FPRoundingModeRTE, spv::ExecutionModeRoundingModeRTE);
  add(spv::FPRoundingModeRTZ, spv::ExecutionModeRoundingModeRTZ);
  add(spv::FPRoundingModeRTP, spv::ExecutionModeRoundingModeRTPINTEL);
  add(spv::FPRoundingModeRTN, spv::ExecutionModeRoundingModeRTNINTEL);
}
template <>
inline void SPIRVMap<spv::FPDenormMode, spv::ExecutionMode>::init() {
  add(spv::FPDenormModeFlushToZero, spv::ExecutionModeDenormFlushToZero);
  add(spv::FPDenormModePreserve, spv::ExecutionModeDenormPreserve);
}
template <>
inline void SPIRVMap<spv::FPOperationMode, spv::ExecutionMode>::init() {
  add(spv::FPOperationModeIEEE, spv::ExecutionModeFloatingPointModeIEEEINTEL);
  add(spv::FPOperationModeALT, spv::ExecutionModeFloatingPointModeALTINTEL);
}
template <>
inline void SPIRVMap<VectorComputeUtil::VCFloatType, unsigned>::init() {
  add(VectorComputeUtil::Double, 64);
  add(VectorComputeUtil::Float, 32);
  add(VectorComputeUtil::Half, 16);
}
} // namespace SPIRV

#endif // SPIRV_VCUTIL_H
