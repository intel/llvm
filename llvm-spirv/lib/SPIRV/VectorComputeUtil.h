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
#include "spirv.hpp"

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
