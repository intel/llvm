//===- LLVMSPIRVOpts.cpp - Defines LLVM/SPIR-V options ----------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2021 Intel Corporation. All rights reserved.
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
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
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
/// \file
///
/// This file provides definitions for LLVM/SPIR-V Translator's CLI
/// functionality.
///
//===----------------------------------------------------------------------===//

#include "LLVMSPIRVOpts.h"

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IntrinsicInst.h>

using namespace llvm;
using namespace SPIRV;

bool TranslatorOpts::isUnknownIntrinsicAllowed(IntrinsicInst *II) const
    noexcept {
  if (!SPIRVAllowUnknownIntrinsics.hasValue())
    return false;
  const auto &IntrinsicPrefixList = SPIRVAllowUnknownIntrinsics.getValue();
  StringRef IntrinsicName = II->getCalledOperand()->getName();
  for (const auto Prefix : IntrinsicPrefixList) {
    if (IntrinsicName.startswith(Prefix)) // Also true if `Prefix` is empty
      return true;
  }
  return false;
}

bool TranslatorOpts::isSPIRVAllowUnknownIntrinsicsEnabled() const noexcept {
  return SPIRVAllowUnknownIntrinsics.hasValue();
}

void TranslatorOpts::setSPIRVAllowUnknownIntrinsics(
    TranslatorOpts::ArgList IntrinsicPrefixList) noexcept {
  SPIRVAllowUnknownIntrinsics = IntrinsicPrefixList;
}
