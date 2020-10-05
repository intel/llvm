//===- SPIRVDebug.h - SPIR-V Debug Utility ----------------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
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
/// This file defines Macros and variables for debugging SPIRV.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVDEBUG_H
#define SPIRV_LIBSPIRV_SPIRVDEBUG_H

#include "SPIRVUtil.h"

#include <iostream>
#include <string>

namespace llvm {
class Module;
}

namespace SPIRV {

// Include source file and line number in error message.
extern bool SPIRVDbgErrorMsgIncludesSourceInfo;

// Enable assert or exit on error
enum class SPIRVDbgErrorHandlingKinds { Abort, Exit, Ignore };
extern SPIRVDbgErrorHandlingKinds SPIRVDbgError;

// Enable debug output.
extern bool SPIRVDbgEnable;

void verifyRegularizationPass(llvm::Module &, const std::string &);

#ifndef _SPIRVDBG
#if !defined(NDEBUG) || defined(_DEBUG)
#define _SPIRVDBG true
#else
#define _SPIRVDBG false
#endif
#endif

#if _SPIRVDBG

#define SPIRVDBG(x)                                                            \
  if (SPIRVDbgEnable) {                                                        \
    x;                                                                         \
  }

// Output stream for SPIRV debug information.
inline spv_ostream &spvdbgs() {
  return std::cerr;
}

#else

#define SPIRVDBG(x)

// Minimal std::basic_ostream mock that ignores everything being printed via
// operator<<
class dev_null_stream {
public:
  void flush() {}
};

template <typename T>
const dev_null_stream &operator<<(const dev_null_stream &Out, const T &) {
  return Out;
}

template <typename T>
const dev_null_stream &&operator<<(const dev_null_stream &&Out, const T &) {
  return std::move(Out);
}

// Output stream for SPIRV debug information.
inline dev_null_stream &spvdbgs() {
  static dev_null_stream Out;
  return Out;
}

#endif

} // namespace SPIRV
#endif // SPIRV_LIBSPIRV_SPIRVDEBUG_H
