//===- SPIRVToolsUtil.h - SPIRV-Tools utility functions ------------------===//
//
//                     The LLVM/SPIRV Translator
//
// Copyright (c) 2026 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares utility functions for optional SPIRV-Tools support.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRVTOOLSUTIL_H
#define SPIRVTOOLSUTIL_H

#include <string>

namespace SPIRV {

/// Return whether this build supports SPIR-V validation, setting \p ErrMsg
/// when support is unavailable.
bool isSPIRVValidationAvailable(std::string &ErrMsg);

/// Validate a binary SPIR-V module, setting \p ErrMsg on failure.
bool validateSPIRVBinary(const std::string &Binary, std::string &ErrMsg);

} // namespace SPIRV

#endif // SPIRVTOOLSUTIL_H
