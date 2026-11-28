//===- SPIRVToolsUtil.cpp - SPIRV-Tools utility functions ----------------===//
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
/// This file defines utility functions for optional SPIRV-Tools support.
///
//===----------------------------------------------------------------------===//

#include "SPIRVToolsUtil.h"
#include "SPIRVInternal.h"

#ifdef LLVM_SPIRV_HAVE_SPIRV_TOOLS
#include "spirv-tools/libspirv.hpp"

#include <cstring>
#include <optional>
#include <vector>

#endif // LLVM_SPIRV_HAVE_SPIRV_TOOLS

namespace SPIRV {

bool isSPIRVValidationAvailable(std::string &ErrMsg) {
#ifdef LLVM_SPIRV_HAVE_SPIRV_TOOLS
  return true;
#else
  ErrMsg = "SPIR-V validation was requested, but this build lacks "
           "SPIR-V Tools validation support";
  return false;
#endif
}

#ifdef LLVM_SPIRV_HAVE_SPIRV_TOOLS

static std::optional<spv_target_env>
getSPIRVToolsTargetEnv(VersionNumber Version) {
  switch (Version) {
  case VersionNumber::SPIRV_1_0:
    return SPV_ENV_UNIVERSAL_1_0;
  case VersionNumber::SPIRV_1_1:
    return SPV_ENV_UNIVERSAL_1_1;
  case VersionNumber::SPIRV_1_2:
    return SPV_ENV_UNIVERSAL_1_2;
  case VersionNumber::SPIRV_1_3:
    return SPV_ENV_UNIVERSAL_1_3;
  case VersionNumber::SPIRV_1_4:
    return SPV_ENV_UNIVERSAL_1_4;
  case VersionNumber::SPIRV_1_5:
    return SPV_ENV_UNIVERSAL_1_5;
  case VersionNumber::SPIRV_1_6:
    return SPV_ENV_UNIVERSAL_1_6;
  default:
    return std::nullopt;
  }
}

bool validateSPIRVBinary(const std::string &Binary, std::string &ErrMsg) {
  if (Binary.size() % sizeof(uint32_t) != 0) {
    ErrMsg = "SPIR-V validation failed: binary size is not a multiple of 4 "
             "bytes";
    return false;
  }

  std::vector<uint32_t> Words(Binary.size() / sizeof(uint32_t));
  if (!Binary.empty())
    std::memcpy(Words.data(), Binary.data(), Binary.size());

  if (Words.size() < 5) {
    ErrMsg = "SPIR-V validation failed: incomplete SPIR-V header";
    return false;
  }

  const auto TargetEnv =
      getSPIRVToolsTargetEnv(static_cast<VersionNumber>(Words[1]));
  if (!TargetEnv) {
    ErrMsg = "SPIR-V validation failed: unsupported SPIR-V version";
    return false;
  }

  std::string ValidationError;
  spvtools::SpirvTools Validator(*TargetEnv);
  Validator.SetMessageConsumer(
      [&ValidationError](spv_message_level_t, const char *,
                         const spv_position_t &,
                         const char *Message) { ValidationError = Message; });
  if (Validator.Validate(Words))
    return true;

  ErrMsg = "SPIR-V validation failed";
  if (!ValidationError.empty())
    ErrMsg += ": " + ValidationError;
  return false;
}

#else

// Fallback definition for builds without libSPIRV-Tools.
bool validateSPIRVBinary(const std::string &, std::string &ErrMsg) {
  return isSPIRVValidationAvailable(ErrMsg);
}

#endif // LLVM_SPIRV_HAVE_SPIRV_TOOLS

} // namespace SPIRV
