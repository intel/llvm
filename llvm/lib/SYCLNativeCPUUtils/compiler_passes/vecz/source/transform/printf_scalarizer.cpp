// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "transform/printf_scalarizer.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cstddef>
#include <cstdint>
#include <string>

#define DEBUG_TYPE "VECZ-PRINTF-SCALARIZER"

using namespace llvm;

namespace vecz {

GlobalVariable *GetFormatStringAsValue(Value *op) {
  if (isa<ConstantExpr>(op)) {
    auto const_string = cast<ConstantExpr>(op);
    if (const_string->getOpcode() != Instruction::GetElementPtr) {
      return nullptr;
    }
    return dyn_cast<GlobalVariable>(const_string->getOperand(0));
  }

  if (isa<GetElementPtrInst>(op)) {
    auto gep_string = cast<GetElementPtrInst>(op);
    return dyn_cast<GlobalVariable>(gep_string->getPointerOperand());
  }

  return dyn_cast<GlobalVariable>(op);
}

std::string GetFormatStringAsString(Value *op) {
  if (!op || !isa<GlobalVariable>(op)) {
    return "";
  }

  auto *string_global = cast<GlobalVariable>(op);

  if (!string_global->hasInitializer()) {
    return "";
  }

  Constant *const string_const = string_global->getInitializer();

  if (!isa<ConstantDataSequential>(string_const)) {
    return "";
  }

  auto *array_string = cast<ConstantDataSequential>(string_const);

  if (!array_string->isString()) {
    return "";
  }

  return array_string->getAsString().str();
}

bool IncrementPtr(const char **fmt) {
  if (*(++(*fmt)) == '\0') {
    return true;
  }
  return false;
}

GlobalVariable *GetNewFormatStringAsGlobalVar(
    Module &module, GlobalVariable *const string_value,
    const std::string &new_format_string) {
  const ArrayRef<uint8_t> Elts((uint8_t *)new_format_string.data(),
                               new_format_string.size());
  Constant *new_format_string_const =
      ConstantDataArray::get(module.getContext(), Elts);

  const bool is_constant = string_value->isConstant();
  const bool is_externally_initialized = false;
  const uint32_t addr_space = string_value->getType()->getPointerAddressSpace();
  const GlobalValue::LinkageTypes linkage_type = string_value->getLinkage();
  const GlobalValue::ThreadLocalMode thread_local_mode =
      string_value->getThreadLocalMode();

  GlobalVariable *new_var = new GlobalVariable(
      module, new_format_string_const->getType(), is_constant, linkage_type,
      new_format_string_const, Twine(string_value->getName() + "_"),
      string_value, thread_local_mode, addr_space, is_externally_initialized);

  new_var->setAlignment(MaybeAlign(string_value->getAlignment()));
  new_var->setUnnamedAddr(string_value->getUnnamedAddr());

  return new_var;
}

EnumPrintfError ScalarizeAndCheckFormatString(const std::string &str,
                                              std::string &new_str) {
  // Set some sensible defaults in case we return error
  new_str = "";

  const char *fmt = str.c_str();

  while (*fmt != '\0') {
    if (*fmt != '%') {
      new_str += *fmt;
    } else {
      std::string specifier_string(1, *fmt);

      if (IncrementPtr(&fmt)) {
        LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                          << str.c_str() << "\"");
        return kPrintfError_invalidFormatString;
      }

      // Parse (zero or more) Flags
      const char *flag_chars = "-+ #0";
      while (strchr(flag_chars, *fmt)) {
        specifier_string += *fmt;
        if (IncrementPtr(&fmt)) {
          LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                            << str.c_str() << "\"");
          return kPrintfError_invalidFormatString;
        }
      }

      // Parse (optional) Width
      if (*fmt == '*') {
        specifier_string += *fmt;
        if (IncrementPtr(&fmt)) {
          LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                            << str.c_str() << "\"");
          return kPrintfError_invalidFormatString;
        }
      } else if (isdigit(*fmt)) {
        while (isdigit(*fmt)) {
          specifier_string += *fmt;
          if (IncrementPtr(&fmt)) {
            LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                              << str.c_str() << "\"");
            return kPrintfError_invalidFormatString;
          }
        }
      }

      // Parse (optional) Precision
      if (*fmt == '.') {
        specifier_string += *fmt;
        if (IncrementPtr(&fmt)) {
          LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                            << str.c_str() << "\"");
          return kPrintfError_invalidFormatString;
        }

        while (isdigit(*fmt)) {
          specifier_string += *fmt;
          if (IncrementPtr(&fmt)) {
            LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                              << str.c_str() << "\"");
            return kPrintfError_invalidFormatString;
          }
        }
      }

      uint32_t vector_length = 1u;
      const bool is_vector = *fmt == 'v';
      // Parse (optional) Vector Specifier
      if (is_vector) {
        if (IncrementPtr(&fmt)) {
          LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                            << str.c_str() << "\"");
          return kPrintfError_invalidFormatString;
        }
        switch (*fmt) {
          default:
            LLVM_DEBUG(dbgs() << "Unexpected character in format string \""
                              << str.c_str() << "\"");
            return kPrintfError_invalidFormatString;
          case '1':
            // Must be 16, else error
            if (IncrementPtr(&fmt)) {
              LLVM_DEBUG(dbgs()
                         << "Expected vector width of 16 in format string \""
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            if (*fmt != '6') {
              LLVM_DEBUG(dbgs()
                         << "Expected vector width of 16 in format string \""
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            vector_length = 16u;
            break;
          case '2':
            vector_length = 2u;
            break;
          case '3':
            vector_length = 3u;
            // Lookahead for vectors of width 32. We know that we won't go out
            // of bounds because worst case scenario there should be a null byte
            // after the '3'.
            if (*(fmt + 1) == '2') {
              IncrementPtr(&fmt);
              vector_length = 32u;
            }
            break;
          case '4':
            vector_length = 4u;
            break;
          case '6':
            // Must be 64, else error
            if (IncrementPtr(&fmt)) {
              LLVM_DEBUG(dbgs()
                         << "Expected vector width of 64 in format string \""
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            if (*fmt != '4') {
              LLVM_DEBUG(dbgs()
                         << "Expected vector width of 64 in format string \""
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            vector_length = 64u;
            break;
          case '8':
            vector_length = 8u;
            break;
        }
        if (IncrementPtr(&fmt)) {
          LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                            << str.c_str() << "\"");
          return kPrintfError_invalidFormatString;
        }
      }

      // Parse Length Modifier
      const char *length_modifier_chars = "hljztL";
      // Length Modifier is required with Vector Specifier
      bool has_used_l_length_modifier = false;
      const bool has_supplied_length_modifier =
          strchr(length_modifier_chars, *fmt);
      if (is_vector && !has_supplied_length_modifier) {
        LLVM_DEBUG(
            dbgs() << "Expected vector width specifier in format string \""
                   << str.c_str() << "\"");
        return kPrintfError_invalidFormatString;
      }

      if (has_supplied_length_modifier) {
        bool consume_next_char = true;
        switch (*fmt) {
          default:
            // The 'j', 'z', 't', and 'L' length modifiers are not supported by
            // OpenCL C.
            LLVM_DEBUG(dbgs() << "Unsupported length modifier '" << *fmt
                              << "'specifier in format string \"" << str.c_str()
                              << "\"");
            return kPrintfError_invalidFormatString;
          case 'h':
            if (IncrementPtr(&fmt)) {
              LLVM_DEBUG(dbgs()
                         << "Unexpected \\0 character in format string \""
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            if (*fmt == 'h') {
              specifier_string += "hh";
            } else if (*fmt == 'l') {
              // Native printf doesn't recognize 'hl' so we don't
              // add it to the new format string.  Luckily, 'hl'
              // is sizeof(int) - the same as the default on
              // native printf!

              // Additionally, 'hl' modifier may only be used in
              // conjunction with the vector specifier
              if (!is_vector) {
                LLVM_DEBUG(dbgs()
                           << "Unexpected \\0 character in format string \""
                           << str.c_str() << "\"");
                return kPrintfError_invalidFormatString;
              }
            } else {
              specifier_string += 'h';
              // We've already incremented the ptr and we found nothing; don't
              // do it again
              consume_next_char = false;
            }
            break;
          case 'l':
            specifier_string += *fmt;
            // Check ahead to see if the user is using the invalid 'll' length
            // modifier
            if (IncrementPtr(&fmt)) {
              LLVM_DEBUG(dbgs()
                         << "Unexpected \\0 character in format string \""
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            if (*fmt == 'l') {
              LLVM_DEBUG(dbgs()
                         << "The 'll' length specifier is invalid in OpenCL "
                            "printf\n  > "
                         << str.c_str() << "\"");
              return kPrintfError_invalidFormatString;
            }
            // We've already incremented the ptr; don't do it again

            // The 'l' specifier for the OpenCL printf expects 64 bits
            // integers, check if the system's long are actually 64 bits wide
            // and if not upgrade the format specifier to 'll'.
            //
            // FIXME: This only works for host based devices, which is fine for
            // our current printf implementation, but it should really be
            // removed once we have a proper printf implementation.
            if (sizeof(long) != 8) {
              specifier_string += 'l';
            }

            consume_next_char = false;
            has_used_l_length_modifier = true;
            break;
        }
        if (consume_next_char) {
          if (IncrementPtr(&fmt)) {
            LLVM_DEBUG(dbgs() << "Unexpected \\0 character in format string \""
                              << str.c_str() << "\"");
            return kPrintfError_invalidFormatString;
          }
        }
      }

      // Parse Specifier
      specifier_string += *fmt;

      switch (*fmt) {
        default:
          break;
        case 'n':
          // The 'n' conversion specifier is not supported by OpenCL C.
          LLVM_DEBUG(dbgs()
                     << "The 'n' conversion specifier is invalid in OpenCL "
                        "printf\n  > "
                     << str.c_str() << "\"");
          return kPrintfError_invalidFormatString;
        case 's':  // Intentional fall-through
        case 'c':
          // The 'l' length modifier followed by the 'c' or 's' conversion
          // specifiers is not supported by OpenCL C.
          if (has_used_l_length_modifier) {
            LLVM_DEBUG(dbgs()
                       << "The 'l' length modifier followed by the 'c' or "
                          "'s' conversion is invalid in OpenCL printf\n  > "
                       << str.c_str() << "\"");
            return kPrintfError_invalidFormatString;
          }
          break;
      }

      // Output the %specifier for each element of the vector,
      // and for every element but the last, follow it by a "," string.
      for (uint32_t i = 0; i < vector_length; ++i) {
        new_str += specifier_string;

        if (i < (vector_length - 1)) {
          new_str += ",";
        }
      }
    }
    ++fmt;
  }

  new_str += '\0';

  return kPrintfError_success;
}
}  // namespace vecz
