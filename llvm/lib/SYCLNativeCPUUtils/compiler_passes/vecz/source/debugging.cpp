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

#include "debugging.h"

#include <llvm/Analysis/OptimizationRemarkEmitter.h>

using namespace llvm;

namespace vecz {

/// @brief Create the std::string containing the message for the remark
///
/// @param[in] V The value (can be `nullptr`) to be included in the remark
/// @param[in] Msg The main remark message
/// @param[in] Note An optional additional note to provide more context/info.
/// @return The remark message as it is to be printed
static std::string createRemarkMessage(const Value *V, StringRef Msg,
                                       StringRef Note = "") {
  std::string helper_str("Vecz: ");
  raw_string_ostream helper_stream(helper_str);
  helper_stream << Msg;
  if (V) {
    if (isa<Instruction>(V)) {
      // Instructions are already prefixed by two spaces when printed
      V->print(helper_stream, /*IsForDebug=*/true);
    } else if (const Function *F = dyn_cast<Function>(V)) {
      // Printing a functions leads to its whole body being printed
      helper_stream << " function \"" << F->getName() << "\"";
    } else {
      helper_stream << " ";
      V->print(helper_stream, /*IsForDebug=*/true);
    }
  }
  helper_stream << '\n';

  // Provide extra context, if supplied
  if (!Note.empty()) {
    helper_stream << "  note: " << Note << '\n';
  }

  return helper_stream.str();
}

void emitVeczRemarkMissed(const Function *F, const Value *V, StringRef Msg,
                          StringRef Note) {
  const Instruction *I = V ? dyn_cast<Instruction>(V) : nullptr;
  auto RemarkMsg = createRemarkMessage(V, Msg, Note);
  OptimizationRemarkEmitter ORE(F);
  if (I) {
    ORE.emit(OptimizationRemarkMissed("vecz", "vecz", I) << RemarkMsg);
  } else {
    const DebugLoc D = I ? DebugLoc(I->getDebugLoc()) : DebugLoc();
    ORE.emit(OptimizationRemarkMissed("vecz", "vecz", D, &(F->getEntryBlock()))
             << RemarkMsg);
  }
}

void emitVeczRemarkMissed(const Function *F, StringRef Msg, StringRef Note) {
  emitVeczRemarkMissed(F, nullptr, Msg, Note);
}

void emitVeczRemark(const Function *F, const Value *V, StringRef Msg) {
  const Instruction *I = V ? dyn_cast<Instruction>(V) : nullptr;
  const DebugLoc D = I ? DebugLoc(I->getDebugLoc()) : DebugLoc();

  auto RemarkMsg = createRemarkMessage(V, Msg);
  OptimizationRemarkEmitter ORE(F);
  ORE.emit(OptimizationRemark("vecz", "vecz", F) << RemarkMsg);
}

void emitVeczRemark(const Function *F, StringRef Msg) {
  emitVeczRemark(F, nullptr, Msg);
}
}  // namespace vecz
