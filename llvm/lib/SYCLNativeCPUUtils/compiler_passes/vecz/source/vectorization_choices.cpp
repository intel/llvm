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

#include <compiler/utils/mangling.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>

#include "vecz/vecz_choices.h"

using namespace llvm;

namespace {
using namespace vecz;
static const VectorizationChoices::ChoiceInfo choicesArray[] = {
    {"PacketizeUniform", VectorizationChoices::eOptimizationPacketizeUniform,
     "Packetizes all packetizable instructions whether they are varying or "
     "not."},

    {"PacketizeUniformInLoops",
     VectorizationChoices::eOptimizationPacketizeUniformInLoops,
     "Packetizes all packetizable instructions in loops, whether they are "
     "varying or not."},

    {"InstantiateCallsInLoops",
     VectorizationChoices::eOptimizationInstantiateCallsInLoops,
     "Uses loops to instantiate call instructions, instead of duplication."},

    {"LinearizeBOSCC", VectorizationChoices::eLinearizeBOSCC,
     "Control Flow Conversion uses Branch On Superword Condition Code."},

    {"FullScalarization", VectorizationChoices::eFullScalarization,
     "The scalarization pass scalarizes everything it can, regardless of any "
     "performance benefit."},

    {"DivisionExceptions", VectorizationChoices::eDivisionExceptions,
     "Specify this when the target throws hardware exceptions on integer "
     "division by zero."},

    {"VectorPredication", VectorizationChoices::eVectorPredication,
     "Generate a vector-predicated kernel safe to run on any workgroup size, "
     "even those smaller than the vectorization width"},

    {"TargetIndependentPacketization",
     VectorizationChoices::eTargetIndependentPacketization,
     "Force target-independent packetization choices (e.g., for testing "
     "purposes)"},
};

}  // namespace

namespace vecz {

VectorizationChoices::VectorizationChoices() {}

bool VectorizationChoices::parseChoicesString(StringRef Str) {
  // If the string is empty, our work here is done
  if (Str.empty()) {
    return true;
  }

  // first = Choice, second = enable
  using ChoiceValuePair = std::pair<Choice, bool>;
  // The lexer implementation from the name mangling module is fairly generic,
  // so we will use it here.
  compiler::utils::Lexer L(Str);
  // We support multiple separators in case of platform-dependent issues
  const StringRef Separators = ":;,";
  // All the parsed choices will be stored in a set and will only be
  // enabled/disabled after the parsing has been completed successfully.
  SmallVector<ChoiceValuePair, 4> ParsedChoices;

  // Start by lexing and parsing the Choices string

  bool read_separator = false;
  do {
    StringRef ParsedChoice;
    // Strip any leading whitespace
    L.ConsumeWhitespace();
    // If we have reached the end of the string, we are done
    if (L.Left() == 0) {
      break;
    }
    // Consume the optional "no" prefix, which disables the given prefix
    const bool disable = L.Consume("no");
    // Consume the Choice name
    if (L.ConsumeAlphanumeric(ParsedChoice)) {
      // Convert the string to a Choice value
      const Choice C = fromString(ParsedChoice);
      if (C == eInvalid) {
        printChoicesParseError(Str, L.CurrentPos() - ParsedChoice.size(),
                               "Invalid Choice \"" + ParsedChoice + "\"");
        return false;
      }
      ParsedChoices.push_back(std::make_pair(C, !disable));
    } else {
      printChoicesParseError(Str, L.CurrentPos(), "Expected Choice");
      return false;
    }
    // Strip any trailing whitespace
    L.ConsumeWhitespace();
    // Consume the separator (if any)
    read_separator = false;
    auto Current = L.Current();
    if (Current != -1 && Separators.contains(char(Current))) {
      L.Consume(1);
      read_separator = true;
    }
  } while (read_separator && L.Left() > 0);

  // If there is any string left, there must be some kind of mistake
  if (L.Left() != 0) {
    printChoicesParseError(Str, L.CurrentPos(), "Expected ';'");
    return false;
  }

  // Set all the choices parsed in the loop

  for (auto C : ParsedChoices) {
    if (C.second == true) {
      enable(C.first);
    } else {
      disable(C.first);
    }
  }

  // We have finished successfully

  return true;
}

VectorizationChoices::Choice VectorizationChoices::fromString(StringRef Str) {
  auto Choose = StringSwitch<Choice>(Str);
  for (const auto &info : ArrayRef<ChoiceInfo>(choicesArray)) {
    Choose.Case(info.name, info.number);
  }
  return Choose.Default(eInvalid);
}

ArrayRef<VectorizationChoices::ChoiceInfo>
VectorizationChoices::queryAvailableChoices() {
  return ArrayRef<VectorizationChoices::ChoiceInfo>(choicesArray);
}

void VectorizationChoices::printChoicesParseError(StringRef Input,
                                                  unsigned Position,
                                                  Twine Msg) {
  errs() << "CODEPLAY_VECZ_CHOICES parsing error: " << Msg << " at position "
         << Position << "\n";
  errs() << "    " << Input << "\n    ";
  // We use the range [1, Position) instead of [0, Position - 1) to avoid
  // an underflow in the case of Position = 0
  for (unsigned i = 0; i < Position; ++i) {
    errs() << ' ';
  }
  errs() << "^\n";
}
}  // namespace vecz
