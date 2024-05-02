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

#include <compiler/utils/attributes.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <optional>

namespace compiler {
namespace utils {
using namespace llvm;

static constexpr const char *MuxKernelAttrName = "mux-kernel";

void setIsKernel(Function &F) { F.addFnAttr(MuxKernelAttrName, ""); }

void setIsKernelEntryPt(Function &F) {
  F.addFnAttr(MuxKernelAttrName, "entry-point");
}

bool isKernel(const Function &F) {
  return F.getFnAttribute(MuxKernelAttrName).isValid();
}

bool isKernelEntryPt(const Function &F) {
  const Attribute Attr = F.getFnAttribute(MuxKernelAttrName);
  if (Attr.isValid()) {
    return Attr.getValueAsString() == "entry-point";
  }
  return false;
}

void dropIsKernel(Function &F) { F.removeFnAttr(MuxKernelAttrName); }

void takeIsKernel(Function &ToF, Function &FromF) {
  if (!isKernel(FromF)) {
    return;
  }
  // Check whether we need to add entry-point data.
  const bool IsEntryPt = isKernelEntryPt(FromF);
  // Drop all data for simplicity
  dropIsKernel(ToF);
  dropIsKernel(FromF);
  // Add the new data
  IsEntryPt ? setIsKernelEntryPt(ToF) : setIsKernel(ToF);
}

static StringRef getFnNameFromAttr(const Function &F, StringRef AttrName) {
  const Attribute Attr = F.getFnAttribute(AttrName);
  if (Attr.isValid()) {
    return Attr.getValueAsString();
  }
  return "";
}

static constexpr const char *OrigFnNameAttr = "mux-orig-fn";

void setOrigFnName(Function &F) { F.addFnAttr(OrigFnNameAttr, F.getName()); }

StringRef getOrigFnName(const Function &F) {
  return getFnNameFromAttr(F, OrigFnNameAttr);
}

StringRef getOrigFnNameOrFnName(const Function &F) {
  auto N = getFnNameFromAttr(F, OrigFnNameAttr);
  return N.empty() ? F.getName() : N;
}

static constexpr const char *BaseFnNameAttr = "mux-base-fn-name";

void setBaseFnName(Function &F, StringRef N) { F.addFnAttr(BaseFnNameAttr, N); }

StringRef getBaseFnName(const Function &F) {
  return getFnNameFromAttr(F, BaseFnNameAttr);
}

StringRef getBaseFnNameOrFnName(const Function &F) {
  auto N = getFnNameFromAttr(F, BaseFnNameAttr);
  return N.empty() ? F.getName() : N;
}

StringRef getOrSetBaseFnName(Function &F, const Function &SetFromF) {
  const Attribute Attr = F.getFnAttribute(BaseFnNameAttr);
  if (Attr.isValid()) {
    return Attr.getValueAsString();
  }

  // Try and peer through the original function's name
  StringRef BaseFnName = getBaseFnNameOrFnName(SetFromF);
  F.addFnAttr(BaseFnNameAttr, BaseFnName);
  setBaseFnName(F, BaseFnName);
  return BaseFnName;
}

static std::optional<int> getStringFnAttrAsInt(const Attribute &Attr) {
  if (Attr.isValid()) {
    int AttrValue = 0;
    if (!Attr.getValueAsString().getAsInteger(10, AttrValue)) {
      return AttrValue;
    }
  }
  return std::nullopt;
}

static constexpr const char *LocalMemUsageAttrName = "mux-local-mem-usage";

void setLocalMemoryUsage(Function &F, uint64_t LocalMemUsage) {
  const Attribute Attr = Attribute::get(F.getContext(), LocalMemUsageAttrName,
                                        itostr(LocalMemUsage));
  F.addFnAttr(Attr);
}

std::optional<uint64_t> getLocalMemoryUsage(const Function &F) {
  const Attribute Attr = F.getFnAttribute(LocalMemUsageAttrName);
  auto Val = getStringFnAttrAsInt(Attr);
  // Only return non-negative integers
  return Val && Val >= 0 ? std::optional<uint64_t>(*Val) : std::nullopt;
}

static constexpr const char *DMAReqdSizeBytesAttrName = "mux-dma-reqd-size";

void setDMAReqdSizeBytes(Function &F, uint32_t DMASizeBytes) {
  const Attribute Attr = Attribute::get(
      F.getContext(), DMAReqdSizeBytesAttrName, itostr(DMASizeBytes));
  F.addFnAttr(Attr);
}

std::optional<uint32_t> getDMAReqdSizeBytes(const Function &F) {
  const Attribute Attr = F.getFnAttribute(DMAReqdSizeBytesAttrName);
  auto Val = getStringFnAttrAsInt(Attr);
  // Only return non-negative integers
  return Val && Val >= 0 ? std::optional<uint32_t>(*Val) : std::nullopt;
}

static constexpr const char *BarrierScheduleAttrName = "mux-barrier-schedule";

void setBarrierSchedule(CallInst &CI, BarrierSchedule Sched) {
  StringRef Val;
  switch (Sched) {
    default:
    case BarrierSchedule::Unordered:
      Val = "unordered";
      break;
    case BarrierSchedule::Once:
      Val = "once";
      break;
    case BarrierSchedule::ScalarTail:
      Val = "scalar-tail";
      break;
    case BarrierSchedule::Linear:
      Val = "linear";
      break;
  }

  const Attribute Attr =
      Attribute::get(CI.getContext(), BarrierScheduleAttrName, Val);
  CI.addFnAttr(Attr);
}

BarrierSchedule getBarrierSchedule(const CallInst &CI) {
  const Attribute Attr = CI.getFnAttr(BarrierScheduleAttrName);
  if (Attr.isValid()) {
    return StringSwitch<BarrierSchedule>(Attr.getValueAsString())
        .Case("once", BarrierSchedule::Once)
        .Case("scalar-tail", BarrierSchedule::ScalarTail)
        .Case("linear", BarrierSchedule::Linear)
        .Default(BarrierSchedule::Unordered);
  }
  return BarrierSchedule::Unordered;
}

static constexpr const char *MuxDegenerateSubgroupsAttrName =
    "mux-degenerate-subgroups";

void setHasDegenerateSubgroups(Function &F) {
  F.addFnAttr(MuxDegenerateSubgroupsAttrName);
}

bool hasDegenerateSubgroups(const Function &F) {
  const Attribute Attr = F.getFnAttribute(MuxDegenerateSubgroupsAttrName);
  return Attr.isValid();
}

static constexpr const char *MuxNoSubgroupsAttrName = "mux-no-subgroups";

void setHasNoExplicitSubgroups(Function &F) {
  F.addFnAttr(MuxNoSubgroupsAttrName);
}

bool hasNoExplicitSubgroups(const Function &F) {
  const Attribute Attr = F.getFnAttribute(MuxNoSubgroupsAttrName);
  return Attr.isValid();
}

unsigned getMuxSubgroupSize(const llvm::Function &) {
  // FIXME: The mux sub-group size is currently assumed to be 1 for all
  // functions, kerrnels, and targets. This helper function is just to avoid
  // hard-coding the constant 1 in places that will eventually need updated.
  return 1;
}
}  // namespace utils
}  // namespace compiler
