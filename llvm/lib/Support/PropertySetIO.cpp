//==- PropertySetIO.cpp - models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PropertySetIO.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm::util;
using namespace llvm;

Expected<std::unique_ptr<PropertySetRegistry>>
PropertySetRegistry::read(const MemoryBuffer *Buf) {
  auto Res = std::make_unique<PropertySetRegistry>();
  PropertySet *CurPropSet = nullptr;
  std::error_code EC;

  for (line_iterator LI(*Buf); !LI.is_at_end(); LI++) {
    // see if this line starts a new property set
    if (LI->startswith("[")) {
      // yes - parse the category (property name)
      auto EndPos = LI->rfind(']');
      if (EndPos == StringRef::npos)
        return createStringError(EC, "invalid line: " + *LI);
      StringRef Category = LI->substr(1, EndPos - 1);
      CurPropSet = &(*Res)[Category];
      continue;
    }
    if (!CurPropSet)
      return createStringError(EC, "property category missing");
    // parse name and type+value
    auto Parts = LI->split('=');

    if (Parts.first.empty() || Parts.second.empty())
      return createStringError(EC, "invalid property line: " + *LI);
    auto TypeVal = Parts.second.split('|');

    if (TypeVal.first.empty() || TypeVal.second.empty())
      return createStringError(EC, "invalid property value: " + Parts.second);
    APInt Tint;

    // parse type
    if (TypeVal.first.getAsInteger(10, Tint))
      return createStringError(EC, "invalid property type: " + TypeVal.first);
    Expected<PropertyValue::Type> Ttag =
        PropertyValue::getTypeTag(static_cast<int>(Tint.getSExtValue()));
    StringRef Val = TypeVal.second;

    if (!Ttag)
      return Ttag.takeError();
    PropertyValue Prop(Ttag.get());

    // parse value depending on its type
    switch (Ttag.get()) {
    case PropertyValue::Type::UINT32: {
      APInt ValV;
      if (Val.getAsInteger(10, ValV))
        return createStringError(EC, "invalid property value: " + Val);
      Prop.set(static_cast<uint32_t>(ValV.getZExtValue()));
      break;
    }
    default:
      return createStringError(EC, "unsupported property type: " + Ttag.get());
    }
    (*CurPropSet)[Parts.first] = Prop;
  }
  if (!CurPropSet)
    return createStringError(EC, "invalid property set registry");

  return Expected<std::unique_ptr<PropertySetRegistry>>(std::move(Res));
}

namespace llvm {
// output a property to a stream
raw_ostream &operator<<(raw_ostream &Out, const PropertyValue &Prop) {
  Out << static_cast<int>(Prop.getType()) << "|";
  switch (Prop.getType()) {
  case PropertyValue::Type::UINT32:
    Out << Prop.asUint32();
    break;
  default:
    llvm_unreachable_internal("unsupported property type: " + Prop.getType());
  }
  return Out;
}
} // namespace llvm

void PropertySetRegistry::write(raw_ostream &Out) const {
  for (const auto &PropSet : PropSetMap) {
    Out << "[" << PropSet.first << "]\n";

    for (const auto &Props : PropSet.second) {
      Out << std::string(Props.first) << "=" << Props.second << "\n";
    }
  }
}

namespace llvm {
namespace util {

template <> uint32_t &PropertyValue::getValueRef<uint32_t>() {
  return Val.UInt32Val;
}
template <> PropertyValue::Type PropertyValue::getTypeTag<uint32_t>() {
  return UINT32;
}

constexpr char PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS[];
constexpr char PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK[];
} // namespace util
} // namespace llvm
