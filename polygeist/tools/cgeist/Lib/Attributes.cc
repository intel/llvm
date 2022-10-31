//===- Attributes.cc - Construct MLIR attributes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <functional>

#define DEBUG_TYPE "attributes"

using namespace llvm;
using namespace mlir;

namespace mlirclang {

static constexpr StringLiteral PassThroughAttrName = "passthrough";

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

static void addToPassThroughAttr(NamedAttribute &PassThroughAttr,
                                 mlir::NamedAttribute Attr, MLIRContext &Ctx) {
  assert(PassThroughAttr.getName() == PassThroughAttrName &&
         "PassThroughAttr is not valid");
  assert(PassThroughAttr.getValue().isa<ArrayAttr>() &&
         "PassThroughAttr should have an ArrayAttr as value");

  LLVM_DEBUG(llvm::dbgs() << "Adding attribute " << Attr.getName() << " to '"
                          << PassThroughAttrName << "'.\n";);

  std::vector<mlir::Attribute> Vec =
      PassThroughAttr.getValue().cast<ArrayAttr>().getValue().vec();

  // TODO: find a way to add the attributes only if one does not exist already,
  // and keep the list in sorted order.
  if (Attr.getValue().isa<UnitAttr>())
    Vec.push_back(Attr.getName());
  else
    Vec.push_back(ArrayAttr::get(&Ctx, {Attr.getName(), Attr.getValue()}));

  PassThroughAttr.setValue(ArrayAttr::get(&Ctx, Vec));

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << PassThroughAttrName << ": ( ";
    for (auto Item : Vec)
      llvm::dbgs() << Item << " ";
    llvm::dbgs() << ")\n";
  });
}

static void addToPassThroughAttr(mlir::NamedAttribute &PassThroughAttr,
                                 mlir::ArrayAttr NewAttrs, MLIRContext &Ctx) {
  assert(PassThroughAttr.getName() == PassThroughAttrName &&
         "PassThroughAttr is not valid");
  assert(PassThroughAttr.getValue().isa<ArrayAttr>() &&
         "PassThroughAttr should have an ArrayAttr as value");

  for (mlir::Attribute NewAttr : NewAttrs) {
    if (auto ArrAttr = NewAttr.dyn_cast<ArrayAttr>()) {
      assert(ArrAttr.size() == 2 && ArrAttr[0].isa<StringAttr>());
      NamedAttribute NamedAttr(ArrAttr[0].cast<StringAttr>(), ArrAttr[1]);
      addToPassThroughAttr(PassThroughAttr, NamedAttr, Ctx);
    } else if (auto StrAttr = NewAttr.dyn_cast<StringAttr>()) {
      NamedAttribute NamedAttr(StrAttr, UnitAttr::get(&Ctx));
      addToPassThroughAttr(PassThroughAttr, NamedAttr, Ctx);
    } else
      llvm_unreachable("Unexpected attribute kind");
  }
}

//===----------------------------------------------------------------------===//
// AttributeList Method Implementations
//===----------------------------------------------------------------------===//

AttributeList &
AttributeList::addAttrs(const AttrBuilder &FnAttrB, const AttrBuilder &RetAttrB,
                        llvm::ArrayRef<mlir::NamedAttrList> Attrs) {
  return addFnAttrs(FnAttrB).addRetAttrs(RetAttrB).addParamAttrs(Attrs);
}

AttributeList &AttributeList::addFnAttrs(const AttrBuilder &B) {
  for (const NamedAttribute &NewNamedAttr : B.getAttrs()) {
    Optional<NamedAttribute> ExistingAttr =
        FnAttrs.getNamed(NewNamedAttr.getName());
    if (!ExistingAttr) {
      FnAttrs.append(NewNamedAttr);
      continue;
    }

    // Merge the 'passthrough' attribute lists.
    if (ExistingAttr->getName() == PassThroughAttrName) {
      auto Attrs = NewNamedAttr.getValue().cast<ArrayAttr>();
      addToPassThroughAttr(*ExistingAttr, Attrs, B.getContext());
      FnAttrs.set(ExistingAttr->getName(), ExistingAttr->getValue());
      continue;
    }

    llvm_unreachable("Function attribute already exists");
  }

  return *this;
}

AttributeList &AttributeList::addRetAttrs(const AttrBuilder &B) {
  for (const NamedAttribute &NewNamedAttr : B.getAttrs()) {
    Optional<NamedAttribute> ExistingAttr =
        RetAttrs.getNamed(NewNamedAttr.getName());
    if (!ExistingAttr) {
      RetAttrs.append(NewNamedAttr);
      continue;
    }
    llvm_unreachable("Return value attribute already exits");
  }

  return *this;
}

AttributeList &
AttributeList::addParamAttrs(llvm::ArrayRef<mlir::NamedAttrList> Attrs) {
  ParamAttrs.reserve(Attrs.size());
  llvm::append_range(ParamAttrs, Attrs);
  return *this;
}

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind) {
  return addAttributeImpl(Kind, LLVM::LLVMDialect::getDialectNamespace(),
                          &AttrBuilder::addAttributeImpl);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
                                       mlir::Type Ty) {
  return addAttributeImpl(Kind, Ty, LLVM::LLVMDialect::getDialectNamespace(),
                          &AttrBuilder::addAttributeImpl);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
                                       uint64_t Val) {
  return addAttributeImpl(Kind, Val, &AttrBuilder::addRawIntAttr);
}

AttrBuilder &AttrBuilder::addAttribute(Twine AttrName, mlir::Attribute Attr) {
  return addAttributeImpl(AttrName, Attr, &AttrBuilder::addAttributeImpl);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind Kind) {
  return addAttributeImpl(Kind, llvm::None,
                          &AttrBuilder::addPassThroughAttributeImpl);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind Kind,
                                     mlir::Type Ty) {
  return addAttributeImpl(Kind, Ty, llvm::None,
                          &AttrBuilder::addPassThroughAttributeImpl);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind Kind,
                                     uint64_t Val) {
  return addAttributeImpl(Kind, Val, &AttrBuilder::addPassThroughRawIntAttr);
}

AttrBuilder &AttrBuilder::addPassThroughAttribute(StringRef AttrName,
                                                  mlir::Attribute Attr) {
  return addAttributeImpl(AttrName, Attr,
                          &AttrBuilder::addPassThroughAttributeImpl);
}

AttrBuilder &AttrBuilder::removeAttribute(llvm::StringRef AttrName) {
  bool ContainsPassThroughAttr = getAttr(PassThroughAttrName).has_value();
  if (ContainsPassThroughAttr) {
    NamedAttribute PassThroughAttr = getAttr(PassThroughAttrName).value();
    auto ArrAttr = PassThroughAttr.getValue().cast<ArrayAttr>();
    std::vector<mlir::Attribute> Vec = ArrAttr.getValue().vec();

    std::remove_if(Vec.begin(), Vec.end(), [AttrName](mlir::Attribute &Attr) {
      if (Attr.isa<StringAttr>())
        return (Attr.cast<StringAttr>().strref() == AttrName);
      if (Attr.isa<ArrayAttr>()) {
        auto ArrAttr = Attr.cast<ArrayAttr>();
        assert(ArrAttr.size() == 2 && ArrAttr[0].isa<StringAttr>());
        return (ArrAttr[0].cast<StringAttr>().strref() == AttrName);
      }
      return false;
    });

    PassThroughAttr.setValue(ArrayAttr::get(&Ctx, Vec));
    return *this;
  }

  Attrs.erase(AttrName);
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(llvm::Attribute::AttrKind Kind) {
  assert((unsigned)Kind < llvm::Attribute::EndAttrKinds &&
         "Attribute out of range!");
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return removeAttribute(AttrName);
}

bool AttrBuilder::contains(StringRef AttrName) const {
  if (containsInPassThrough(AttrName))
    return true;
  return getAttr(AttrName).has_value();
}

bool AttrBuilder::contains(llvm::Attribute::AttrKind Kind) const {
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return contains(AttrName);
}

Optional<NamedAttribute> AttrBuilder::getAttr(StringRef AttrName) const {
  return Attrs.getNamed(AttrName);
}

Optional<NamedAttribute>
AttrBuilder::getAttr(llvm::Attribute::AttrKind Kind) const {
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return getAttr(AttrName);
}

StringAttr AttrBuilder::createStringAttr(Twine AttrName,
                                         Optional<StringLiteral> Prefix,
                                         MLIRContext &Ctx) {
  return (Prefix) ? StringAttr::get(&Ctx, *Prefix + "." + AttrName)
                  : StringAttr::get(&Ctx, AttrName);
}

AttrBuilder &AttrBuilder::addAttributeImpl(llvm::Attribute::AttrKind Kind,
                                           Optional<StringLiteral> Dialect,
                                           AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");

  // TODO: Replace with std::invoke once C++17 headers are available.
  auto Invoke = [this](AddAttrFuncPtr AddAttrPtr,
                       auto... Args) -> AttrBuilder & {
    return (this->*AddAttrPtr)(Args...);
  };

  OpBuilder Builder(&Ctx);
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  NamedAttribute NamedAttr(createStringAttr(AttrName, Dialect, Ctx),
                           Builder.getUnitAttr());
  return Invoke(AddAttrPtr, NamedAttr);
}

AttrBuilder &AttrBuilder::addAttributeImpl(llvm::Attribute::AttrKind Kind,
                                           mlir::Type Ty,
                                           Optional<StringLiteral> Dialect,
                                           AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");

  // TODO: Replace with std::invoke once C++17 headers are available.
  auto Invoke = [this](AddAttrFuncPtr AddAttrPtr,
                       auto... Args) -> AttrBuilder & {
    return (this->*AddAttrPtr)(Args...);
  };

  OpBuilder Builder(&Ctx);
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  NamedAttribute NamedAttr(createStringAttr(AttrName, Dialect, Ctx),
                           mlir::TypeAttr::get(Ty));
  return Invoke(AddAttrPtr, NamedAttr);
}

AttrBuilder &
AttrBuilder::addAttributeImpl(llvm::Attribute::AttrKind Kind, uint64_t Val,
                              AddRawIntAttrFuncPtr AddRawIntAttrPtr) {
  assert(AddRawIntAttrPtr &&
         "'AddRawIntAttrPtr' should be a valid function pointer");

  // TODO: Replace with std::invoke once C++17 headers are available.
  auto Invoke = [this](AddRawIntAttrFuncPtr AddRawIntAttrPtr,
                       auto... Args) -> AttrBuilder & {
    return (this->*AddRawIntAttrPtr)(Args...);
  };

  switch (Kind) {
  case llvm::Attribute::AttrKind::Alignment:
    assert(Val <= llvm::Value::MaximumAlignment && "Alignment too large");
    LLVM_FALLTHROUGH;
  case llvm::Attribute::AttrKind::StackAlignment:
    assert(Val <= 0x100 && "Alignment too large.");
    LLVM_FALLTHROUGH;
  case llvm::Attribute::AttrKind::Dereferenceable:
  case llvm::Attribute::AttrKind::DereferenceableOrNull:
  case llvm::Attribute::AttrKind::UWTable:
    return (!Val) ? *this : Invoke(AddRawIntAttrPtr, Kind, Val);

  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAttributeImpl(Twine AttrName, mlir::Attribute Attr,
                                           AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");

  // TODO: Replace with std::invoke once C++17 headers are available.
  auto Invoke = [this](AddAttrFuncPtr AddAttrPtr,
                       auto... Args) -> AttrBuilder & {
    return (this->*AddAttrPtr)(Args...);
  };

  NamedAttribute NamedAttr(StringAttr::get(&Ctx, AttrName), Attr);
  return Invoke(AddAttrPtr, NamedAttr);
}

AttrBuilder &AttrBuilder::addAttributeImpl(mlir::NamedAttribute Attr) {
  Attrs.set(Attr.getName(), Attr.getValue());
  return *this;
}

AttrBuilder &
AttrBuilder::addPassThroughAttributeImpl(mlir::NamedAttribute Attr) {
  NamedAttribute PassThroughAttr = getOrCreatePassThroughAttr();
  addToPassThroughAttr(PassThroughAttr, Attr, Ctx);
  return addAttributeImpl(PassThroughAttr);
}

AttrBuilder &AttrBuilder::addRawIntAttr(llvm::Attribute::AttrKind Kind,
                                        uint64_t Value) {
  OpBuilder Builder(&Ctx);
  NamedAttribute NamedAttr(
      createStringAttr(llvm::Attribute::getNameFromAttrKind(Kind),
                       LLVM::LLVMDialect::getDialectNamespace(), Ctx),
      Builder.getIntegerAttr(Builder.getIntegerType(64), Value));
  return addAttributeImpl(NamedAttr);
}

AttrBuilder &
AttrBuilder::addPassThroughRawIntAttr(llvm::Attribute::AttrKind Kind,
                                      uint64_t Value) {
  OpBuilder Builder(&Ctx);
  NamedAttribute NamedAttr(
      StringAttr::get(&Ctx, llvm::Attribute::getNameFromAttrKind(Kind)),
      Builder.getIntegerAttr(Builder.getIntegerType(64), Value));
  return addPassThroughAttributeImpl(NamedAttr);
}

NamedAttribute AttrBuilder::getOrCreatePassThroughAttr() const {
  Optional<NamedAttribute> PassThroughAttr = getAttr(PassThroughAttrName);
  if (!PassThroughAttr) {
    LLVM_DEBUG(llvm::dbgs()
               << "Creating empty '" << PassThroughAttrName << "' attribute\n");
    PassThroughAttr = NamedAttribute(StringAttr::get(&Ctx, PassThroughAttrName),
                                     ArrayAttr::get(&Ctx, {}));
  }
  return *PassThroughAttr;
}

bool AttrBuilder::containsInPassThrough(StringRef AttrName) const {
  if (!getAttr(PassThroughAttrName).has_value())
    return false;

  NamedAttribute PassThroughAttr = getAttr(PassThroughAttrName).value();
  assert(PassThroughAttr.getValue().isa<ArrayAttr>() &&
         "passthrough attribute value should be an ArrayAttr");

  return llvm::any_of(
      PassThroughAttr.getValue().cast<ArrayAttr>(),
      [AttrName](mlir::Attribute Attr) {
        if (Attr.isa<ArrayAttr>()) {
          auto ArrAttr = Attr.cast<ArrayAttr>();
          assert(ArrAttr.size() == 2 && ArrAttr[0].isa<StringAttr>());
          return ArrAttr[0].cast<StringAttr>() == AttrName;
        }

        assert(Attr.isa<StringAttr>() && "Unexpected attribute Kind");
        return Attr.cast<StringAttr>() == AttrName;
      });
}

bool AttrBuilder::containsInPassThrough(llvm::Attribute::AttrKind Kind) const {
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return containsInPassThrough(AttrName);
}

} // end namespace mlirclang
