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
// AttributeList Method Implementations
//===----------------------------------------------------------------------===//

AttributeList &
AttributeList::addAttrs(const AttrBuilder &FnAttrB, const AttrBuilder &RetAttrB,
                        llvm::ArrayRef<mlir::NamedAttrList> Attrs) {
  return addFnAttrs(FnAttrB).addRetAttrs(RetAttrB).addParmAttrs(Attrs);
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
      AttrBuilder::addToPassThroughAttr(*ExistingAttr, Attrs, B.getContext());
      FnAttrs.set(ExistingAttr->getName(), ExistingAttr->getValue());
    }
  }

  return *this;
}

AttributeList &AttributeList::addRetAttrs(const AttrBuilder &B) {
  RetAttrs.append(B.getAttrs());
  return *this;
}

AttributeList &
AttributeList::addParmAttrs(llvm::ArrayRef<mlir::NamedAttrList> Attrs) {
  ParmAttrs.reserve(Attrs.size());
  llvm::append_range(ParmAttrs, Attrs);
  return *this;
}

mlir::NamedAttrList AttributeList::getParmAttrs(unsigned Index) const {
  assert(Index < ParmAttrs.size() && "Index out of range");
  return ParmAttrs[Index];
}

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
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
  NamedAttribute NamedAttr = createNamedAttr(
      createStringAttr(AttrName, Dialect, Ctx), Builder.getUnitAttr());
  return Invoke(AddAttrPtr, NamedAttr);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
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
  NamedAttribute NamedAttr = createNamedAttr(
      createStringAttr(AttrName, Dialect, Ctx), mlir::TypeAttr::get(Ty));

  return Invoke(AddAttrPtr, NamedAttr);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
                                       uint64_t Val,
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
    LLVM_FALLTHROUGH;
  case llvm::Attribute::AttrKind::StackAlignment:
    assert(Val <= llvm::Value::MaximumAlignment && "Alignment too large");
    return (!Val) ? *this : Invoke(AddRawIntAttrPtr, Kind, Val);

  case llvm::Attribute::AttrKind::Dereferenceable:
    LLVM_FALLTHROUGH;
  case llvm::Attribute::AttrKind::DereferenceableOrNull:
    LLVM_FALLTHROUGH;
  case llvm::Attribute::AttrKind::UWTable:
    return (!Val) ? *this : Invoke(AddRawIntAttrPtr, Kind, Val);

  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(Twine AttrName, mlir::Attribute Attr,
                                       AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");

  // TODO: Replace with std::invoke once C++17 headers are available.
  auto Invoke = [this](AddAttrFuncPtr AddAttrPtr,
                       auto... Args) -> AttrBuilder & {
    return (this->*AddAttrPtr)(Args...);
  };

  NamedAttribute NamedAttr =
      createNamedAttr(createStringAttr(AttrName, Ctx), Attr);
  return Invoke(AddAttrPtr, NamedAttr);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind Kind) {
  return addAttribute(Kind, llvm::None, &AttrBuilder::addPassThroughAttribute);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind Kind,
                                     mlir::Type Ty) {
  return addAttribute(Kind, Ty, llvm::None,
                      &AttrBuilder::addPassThroughAttribute);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind Kind,
                                     uint64_t Val) {
  return addAttribute(Kind, Val, &AttrBuilder::addPassThroughRawIntAttr);
}

AttrBuilder &AttrBuilder::addPassThroughAttribute(StringRef AttrName,
                                                  mlir::Attribute Attr) {
  return addAttribute(AttrName, Attr, &AttrBuilder::addPassThroughAttribute);
}

AttrBuilder &AttrBuilder::addPassThroughAttribute(mlir::NamedAttribute Attr) {
  NamedAttribute PassThroughAttr = getOrCreatePassThroughAttr();
  addToPassThroughAttr(PassThroughAttr, Attr, Ctx);
  return addAttribute(PassThroughAttr);
}

void AttrBuilder::addToPassThroughAttr(NamedAttribute &PassThroughAttr,
                                       mlir::NamedAttribute Attr,
                                       MLIRContext &Ctx) {
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

void AttrBuilder::addToPassThroughAttr(mlir::NamedAttribute &PassThroughAttr,
                                       mlir::ArrayAttr NewAttrs,
                                       MLIRContext &Ctx) {
  assert(PassThroughAttr.getName() == PassThroughAttrName &&
         "PassThroughAttr is not valid");
  assert(PassThroughAttr.getValue().isa<ArrayAttr>() &&
         "PassThroughAttr should have an ArrayAttr as value");

  for (mlir::Attribute NewAttr : NewAttrs) {
    if (NewAttr.isa<ArrayAttr>()) {
      auto ArrAttr = NewAttr.cast<ArrayAttr>();
      assert(ArrAttr.size() == 2 && ArrAttr[0].isa<StringAttr>());
      addToPassThroughAttr(
          PassThroughAttr,
          createNamedAttr(ArrAttr[0].cast<StringAttr>(), ArrAttr[1]), Ctx);
    } else if (NewAttr.isa<StringAttr>())
      addToPassThroughAttr(
          PassThroughAttr,
          createNamedAttr(NewAttr.cast<StringAttr>(), UnitAttr::get(&Ctx)),
          Ctx);
    else
      llvm_unreachable("Unexpected attribute kind");
  }
}

AttrBuilder &AttrBuilder::removeAttribute(llvm::StringRef AttrName) {
  if (containsInPassThrough(AttrName)) {
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

NamedAttribute AttrBuilder::createNamedAttr(StringAttr AttrName,
                                            mlir::Attribute Attr) {
  NamedAttribute NamedAttr(AttrName, Attr);
  return NamedAttr;
}

StringAttr AttrBuilder::createStringAttr(Twine AttrName, MLIRContext &Ctx) {
  return StringAttr::get(&Ctx, AttrName);
}

StringAttr AttrBuilder::createStringAttr(Twine AttrName,
                                         Optional<StringLiteral> Prefix,
                                         MLIRContext &Ctx) {
  return (Prefix) ? createStringAttr(*Prefix + "." + AttrName, Ctx)
                  : createStringAttr(AttrName, Ctx);
}

AttrBuilder &AttrBuilder::addRawIntAttr(llvm::Attribute::AttrKind Kind,
                                        uint64_t Value) {
  OpBuilder Builder(&Ctx);
  NamedAttribute NamedAttr = createNamedAttr(
      createStringAttr(llvm::Attribute::getNameFromAttrKind(Kind),
                       LLVM::LLVMDialect::getDialectNamespace(), Ctx),
      Builder.getIntegerAttr(Builder.getIntegerType(64), Value));
  return addAttribute(NamedAttr);
}

AttrBuilder &
AttrBuilder::addPassThroughRawIntAttr(llvm::Attribute::AttrKind Kind,
                                      uint64_t Value) {
  OpBuilder Builder(&Ctx);
  NamedAttribute NamedAttr = createNamedAttr(
      createStringAttr(llvm::Attribute::getNameFromAttrKind(Kind), Ctx),
      Builder.getIntegerAttr(Builder.getIntegerType(64), Value));
  return addPassThroughAttribute(NamedAttr);
}

NamedAttribute AttrBuilder::getOrCreatePassThroughAttr() const {
  Optional<NamedAttribute> PassThroughAttr = getAttr(PassThroughAttrName);
  if (!PassThroughAttr) {
    LLVM_DEBUG(llvm::dbgs()
               << "Creating empty '" << PassThroughAttrName << "' attribute\n");
    PassThroughAttr = NamedAttribute(createStringAttr(PassThroughAttrName, Ctx),
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
