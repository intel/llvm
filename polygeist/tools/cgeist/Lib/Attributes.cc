//===- Attributes.cc - Construct LLVMIR attributes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attributes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "attributes"

using namespace llvm;
using namespace mlir;

namespace mlirclang {

static constexpr StringLiteral passThroughAttrName = "passthrough";

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind kind) {
  OpBuilder builder(&ctx);
  constexpr StringLiteral dialect = LLVM::LLVMDialect::getDialectNamespace();
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  NamedAttribute namedAttr = createNamedAttr(
      createStringAttr(attrName, dialect), builder.getUnitAttr());
  return addAttribute(namedAttr);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind kind,
                                       mlir::Type Ty) {
  OpBuilder builder(&ctx);
  constexpr StringLiteral dialect = LLVM::LLVMDialect::getDialectNamespace();
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  return addAttribute(dialect + "." + attrName, mlir::TypeAttr::get(Ty));
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind kind,
                                       uint64_t val) {
  OpBuilder builder(&ctx);
  constexpr StringLiteral dialect = LLVM::LLVMDialect::getDialectNamespace();
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  using AttrKind = llvm::Attribute::AttrKind;

  switch (kind) {
  case AttrKind::Alignment:
    assert(val <= llvm::Value::MaximumAlignment && "Alignment too large");
    LLVM_FALLTHROUGH;
  case AttrKind::Dereferenceable:
    LLVM_FALLTHROUGH;
  case AttrKind::DereferenceableOrNull: {
    if (val > 0) {
      NamedAttribute namedAttr = createNamedAttr(
          createStringAttr(attrName, dialect),
          builder.getIntegerAttr(builder.getIntegerType(64), val));
      addAttribute(namedAttr);
    }
  } break;
  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(Twine attrName, mlir::Attribute attr) {
  NamedAttribute namedAttr = createNamedAttr(createStringAttr(attrName), attr);
  return addAttribute(namedAttr);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind kind) {
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  return addPassThroughAttribute(createStringAttr(attrName));
}

AttrBuilder &AttrBuilder::addPassThroughAttribute(mlir::Attribute attr) {
  NamedAttribute passThrough = getOrCreatePassThroughAttr();
  assert(passThrough.getValue().isa<ArrayAttr>() &&
         "passthrough attribute should have an ArrayAttr as value");

  LLVM_DEBUG(llvm::dbgs() << "Adding attribute " << attr << " to '"
                          << passThroughAttrName << "'.\n");
  std::vector<mlir::Attribute> vec =
      passThrough.getValue().cast<ArrayAttr>().getValue().vec();
  vec.push_back(attr);
  passThrough.setValue(ArrayAttr::get(&ctx, vec));

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << passThroughAttrName << ": ( ";
    for (auto item : vec)
      llvm::dbgs() << item << " ";
    llvm::dbgs() << ")\n";
  });

  return addAttribute(passThrough);
}

bool AttrBuilder::contains(StringRef attrName) const {
  return getAttr(attrName).has_value();
}

bool AttrBuilder::contains(llvm::Attribute::AttrKind kind) const {
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  return contains(attrName);
}

bool AttrBuilder::containsInPassThrough(StringRef attrName) const {
  if (!contains(passThroughAttrName))
    return false;

  NamedAttribute passThrough = getAttr(passThroughAttrName).value();
  assert(passThrough.getValue().isa<ArrayAttr>() &&
         "passthrough attribute value should be an ArrayAttr");

  return llvm::any_of(passThrough.getValue().cast<ArrayAttr>(),
                      [attrName](mlir::Attribute attr) {
                        assert(attr.isa<StringAttr>() &&
                               "Unexpected attribute kind");
                        return attr.cast<StringAttr>() == attrName;
                      });
}

bool AttrBuilder::containsInPassThrough(llvm::Attribute::AttrKind kind) const {
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  return containsInPassThrough(attrName);
}

Optional<NamedAttribute> AttrBuilder::getAttr(StringRef attrName) const {
  return attrs.getNamed(attrName);
}

Optional<NamedAttribute>
AttrBuilder::getAttr(llvm::Attribute::AttrKind kind) const {
  StringRef attrName = llvm::Attribute::getNameFromAttrKind(kind);
  return getAttr(attrName);
}

NamedAttribute AttrBuilder::getOrCreatePassThroughAttr() const {
  Optional<NamedAttribute> passThrough = getAttr(passThroughAttrName);
  if (!passThrough) {
    LLVM_DEBUG(llvm::dbgs()
               << "Creating empty '" << passThroughAttrName << "' attribute\n");
    passThrough = NamedAttribute(createStringAttr(passThroughAttrName),
                                 ArrayAttr::get(&ctx, {}));
  }
  return *passThrough;
}

NamedAttribute AttrBuilder::createNamedAttr(StringAttr attrName,
                                            mlir::Attribute attr) const {
  NamedAttribute namedAttr(attrName, attr);
  return namedAttr;
}

StringAttr AttrBuilder::createStringAttr(Twine attrName) const {
  return StringAttr::get(&ctx, attrName);
}

StringAttr AttrBuilder::createStringAttr(Twine attrName,
                                         StringLiteral prefix) const {
  return createStringAttr(prefix + "." + attrName);
}

} // end namespace mlirclang
