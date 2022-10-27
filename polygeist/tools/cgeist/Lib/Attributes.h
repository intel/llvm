//===- Attributes.h - Construct LLVMIR attributes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_ATTRIBUTES_H
#define CGEIST_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/IR/Attributes.h"

namespace mlirclang {

/// \class
/// Facilitates the construction of LLVM dialect attributes for a particular
/// argument, parameter, function, or return value.
class AttrBuilder {
public:
  AttrBuilder(mlir::MLIRContext &ctx) : ctx(ctx) {}
  AttrBuilder(const AttrBuilder &) = delete;
  AttrBuilder(AttrBuilder &&) = default;

  /// Add the LLVM attribute identified by \p kind to the builder.
  /// Note: only unit attributes can be added using this member function.
  AttrBuilder &addAttribute(llvm::Attribute::AttrKind kind);

  /// Add the LLVM attribute identified by \p kind with a type given by \p Ty
  /// to the builder.
  AttrBuilder &addAttribute(llvm::Attribute::AttrKind kind, mlir::Type Ty);

  /// Add the LLVM attribute identified by \p kind with a value given by \p val
  /// to the builder.
  AttrBuilder &addAttribute(llvm::Attribute::AttrKind kind, uint64_t val);

  /// Create a NamedAttribute with name \p attrName and value \p attr and add it
  /// to the builder.
  AttrBuilder &addAttribute(llvm::Twine attrName, mlir::Attribute attr);

  /// Add the given named attribute \p attr to the builder.
  AttrBuilder &addAttribute(mlir::NamedAttribute attr) {
    attrs.set(attr.getName(), attr.getValue());
    return *this;
  }

  /// Add the LLVM attribute identified by \p kind to the builder "passthrough"
  /// named attribute.
  AttrBuilder &addPassThroughAttribute(llvm::Attribute::AttrKind kind);

  /// Add the given attribute \p attr to the builder "passthrough" named
  /// attribute.
  AttrBuilder &addPassThroughAttribute(mlir::Attribute attr);

  /// Return true if the builder contains the specified attribute.
  bool contains(llvm::StringRef attrName) const;
  bool contains(llvm::Attribute::AttrKind kind) const;

  /// Return true if the builder contains the specified attribute within the
  /// 'passthrough' attribute.
  bool containsInPassThrough(llvm::StringRef attrName) const;
  bool containsInPassThrough(llvm::Attribute::AttrKind kind) const;

  /// Return true if the builder contains any attribute and false otherwise.
  bool hasAttributes() const { return !attrs.empty(); }

  /// Return the given attribute if the builder contains it and llvm::None
  /// otherwise.
  llvm::Optional<mlir::NamedAttribute> getAttr(llvm::StringRef attrName) const;
  llvm::Optional<mlir::NamedAttribute>
  getAttr(llvm::Attribute::AttrKind kind) const;

  /// Returns the attributes contained in the builder.
  llvm::ArrayRef<mlir::NamedAttribute> getAttrs() const { return attrs; }

private:
  /// Retrieve the "passthrough" named attribute if present, create it with an
  /// empty list otherwise.
  mlir::NamedAttribute getOrCreatePassThroughAttr() const;

  /// returns a NamedAttribute with name \p attrName , and value \p attr.
  mlir::NamedAttribute createNamedAttr(mlir::StringAttr attrName,
                                       mlir::Attribute attr) const;

  /// Returns a StringAttr of the form 'attrName'.
  mlir::StringAttr createStringAttr(llvm::Twine attrName) const;

  /// Returns a StringAttr of the form 'prefix.attrName'.
  mlir::StringAttr createStringAttr(llvm::Twine attrName,
                                    llvm::StringLiteral prefix) const;

  mlir::MLIRContext &ctx;
  mlir::NamedAttrList attrs;
};

} // end namespace mlirclang

#endif // CGEIST_ATTRIBUTES_H
