//===- Attributes.h - Construct LLVMIR attributes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_ATTRIBUTES_H
#define CGEIST_ATTRIBUTES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/IR/Attributes.h"

namespace mlirclang {

class AttrBuilder;

/// \class
/// This class holds the attributes for a function, its return value, and
/// its parameters. You access the attributes for each of them via an index into
/// the AttributeList object. The function attributes are at index
/// `AttributeList::FunctionIndex', the return value is at index
/// `AttributeList::ReturnIndex', and the attributes for the parameters start at
/// index `AttributeList::FirstArgIndex'.
class AttributeList {
public:
  AttributeList() = default;

  //===--------------------------------------------------------------------===//
  // AttributeList Mutation
  //===--------------------------------------------------------------------===//

  /// Add function. return value, and parameters attributes to the list.
  AttributeList &addAttrs(const AttrBuilder &FnAttrB,
                          const AttrBuilder &RetAttrB,
                          llvm::ArrayRef<mlir::NamedAttrList> Attrs);

  /// Add function attributes to the list.
  AttributeList &addFnAttrs(const AttrBuilder &B);

  /// Add return value attributes to the list.
  AttributeList &addRetAttrs(const AttrBuilder &B);

  /// Add parameters attributes to the list.
  AttributeList &addParmAttrs(llvm::ArrayRef<mlir::NamedAttrList> Attrs);

  /// The function attributes are returned.
  mlir::NamedAttrList getFnAttrs() const { return FnAttrs; }

  /// The attributes for the ret value are returned.
  mlir::NamedAttrList getRetAttrs() const { return RetAttrs; }

  /// The attributes for the parameters are returned.
  mlir::ArrayRef<mlir::NamedAttrList> getParmAttrs() const { return ParmAttrs; }

  /// The attributes for the parameter at the given index are returned.
  mlir::NamedAttrList getParmAttrs(unsigned Index) const;

private:
  /// The attributes that we are managing.
  mlir::NamedAttrList FnAttrs;
  mlir::NamedAttrList RetAttrs;
  llvm::SmallVector<mlir::NamedAttrList, 8> ParmAttrs;
};

/// \class
/// Facilitates the construction of LLVM dialect attributes for a particular
/// argument, parameter, function, or return value.
class AttrBuilder {
public:
  AttrBuilder(mlir::MLIRContext &Ctx) : Ctx(Ctx) {}
  AttrBuilder(const AttrBuilder &) = delete;
  AttrBuilder(AttrBuilder &&) = default;

  using AddAttrFuncPtr =
      AttrBuilder &(AttrBuilder::*)(mlir::NamedAttribute Attr);
  using AddRawIntAttrFuncPtr = AttrBuilder &(
      AttrBuilder::*)(llvm::Attribute::AttrKind Kind, uint64_t Value);

  /// Add the LLVM attribute identified by \p Kind to the builder, optionally
  /// prefixing the attribute name with \p Dialect.
  /// Note: \p AddAttrPtr is used to provide a concrete implementation of where
  /// to add the attribute (to the builder 'passthrough' named attribute or
  /// not).
  AttrBuilder &
  addAttribute(llvm::Attribute::AttrKind Kind,
               llvm::Optional<llvm::StringLiteral> Dialect =
                   mlir::LLVM::LLVMDialect::getDialectNamespace(),
               AddAttrFuncPtr AddAttrPtr = &AttrBuilder::addAttribute);

  /// Add the LLVM attribute identified by \p Kind with a value given by \p Val
  /// to the builder.
  /// Note: \p AddRawIntAttrPtr is used to provide a concrete implementation of
  /// where to add the attribute (to the builder 'passthrough' named attribute
  /// or not).
  AttrBuilder &addAttribute(
      llvm::Attribute::AttrKind Kind, uint64_t Val,
      AddRawIntAttrFuncPtr AddRawIntAttrPtr = &AttrBuilder::addRawIntAttr);

  /// Create a NamedAttribute with name \p AttrName and value \p Attr and
  /// add it to the builder.
  /// Note: \p AddAttrPtr is used to provide a concrete implementation of where
  /// to add the attribute (to the builder 'passthrough' named attribute or
  /// not).
  AttrBuilder &
  addAttribute(llvm::StringRef AttrName, mlir::Attribute Attr,
               AddAttrFuncPtr AddAttrPtr = &AttrBuilder::addAttribute);

  /// Add the given named attribute \p Attr to the builder.
  AttrBuilder &addAttribute(mlir::NamedAttribute Attr) {
    Attrs.set(Attr.getName(), Attr.getValue());
    return *this;
  }

  /// Add the LLVM attribute identified by \p Kind to the builder "passthrough"
  /// named attribute.
  AttrBuilder &addPassThroughAttribute(llvm::Attribute::AttrKind Kind);

  /// Add the LLVM attribute identified by \p Kind with a value given by \p Val
  /// to the builder "passthrough" named attribute.
  AttrBuilder &addPassThroughAttribute(llvm::Attribute::AttrKind Kind,
                                       uint64_t Val);

  /// Create a NamedAttribute with name \p AttrName and value \p Attr and add it
  /// to the builder "passthrough" named attribute.
  AttrBuilder &addPassThroughAttribute(llvm::StringRef AttrName,
                                       mlir::Attribute Attr);

  /// Add the given named attribute \p Attr to the builder "passthrough" named
  /// attribute.
  AttrBuilder &addPassThroughAttribute(mlir::NamedAttribute Attr);

  /// Add \p Attr to the \p PassThroughAttr list.
  /// Note: \p PassThroughAttr must have name "passthrough" and value of type
  /// ArrayAttr.
  static void addToPassThroughAttr(mlir::NamedAttribute &PassThroughAttr,
                                   mlir::NamedAttribute Attr,
                                   mlir::MLIRContext &Ctx);

  /// Add \p NewAttrs into the attribute list of \p PassThroughAttr.
  /// Note: \p PassThroughAttr must have name "passthrough" and value of type
  /// ArrayAttr.
  static void addToPassThroughAttr(mlir::NamedAttribute &PassThroughAttr,
                                   mlir::ArrayAttr NewAttrs,
                                   mlir::MLIRContext &Ctx);

  /// Remove an attribute from the builder (if present).
  /// Note: the given attribute will be removed even if it is contained by the
  /// 'passthrough' named attribute.
  AttrBuilder &removeAttribute(llvm::StringRef AttrName);
  AttrBuilder &removeAttribute(llvm::Attribute::AttrKind Kind);

  /// Return true if the builder contains the specified attribute.
  /// Note: these member functions also lookup for the given attribute in the
  /// 'passthrough' named attribute if it exists.
  bool contains(llvm::StringRef AttrName) const;
  bool contains(llvm::Attribute::AttrKind Kind) const;

  /// Return true if the builder contains any attribute and false otherwise.
  bool hasAttributes() const { return !Attrs.empty(); }

  /// Return the given attribute if the builder contains it and llvm::None
  /// otherwise.
  llvm::Optional<mlir::NamedAttribute> getAttr(llvm::StringRef AttrName) const;
  llvm::Optional<mlir::NamedAttribute>
  getAttr(llvm::Attribute::AttrKind Kind) const;

  /// Returns the attributes contained in the builder.
  llvm::ArrayRef<mlir::NamedAttribute> getAttrs() const { return Attrs; }

  mlir::MLIRContext &getContext() const { return Ctx; }

  /// returns a NamedAttribute with name \p AttrName , and value \p Attr.
  static mlir::NamedAttribute createNamedAttr(mlir::StringAttr AttrName,
                                              mlir::Attribute Attr);

  /// Returns a StringAttr of the form 'AttrName'.
  static mlir::StringAttr createStringAttr(llvm::Twine AttrName,
                                           mlir::MLIRContext &Ctx);

  /// Returns a StringAttr of the form 'prefix.AttrName'.
  static mlir::StringAttr
  createStringAttr(llvm::Twine AttrName,
                   llvm::Optional<llvm::StringLiteral> Prefix,
                   mlir::MLIRContext &Ctx);

private:
  /// Add integer attribute with raw value (packed/encoded if necessary).
  AttrBuilder &addRawIntAttr(llvm::Attribute::AttrKind Kind, uint64_t Value);

  /// Add integer attribute with raw value (packed/encoded if necessary) to the
  /// builder "passthrough" named attribute.
  AttrBuilder &addPassThroughRawIntAttr(llvm::Attribute::AttrKind Kind,
                                        uint64_t Value);

  /// Retrieve the "passthrough" named attribute if present, create it with an
  /// empty list otherwise.
  mlir::NamedAttribute getOrCreatePassThroughAttr() const;

  /// Return true if the builder contains the specified attribute within the
  /// 'passthrough' attribute.
  bool containsInPassThrough(llvm::StringRef AttrName) const;
  bool containsInPassThrough(llvm::Attribute::AttrKind Kind) const;

  mlir::MLIRContext &Ctx;
  mlir::NamedAttrList Attrs;
};

} // end namespace mlirclang

#endif // CGEIST_ATTRIBUTES_H
