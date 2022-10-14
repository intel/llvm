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

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind kind) {
  OpBuilder builder(&ctx);
  UnitAttr unitAttr = builder.getUnitAttr();
  constexpr StringLiteral dialect = LLVM::LLVMDialect::getDialectNamespace();

  switch (kind) {
  /// Pass structure by value.
  case llvm::Attribute::AttrKind::ByVal: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("byval", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function can only be moved to control-equivalent blocks.
  case llvm::Attribute::AttrKind::Convergent: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("convergent", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function is required to make forward progress.
  case llvm::Attribute::AttrKind::MustProgress: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("mustprogress", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Nested function static chain.
  case llvm::Attribute::AttrKind::Nest: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("nest", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Considered to not alias after call.
  case llvm::Attribute::AttrKind::NoAlias: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("noalias", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function creates no aliases of pointer.
  case llvm::Attribute::AttrKind::NoCapture: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("nocapture", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function does not deallocate memory.
  case llvm::Attribute::AttrKind::NoFree: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("nofree", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function is never inlined.
  case llvm::Attribute::AttrKind::NoInline: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("noinline", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Pointer is known to be not null.
  case llvm::Attribute::AttrKind::NonNull: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("nonnull", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function does not recurse.
  case llvm::Attribute::AttrKind::NoRecurse: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("norecurse", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function does not return.
  case llvm::Attribute::AttrKind::NoReturn: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("noreturn", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Parameter or return value may not contain uninitialized or poison bits.
  case llvm::Attribute::AttrKind::NoUndef: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("noundef", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function doesn't unwind stack.
  case llvm::Attribute::AttrKind::NoUnwind: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("nounwind", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function does not access memory.
  case llvm::Attribute::AttrKind::ReadNone: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("readnone", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function only reads from memory.
  case llvm::Attribute::AttrKind::ReadOnly: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("readonly", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Hidden pointer to structure to return.
  case llvm::Attribute::AttrKind::StructRet: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("sret", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function always comes back to callsite.
  case llvm::Attribute::AttrKind::WillReturn: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("willreturn", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  /// Function only writes to memory.
  case llvm::Attribute::AttrKind::WriteOnly: {
    NamedAttribute namedAttr =
        createNamedAttr(createStringAttr("writeonly", dialect), unitAttr);
    addAttribute(namedAttr);
  } break;
  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind kind,
                                       int64_t val) {
  OpBuilder builder(&ctx);
  constexpr StringLiteral dialect = LLVM::LLVMDialect::getDialectNamespace();

  switch (kind) {
  /// Alignment of parameter.
  case llvm::Attribute::AttrKind::Alignment: {
    assert(val > 0 && "Invalid alignment value");
    assert(val <= 0x100 && "Alignment too large");
    NamedAttribute namedAttr = createNamedAttr(
        createStringAttr("align", dialect),
        builder.getIntegerAttr(builder.getIntegerType(32), val));
    addAttribute(namedAttr);
  } break;
  /// Pointer is known to be dereferenceable.
  case llvm::Attribute::AttrKind::Dereferenceable: {
    assert(val > 0 && "Invalid number of bytes");
    NamedAttribute namedAttr = createNamedAttr(
        createStringAttr("dereferenceable", dialect),
        builder.getIntegerAttr(builder.getIntegerType(64), val));
    addAttribute(namedAttr);
  } break;
  /// Pointer is either null or dereferenceable.
  case llvm::Attribute::AttrKind::DereferenceableOrNull: {
    assert(val > 0 && "Invalid number of bytes");
    NamedAttribute namedAttr = createNamedAttr(
        createStringAttr("dereferenceable_or_null", dialect),
        builder.getIntegerAttr(builder.getIntegerType(64), val));
    addAttribute(namedAttr);
  } break;
  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(StringRef attrName,
                                       mlir::Attribute attr) {
  NamedAttribute namedAttr = createNamedAttr(createStringAttr(attrName), attr);
  return addAttribute(namedAttr);
}

AttrBuilder &
AttrBuilder::addPassThroughAttribute(llvm::Attribute::AttrKind kind) {
  switch (kind) {
  /// Pass structure by value.
  case llvm::Attribute::AttrKind::ByVal:
    addPassThroughAttribute(createStringAttr("byval"));
    break;
  /// Function can only be moved to control-equivalent blocks.
  case llvm::Attribute::AttrKind::Convergent:
    addPassThroughAttribute(createStringAttr("convergent"));
    break;
  /// Function is required to make forward progress.
  case llvm::Attribute::AttrKind::MustProgress:
    addPassThroughAttribute(createStringAttr("mustprogress"));
    break;
  /// Nested function static chain.
  case llvm::Attribute::AttrKind::Nest:
    addPassThroughAttribute(createStringAttr("nest"));
    break;
  /// Considered to not alias after call.
  case llvm::Attribute::AttrKind::NoAlias:
    addPassThroughAttribute(createStringAttr("noalias"));
    break;
  /// Function creates no aliases of pointer.
  case llvm::Attribute::AttrKind::NoCapture:
    addPassThroughAttribute(createStringAttr("nocapture"));
    break;
  /// Function does not deallocate memory.
  case llvm::Attribute::AttrKind::NoFree:
    addPassThroughAttribute(createStringAttr("nofree"));
    break;
  /// Function is never inlined.
  case llvm::Attribute::AttrKind::NoInline:
    addPassThroughAttribute(createStringAttr("noinline"));
    break;
  /// Pointer is known to be not null.
  case llvm::Attribute::AttrKind::NonNull:
    addPassThroughAttribute(createStringAttr("nonnull"));
    break;
  /// Function does not recurse.
  case llvm::Attribute::AttrKind::NoRecurse:
    addPassThroughAttribute(createStringAttr("norecurse"));
    break;
  /// Function does not return.
  case llvm::Attribute::AttrKind::NoReturn:
    addPassThroughAttribute(createStringAttr("noreturn"));
    break;
  /// Parameter or return value may not contain uninitialized or poison bits.
  case llvm::Attribute::AttrKind::NoUndef:
    addPassThroughAttribute(createStringAttr("noundef"));
    break;
  /// Function doesn't unwind stack.
  case llvm::Attribute::AttrKind::NoUnwind:
    addPassThroughAttribute(createStringAttr("nounwind"));
    break;
  /// Function does not access memory.
  case llvm::Attribute::AttrKind::ReadNone:
    addPassThroughAttribute(createStringAttr("readnone"));
    break;
  /// Function only reads from memory.
  case llvm::Attribute::AttrKind::ReadOnly:
    addPassThroughAttribute(createStringAttr("readonly"));
    break;
  /// Hidden pointer to structure to return.
  case llvm::Attribute::AttrKind::StructRet:
    addPassThroughAttribute(createStringAttr("sret"));
    break;
  /// Function always comes back to callsite.
  case llvm::Attribute::AttrKind::WillReturn:
    addPassThroughAttribute(createStringAttr("willreturn"));
    break;
  /// Function only writes to memory.
  case llvm::Attribute::AttrKind::WriteOnly:
    addPassThroughAttribute(createStringAttr("writeonly"));
    break;
  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addPassThroughAttribute(mlir::Attribute attr) {
  NamedAttribute passThrough = getOrCreatePassThroughAttr();
  assert(passThrough.getValue().isa<ArrayAttr>() &&
         "PassThrough attribute should have an ArrayAttr as value");

  LLVM_DEBUG(llvm::dbgs() << "Adding attribute " << attr
                          << " to 'passthrough'.\n");
  std::vector<mlir::Attribute> vec =
      passThrough.getValue().cast<ArrayAttr>().getValue().vec();
  vec.push_back(attr);
  passThrough.setValue(ArrayAttr::get(&ctx, vec));

  LLVM_DEBUG({
    llvm::dbgs() << "  passthrough: ( ";
    for (auto item : vec)
      llvm::dbgs() << item << " ";
    llvm::dbgs() << ")\n";
  });

  return addAttribute(passThrough);
}

bool AttrBuilder::contains(StringRef attrName) const {
  return getAttr(attrName).hasValue();
}

Optional<NamedAttribute> AttrBuilder::getAttr(StringRef attrName) const {
  return attrs.getNamed(attrName);
}

NamedAttribute AttrBuilder::getOrCreatePassThroughAttr() const {
  const char *name = "passthrough";
  Optional<NamedAttribute> passThrough = attrs.getNamed(name);
  if (!passThrough) {
    LLVM_DEBUG(llvm::dbgs() << "Creating empty 'passthrough' attribute\n");
    passThrough =
        NamedAttribute(createStringAttr(name), ArrayAttr::get(&ctx, {}));
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
