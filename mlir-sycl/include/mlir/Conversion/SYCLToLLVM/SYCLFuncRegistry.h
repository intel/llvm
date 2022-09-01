//===- SYCLFuncRegistry.h - Registry of SYCL Functions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declare a registry of SYCL functions callable from the compiler.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOLLVM_SYCLFUNCREGISTRY_H
#define MLIR_CONVERSION_SYCLTOLLVM_SYCLFUNCREGISTRY_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace sycl {
class SYCLFuncRegistry;

/// \class SYCLFuncDescriptor
/// Represents a SYCL function (defined in a registry) that can be called by the
/// compiler.
/// Note: when a new enumerator is added, the corresponding SYCLFuncDescriptor
/// needs to be created in SYCLFuncRegistry constructor.
class SYCLFuncDescriptor {
  friend class SYCLFuncRegistry;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const SYCLFuncDescriptor &);

public:
  /// Enumerates SYCL functions.
  // clang-format off
  enum class FuncId {
    Unknown, 

    // Member functions for the sycl:id<n> class.
    Id1CtorDefault, // sycl::id<1>::id()
    Id2CtorDefault, // sycl::id<2>::id()
    Id3CtorDefault, // sycl::id<3>::id()
    Id1CtorSizeT,   // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type)
    Id2CtorSizeT,   // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type)
    Id3CtorSizeT,   // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type)
    Id1Ctor2SizeT,  // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
    Id2Ctor2SizeT,  // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
    Id3Ctor2SizeT,  // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
    Id1Ctor3SizeT,  // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
    Id2Ctor3SizeT,  // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
    Id3Ctor3SizeT,  // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
    Id1CopyCtor,    // sycl::id<1>::id(sycl::id<1> const&)
    Id2CopyCtor,    // sycl::id<2>::id(sycl::id<2> const&)
    Id3CopyCtor,    // sycl::id<3>::id(sycl::id<3> const&)
    
    // Member functions for the sycl::Range<n> class.
    Range1CtorDefault, // sycl::Range<1>::range()
    Range2CtorDefault, // sycl::range<2>::range()
    Range3CtorDefault, // sycl::range<3>::range()
    Range1CtorSizeT,   // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type)
    Range2CtorSizeT,   // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type)
    Range3CtorSizeT,   // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type)
    Range1Ctor2SizeT,  // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
    Range2Ctor2SizeT,  // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
    Range3Ctor2SizeT,  // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
    Range1Ctor3SizeT,  // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
    Range2Ctor3SizeT,  // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
    Range3Ctor3SizeT,  // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
    Range1CopyCtor,    // sycl::range<1>::range(sycl::range<1> const&)
    Range2CopyCtor,    // sycl::range<2>::range(sycl::range<2> const&)
    Range3CopyCtor,    // sycl::range<3>::range(sycl::range<3> const&)
  };
  // clang-format on

  /// Enumerates the kind of FuncId.
  enum class FuncIdKind {
    Unknown,
    IdCtor,   // any sycl::id<n> constructors.
    RangeCtor // any sycl::range<n> constructors.
  };

  /// Returns the funcIdKind given a \p funcId.
  static FuncIdKind getFuncIdKind(FuncId funcId);

  /// Retuns a descriptive name for the given \p funcIdKind.
  static std::string funcIdKindToName(FuncIdKind funcIdKind);

  /// Retuns the FuncIdKind given a descriptive \p name.
  static FuncIdKind nameToFuncIdKind(Twine name);

  // Call the SYCL constructor identified by \p id with the given \p args.
  static Value call(FuncId id, ValueRange args,
                    const SYCLFuncRegistry &registry, OpBuilder &b,
                    Location loc);

private:
  /// Private constructor: only available to 'SYCLFuncRegistry'.
  SYCLFuncDescriptor(FuncId id, StringRef name, Type outputTy,
                     ArrayRef<Type> argTys)
      : funcId(id), funcIdKind(getFuncIdKind(id)), name(name),
        outputTy(outputTy), argTys(argTys.begin(), argTys.end()) {
    assert(funcId != FuncId::Unknown && "Illegal function id");
    assert(funcIdKind != FuncIdKind::Unknown && "Illegal function id kind");
  }

  /// Inject the declaration for this function into the module.
  void declareFunction(ModuleOp &module, OpBuilder &b);

  /// Returns true if the given \p funcId is for a sycl::id<n> constructor.
  static bool isIdCtor(FuncId funcId);

private:
  FuncId funcId = FuncId::Unknown;             // SYCL function identifier
  FuncIdKind funcIdKind = FuncIdKind::Unknown; // SYCL function kind
  StringRef name;              // SYCL function name
  Type outputTy;               // SYCL function output type
  SmallVector<Type, 4> argTys; // SYCL function arguments types
  FlatSymbolRefAttr funcRef;   // Reference to the SYCL function 
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SYCLFuncDescriptor &desc) {
  os << "funcId=" << (int)desc.funcId
     << ", funcIdKind=" << SYCLFuncDescriptor::funcIdKindToName(desc.funcIdKind)
     << ", name='" << desc.name.str() << "')";
  return os;
}

/// \class SYCLFuncRegistry
/// Singleton class representing the set of SYCL functions callable from the
/// compiler.
class SYCLFuncRegistry {
public:
  ~SYCLFuncRegistry() { instance = nullptr; }

  static const SYCLFuncRegistry create(ModuleOp &module, OpBuilder &builder);

  const SYCLFuncDescriptor &getFuncDesc(SYCLFuncDescriptor::FuncId id) const {
    assert((registry.find(id) != registry.end()) &&
           "function identified by 'id' not found in the SYCL function "
           "registry");
    return registry.at(id);
  }

  // Returns the SYCLFuncDescriptor::FuncId corresponding to the function
  // descriptor that matches the given signature and funcIdKind.
  SYCLFuncDescriptor::FuncId
  getFuncId(SYCLFuncDescriptor::FuncIdKind funcIdKind, Type retType,
            TypeRange argTypes) const;

private:
  SYCLFuncRegistry(ModuleOp &module, OpBuilder &builder);

  using Registry = std::map<SYCLFuncDescriptor::FuncId, SYCLFuncDescriptor>;
  static SYCLFuncRegistry *instance;
  Registry registry;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLFUNCREGISTRY_H
