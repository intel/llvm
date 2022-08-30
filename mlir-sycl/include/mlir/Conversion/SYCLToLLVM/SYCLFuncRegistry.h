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

public:
  /// Enumerates SYCL functions.
  // clang-format off
  enum class FuncId {
    // Member functions for the sycl:id<n> class.
    Id1CtorDefault, // sycl::id<1>::id()
    Id2CtorDefault, // sycl::id<2>::id()
    Id3CtorDefault, // sycl::id<3>::id()
    Id1CtorSizeT,   // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type)
    Id2CtorSizeT,   // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type)
    Id3CtorSizeT,   // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type)
    Id1CtorRange,   // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
    Id2CtorRange,   // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
    Id3CtorRange,   // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
    Id1CtorItem,    // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
    Id2CtorItem,    // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
    Id3CtorItem,    // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)

    // Member functions for ..TODO..
  };
  // clang-format on

  // Call the SYCL constructor identified by \p id with the given \p args.
  static Value call(FuncId id, ValueRange args,
                    const SYCLFuncRegistry &registry, OpBuilder &b,
                    Location loc);

private:
  /// Private constructor: only available to 'SYCLFuncRegistry'.
  SYCLFuncDescriptor(FuncId id, StringRef name, Type outputTy,
                     ArrayRef<Type> argTys)
      : id(id), name(name), outputTy(outputTy),
        argTys(argTys.begin(), argTys.end()) {}

  // Inject the declaration for this function into the module.
  void declareFunction(ModuleOp &module, OpBuilder &b);

private:
  FuncId id;                   // unique identifier for a SYCL function
  StringRef name;              // SYCL function name
  Type outputTy;               // SYCL function output type
  SmallVector<Type, 4> argTys; // SYCL function arguments types
  FlatSymbolRefAttr funcRef;   // Reference to the SYCL function declaration
};

/// \class SYCLFuncRegistry
/// Singleton class representing the set of SYCL functions callable from the
/// compiler.
class SYCLFuncRegistry {
public:
  ~SYCLFuncRegistry() { instance = nullptr; }

  static const SYCLFuncRegistry create(ModuleOp &module, OpBuilder &builder);

  const SYCLFuncDescriptor &getFuncDesc(SYCLFuncDescriptor::FuncId id) const {
    assert(
        (registry.find(id) != registry.end()) &&
        "function identified by 'id' not found in the SYCL function registry");
    return registry.at(id);
  }

private:
  SYCLFuncRegistry(ModuleOp &module, OpBuilder &builder);

  using Registry = std::map<SYCLFuncDescriptor::FuncId, SYCLFuncDescriptor>;
  static SYCLFuncRegistry *instance;
  Registry registry;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLFUNCREGISTRY_H
