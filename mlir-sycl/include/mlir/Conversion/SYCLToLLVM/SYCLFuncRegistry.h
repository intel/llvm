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
class LLVMTypeConverter;

namespace sycl {
class SYCLFuncRegistry;

/// \class SYCLFuncDescriptor
/// Represents a SYCL function (defined in a registry) that can be called by the
/// compiler.
/// Note: when a new enumerator is added, the corresponding SYCLFuncDescriptor
/// needs to be created in SYCLFuncRegistry constructor.
class SYCLFuncDescriptor {
  friend class SYCLFuncRegistry;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const SYCLFuncDescriptor &);

public:
  virtual ~SYCLFuncDescriptor() {}

  /// Enumerates SYCL functions.
  // clang-format off
  enum class FuncId {
    Unknown, 
  };
  // clang-format on

  /// Returns true if the given \p funcId is valid.
  virtual bool isValid(FuncId funcId) const { return false; };

  /// Call the SYCL constructor identified by \p funcId with the given \p args.
  static Value call(FuncId funcId, ValueRange args,
                    const SYCLFuncRegistry &registry, OpBuilder &b,
                    Location loc);

protected:
  SYCLFuncDescriptor(FuncId funcId, StringRef name, Type outputTy,
                     ArrayRef<Type> argTys)
      : funcId(funcId), name(name), outputTy(outputTy),
        argTys(argTys.begin(), argTys.end()) {}

private:
  /// Inject the declaration for this function into the module.
  void declareFunction(ModuleOp &module, OpBuilder &b);

  FuncId funcId;               // unique identifier
  StringRef name;              // SYCL function name
  Type outputTy;               // SYCL function output type
  SmallVector<Type, 4> argTys; // SYCL function arguments types
  FlatSymbolRefAttr funcRef;   // Reference to the SYCL function
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SYCLFuncDescriptor &desc) {
  os << "(funcId=" << (int)desc.funcId << ", name='" << desc.name.str() << "')";
  return os;
}

/// \class SYCLFuncRegistry
/// Singleton class representing the set of SYCL functions callable from the
/// compiler.
class SYCLFuncRegistry {
  using FuncId = SYCLFuncDescriptor::FuncId;
  using Registry = std::map<FuncId, SYCLFuncDescriptor>;

public:
  ~SYCLFuncRegistry() { instance = nullptr; }

  /// Populate the registry.
  static const SYCLFuncRegistry create(ModuleOp &module, OpBuilder &builder);

  /// Return the function descriptor corresponding to the given \p funcId.
  const SYCLFuncDescriptor &getFuncDesc(FuncId funcId) const {
    assert((registry.find(funcId) != registry.end()) &&
           "function identified by 'funcId' not found in the SYCL function "
           "registry");
    return registry.at(funcId);
  }

  /// Returns the SYCLFuncDescriptor::Id::FuncId corresponding to the function
  /// descriptor that matches the given signature.
  FuncId getFuncId(Type retType, TypeRange argTypes) const;

private:
  SYCLFuncRegistry(ModuleOp &module, OpBuilder &builder);

  /// Declare function descriptors and add them to the registry.
  void declareFuncDescriptors(std::vector<SYCLFuncDescriptor> &descriptors,
                              ModuleOp &module, OpBuilder &builder);

  static SYCLFuncRegistry *instance;
  Registry registry;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLFUNCREGISTRY_H
