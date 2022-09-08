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

  /// Enumerates the descriptor kind.
  enum class Kind {
    Unknown,
  };

  /// Each descriptor is uniquely identified by the pair {FuncId, Kind}.
  class Id {
  public:
    friend class SYCLFuncRegistry;
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Id &);

    Id(FuncId id, Kind kind) : funcId(id), kind(kind) {
      assert(funcId != FuncId::Unknown && "Illegal function id");
      assert(kind != Kind::Unknown && "Illegal descriptor kind");
    }

    static std::map<SYCLFuncDescriptor::Kind, std::string> kindToName;

    /// Maps a descriptive name to a Kind.
    static std::map<std::string, SYCLFuncDescriptor::Kind> nameToKind;

  private:
    FuncId funcId = FuncId::Unknown;
    Kind kind = Kind::Unknown;
  };

  /// Returns true if the given \p funcId is valid.
  virtual bool isValid(FuncId funcId) const { return false; };

  /// Call the SYCL constructor identified by \p funcId with the given \p args.
  static Value call(FuncId funcId, ValueRange args,
                    const SYCLFuncRegistry &registry, OpBuilder &b,
                    Location loc);

protected:
  SYCLFuncDescriptor(FuncId funcId, Kind kind, StringRef name,
                     Type outputTy, ArrayRef<Type> argTys)
      : descId(funcId, kind), name(name), outputTy(outputTy),
        argTys(argTys.begin(), argTys.end()) {}

private:
  /// Inject the declaration for this function into the module.
  void declareFunction(ModuleOp &module, OpBuilder &b);

  Id descId;                   // unique identifier
  StringRef name;              // SYCL function name
  Type outputTy;               // SYCL function output type
  SmallVector<Type, 4> argTys; // SYCL function arguments types
  FlatSymbolRefAttr funcRef;   // Reference to the SYCL function
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SYCLFuncDescriptor::Id &id) {
  os << "funcId=" << (int)id.funcId
     << ", kind=" << SYCLFuncDescriptor::Id::kindToName.at(id.kind);
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SYCLFuncDescriptor &desc) {
  os << "(" << desc.descId << ", name='" << desc.name.str() << "')";
  return os;
}

//===----------------------------------------------------------------------===//
// Derived classes specializing the generic SYCLFuncDescriptor.
//===----------------------------------------------------------------------===//

#define DEFINE_CLASS(ClassName, ClassKind)                                     \
  class ClassName : public SYCLFuncDescriptor {                                \
  public:                                                                      \
    friend class SYCLFuncRegistry;                                             \
    using FuncId = SYCLFuncDescriptor::FuncId;                                 \
    using Kind = SYCLFuncDescriptor::Kind;                                     \
                                                                               \
  private:                                                                     \
    ClassName(FuncId funcId, StringRef name, Type outputTy,                    \
              ArrayRef<Type> argTys)                                           \
        : SYCLFuncDescriptor(funcId, ClassKind, name, outputTy, argTys) {      \
      assert(isValid(funcId) && "Invalid function id");                        \
    }                                                                          \
    virtual ~ClassName() {}                                                    \
    bool isValid(FuncId) const override;                                       \
  };
#undef DEFINE_CLASS

/// \class SYCLFuncRegistry
/// Singleton class representing the set of SYCL functions callable from the
/// compiler.
class SYCLFuncRegistry {
  using FuncId = SYCLFuncDescriptor::FuncId;
  using Kind = SYCLFuncDescriptor::Kind;
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
  /// descriptor that matches the given \p kind and signature.
  FuncId getFuncId(Kind kind, Type retType, TypeRange argTypes) const;

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
