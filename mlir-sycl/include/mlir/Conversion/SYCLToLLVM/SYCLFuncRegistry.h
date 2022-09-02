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
  /// Enumerates SYCL functions.
  // clang-format off
  enum class FuncId {
    Unknown, 

    // Member functions for the sycl:accessor class.
    AccessorInt1ReadWriteGlobalBufferFalseInit,  // sycl::accessor<int, 1, read_write, global_buffer, (placeholder)0>::
                                                 //   __init(int AS1*, sycl::range<1>, sycl::range<1>, sycl::id<1>)

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
  enum class FuncKind {
    Unknown,
    Accessor, // sycl::accessor class
    Id,       // sycl::id<n> class
    Range,    // sycl::range<n> class
  };

  /// Each descriptor is uniquely identified by the pair {FuncId, FuncKind}.
  class Id {
  public:
    friend class SYCLFuncRegistry;
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Id &);

    Id(FuncId id, FuncKind kind) : funcId(id), funcKind(kind) {
      assert(funcId != FuncId::Unknown && "Illegal function id");
      assert(funcKind != FuncKind::Unknown && "Illegal function id kind");
    }

    /// Maps a FuncKind to a descriptive name.
    static std::map<SYCLFuncDescriptor::FuncKind, std::string> funcKindToName;

    /// Maps a descriptive name to a FuncKind.
    static std::map<std::string, SYCLFuncDescriptor::FuncKind> nameToFuncKind;

  private:
    FuncId funcId = FuncId::Unknown;
    FuncKind funcKind = FuncKind::Unknown;
  };

  /// Returns true if the given \p funcId is valid.
  virtual bool isValid(FuncId funcId) const { return false; };

  /// Call the SYCL constructor identified by \p funcId with the given \p args.
  static Value call(FuncId funcId, ValueRange args,
                    const SYCLFuncRegistry &registry, OpBuilder &b,
                    Location loc);

protected:
  SYCLFuncDescriptor(FuncId funcId, FuncKind kind, StringRef name,
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
     << ", funcKind=" << SYCLFuncDescriptor::Id::funcKindToName[id.funcKind];
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
    using FuncKind = SYCLFuncDescriptor::FuncKind;                             \
                                                                               \
  private:                                                                     \
    ClassName(FuncId funcId, StringRef name, Type outputTy,                    \
              ArrayRef<Type> argTys)                                           \
        : SYCLFuncDescriptor(funcId, ClassKind, name, outputTy, argTys) {      \
      assert(isValid(funcId) && "Invalid function id");                        \
    }                                                                          \
    bool isValid(FuncId) const override;                                       \
  };
DEFINE_CLASS(SYCLAccessorFuncDescriptor, FuncKind::Accessor)
DEFINE_CLASS(SYCLIdFuncDescriptor, FuncKind::Id)
DEFINE_CLASS(SYCLRangeFuncDescriptor, FuncKind::Range)
#undef DEFINE_CLASS

/// \class SYCLFuncRegistry
/// Singleton class representing the set of SYCL functions callable from the
/// compiler.
class SYCLFuncRegistry {
  using FuncId = SYCLFuncDescriptor::FuncId;
  using FuncKind = SYCLFuncDescriptor::FuncKind;
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
  /// descriptor that matches the given \p funcKind and signature.
  FuncId getFuncId(FuncKind funcKind, Type retType, TypeRange argTypes) const;

private:
  SYCLFuncRegistry(ModuleOp &module, OpBuilder &builder);

  /// Declare sycl::accessor<n> function descriptors and add them to the
  /// registry.
  void declareAccessorFuncDescriptors(LLVMTypeConverter &converter,
                                      ModuleOp &module, OpBuilder &builder);

  /// Declare sycl::id<n> function descriptors and add them to the registry.
  void declareIdFuncDescriptors(LLVMTypeConverter &converter, ModuleOp &module,
                                OpBuilder &builder);

  /// Declare sycl::range<n> function descriptors and add them to the registry.
  void declareRangeFuncDescriptors(LLVMTypeConverter &converter,
                                   ModuleOp &module, OpBuilder &builder);

  /// Declare function descriptors and add them to the registry.
  void declareFuncDescriptors(std::vector<SYCLFuncDescriptor> &descriptors,
                              ModuleOp &module, OpBuilder &builder);

  static SYCLFuncRegistry *instance;
  Registry registry;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLFUNCREGISTRY_H
