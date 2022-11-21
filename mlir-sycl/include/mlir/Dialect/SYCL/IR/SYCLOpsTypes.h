//===--- SYCLOpsTypes.h ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_TYPES_H_
#define MLIR_SYCL_OPS_TYPES_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace sycl {
enum class MemoryAccessMode {
  Read = 1024,
  Write,
  ReadWrite,
  DiscardWrite,
  DiscardReadWrite,
  Atomic
};

enum class MemoryTargetMode {
  GlobalBuffer = 2014,
  ConstantBuffer,
  Local,
  Image,
  HostBuffer,
  HostImage,
  ImageArray
};
} // namespace sycl
} // namespace mlir

////////////////////////////////////////////////////////////////////////////////
// Type Storage Definitions
////////////////////////////////////////////////////////////////////////////////

namespace mlir {
namespace sycl {
namespace detail {
struct IDTypeStorage : public TypeStorage {
  using KeyTy = unsigned int;

  IDTypeStorage(const KeyTy &Key) : Dimension(Key) {}

  bool operator==(const KeyTy &Key) const { return Key == KeyTy{Dimension}; }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_value(Key);
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static IDTypeStorage *construct(TypeStorageAllocator &Allocator,
                                  const KeyTy &Key) {
    return new (Allocator.allocate<IDTypeStorage>()) IDTypeStorage(Key);
  }

  unsigned int Dimension;
};

struct AccessorCommonTypeStorage : public TypeStorage {
  using KeyTy = uint8_t;

  AccessorCommonTypeStorage() = default;

  bool operator==(const KeyTy &Key) const { return true; }

  static AccessorCommonTypeStorage *construct(TypeStorageAllocator &Allocator,
                                              const KeyTy &Key) {
    return new (Allocator.allocate<AccessorCommonTypeStorage>())
        AccessorCommonTypeStorage();
  }
};

struct AccessorTypeStorage : public TypeStorage {
  using KeyTy =
      std::tuple<mlir::Type, unsigned int, mlir::sycl::MemoryAccessMode,
                 mlir::sycl::MemoryTargetMode,
                 llvm::SmallVector<mlir::Type, 4>>;

  AccessorTypeStorage(const KeyTy &Key)
      : Type(std::get<0>(Key)), Dimension(std::get<1>(Key)),
        AccessMode(std::get<2>(Key)), TargetMode(std::get<3>(Key)),
        Body(std::get<4>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Type, Dimension, AccessMode, TargetMode, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key),
                              std::get<2>(Key), std::get<3>(Key),
                              std::get<4>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static AccessorTypeStorage *construct(TypeStorageAllocator &Allocator,
                                        const KeyTy &Key) {
    return new (Allocator.allocate<AccessorTypeStorage>())
        AccessorTypeStorage(Key);
  }

  mlir::Type Type;
  unsigned int Dimension;
  mlir::sycl::MemoryAccessMode AccessMode;
  mlir::sycl::MemoryTargetMode TargetMode;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct RangeTypeStorage : public TypeStorage {
  using KeyTy = unsigned int;

  RangeTypeStorage(const KeyTy &Key) : Dimension(Key) {}

  bool operator==(const KeyTy &Key) const { return Key == KeyTy{Dimension}; }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_value(Key);
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static RangeTypeStorage *construct(TypeStorageAllocator &Allocator,
                                     const KeyTy &Key) {
    return new (Allocator.allocate<RangeTypeStorage>()) RangeTypeStorage(Key);
  }

  unsigned int Dimension;
};

struct NdRangeTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<unsigned int, llvm::SmallVector<mlir::Type, 4>>;

  NdRangeTypeStorage(const KeyTy &Key)
      : Dimension(std::get<0>(Key)), Body(std::get<1>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Dimension, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static NdRangeTypeStorage *construct(TypeStorageAllocator &Allocator,
                                       const KeyTy &Key) {
    return new (Allocator.allocate<NdRangeTypeStorage>())
        NdRangeTypeStorage(Key);
  }

  unsigned int Dimension;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct AccessorImplDeviceStorage : public TypeStorage {
  using KeyTy = std::tuple<unsigned int, llvm::SmallVector<mlir::Type, 4>>;

  AccessorImplDeviceStorage(const KeyTy &Key)
      : Dimension(std::get<0>(Key)), Body(std::get<1>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Dimension, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static AccessorImplDeviceStorage *construct(TypeStorageAllocator &Allocator,
                                              const KeyTy &Key) {
    return new (Allocator.allocate<AccessorImplDeviceStorage>())
        AccessorImplDeviceStorage(Key);
  }

  unsigned int Dimension;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct AccessorSubscriptStorage : public TypeStorage {
  using KeyTy = std::tuple<int, llvm::SmallVector<mlir::Type, 4>>;

  AccessorSubscriptStorage(const KeyTy &Key)
      : CurrentDimension(std::get<0>(Key)), Body(std::get<1>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{CurrentDimension, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static AccessorSubscriptStorage *construct(TypeStorageAllocator &Allocator,
                                             const KeyTy &Key) {
    return new (Allocator.allocate<AccessorSubscriptStorage>())
        AccessorSubscriptStorage(Key);
  }

  int CurrentDimension;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct ArrayTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<unsigned int, llvm::SmallVector<mlir::Type, 4>>;

  ArrayTypeStorage(const KeyTy &Key)
      : Dimension(std::get<0>(Key)), Body(std::get<1>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Dimension, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static ArrayTypeStorage *construct(TypeStorageAllocator &Allocator,
                                     const KeyTy &Key) {
    return new (Allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(Key);
  }

  unsigned int Dimension;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct ItemTypeStorage : public TypeStorage {
  using KeyTy =
      std::tuple<unsigned int, bool, llvm::SmallVector<mlir::Type, 4>>;

  ItemTypeStorage(const KeyTy &Key)
      : Dimension(std::get<0>(Key)), WithOffset(std::get<1>(Key)),
        Body(std::get<2>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Dimension, WithOffset, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key),
                              std::get<2>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static ItemTypeStorage *construct(TypeStorageAllocator &Allocator,
                                    const KeyTy &Key) {
    return new (Allocator.allocate<ItemTypeStorage>()) ItemTypeStorage(Key);
  }

  unsigned int Dimension;
  bool WithOffset;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct NdItemTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<unsigned int, llvm::SmallVector<mlir::Type, 4>>;

  NdItemTypeStorage(const KeyTy &Key)
      : Dimension(std::get<0>(Key)), Body(std::get<1>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Dimension, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static NdItemTypeStorage *construct(TypeStorageAllocator &Allocator,
                                      const KeyTy &Key) {
    return new (Allocator.allocate<NdItemTypeStorage>()) NdItemTypeStorage(Key);
  }

  unsigned int Dimension;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct GroupTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<unsigned int, llvm::SmallVector<mlir::Type, 4>>;

  GroupTypeStorage(const KeyTy &Key)
      : Dimension(std::get<0>(Key)), Body(std::get<1>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{Dimension, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static GroupTypeStorage *construct(TypeStorageAllocator &Allocator,
                                     const KeyTy &Key) {
    return new (Allocator.allocate<GroupTypeStorage>()) GroupTypeStorage(Key);
  }

  unsigned int Dimension;
  llvm::SmallVector<mlir::Type, 4> Body;
};

struct VecTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<mlir::Type, int, llvm::SmallVector<mlir::Type, 4>>;

  VecTypeStorage(const KeyTy &Key)
      : DataT(std::get<0>(Key)), NumElements(std::get<1>(Key)),
        Body(std::get<2>(Key)) {}

  bool operator==(const KeyTy &Key) const {
    return Key == KeyTy{DataT, NumElements, Body};
  }

  static llvm::hash_code hashKey(const KeyTy &Key) {
    return llvm::hash_combine(std::get<0>(Key), std::get<1>(Key),
                              std::get<2>(Key));
  }

  static KeyTy getKey(const KeyTy &Key) { return KeyTy{Key}; }

  static VecTypeStorage *construct(TypeStorageAllocator &Allocator,
                                   const KeyTy &Key) {
    return new (Allocator.allocate<VecTypeStorage>()) VecTypeStorage(Key);
  }

  mlir::Type DataT;
  int NumElements;
  llvm::SmallVector<mlir::Type, 4> Body;
};

} // namespace detail
} // namespace sycl
} // namespace mlir

////////////////////////////////////////////////////////////////////////////////
// Trait Definitions
////////////////////////////////////////////////////////////////////////////////

namespace mlir {
namespace sycl {
template <typename Parameter> class SYCLInheritanceTypeInterface {
public:
  template <typename ConcreteType>
  class Trait : public mlir::TypeTrait::TraitBase<ConcreteType, Trait> {};
};
} // namespace sycl
} // namespace mlir

////////////////////////////////////////////////////////////////////////////////
// Type Definitions
////////////////////////////////////////////////////////////////////////////////

namespace mlir {
namespace sycl {

class ArrayType
    : public Type::TypeBase<ArrayType, Type, detail::ArrayTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::ArrayType get(MLIRContext *Context, unsigned int Dimension,
                                   llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class IDType
    : public Type::TypeBase<IDType, Type, detail::IDTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait,
                            mlir::sycl::SYCLInheritanceTypeInterface<
                                mlir::sycl::ArrayType>::Trait> {
public:
  using Base::Base;

  static mlir::sycl::IDType get(MLIRContext *Context, unsigned int Dimension);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
};

class AccessorCommonType
    : public Type::TypeBase<AccessorCommonType, Type,
                            detail::AccessorCommonTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::AccessorCommonType get(MLIRContext *Context);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);
};

class AccessorType
    : public Type::TypeBase<AccessorType, Type, detail::AccessorTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait,
                            mlir::sycl::SYCLInheritanceTypeInterface<
                                mlir::sycl::AccessorCommonType>::Trait> {
public:
  using Base::Base;

  static mlir::sycl::AccessorType get(MLIRContext *Context, mlir::Type Type,
                                      unsigned int Dimension,
                                      mlir::sycl::MemoryAccessMode AccessMode,
                                      mlir::sycl::MemoryTargetMode TargetMode,
                                      llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  mlir::Type getType() const;
  mlir::sycl::MemoryAccessMode getAccessMode() const;
  mlir::sycl::MemoryTargetMode getTargetMode() const;
  mlir::StringRef getAccessModeAsString() const;
  mlir::StringRef getTargetModeAsString() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class RangeType
    : public Type::TypeBase<RangeType, Type, detail::RangeTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait,
                            mlir::sycl::SYCLInheritanceTypeInterface<
                                mlir::sycl::ArrayType>::Trait> {
public:
  using Base::Base;

  static mlir::sycl::RangeType get(MLIRContext *Context,
                                   unsigned int Dimension);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
};

class NdRangeType
    : public Type::TypeBase<NdRangeType, Type, detail::NdRangeTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::NdRangeType get(MLIRContext *Context,
                                     unsigned int Dimension,
                                     llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class AccessorImplDeviceType
    : public Type::TypeBase<AccessorImplDeviceType, Type,
                            detail::AccessorImplDeviceStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::AccessorImplDeviceType
  get(MLIRContext *Context, unsigned int Dimension,
      llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class AccessorSubscriptType
    : public Type::TypeBase<AccessorSubscriptType, Type,
                            detail::AccessorSubscriptStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::AccessorSubscriptType
  get(MLIRContext *Context, int CurrentDimension,
      llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  int getCurrentDimension() const;
  mlir::sycl::AccessorType getAccessorType() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class ItemType
    : public Type::TypeBase<ItemType, Type, detail::ItemTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::ItemType get(MLIRContext *Context, unsigned int Dimension,
                                  bool WithOffset,
                                  llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  bool getWithOffset() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class ItemBaseType
    : public Type::TypeBase<ItemBaseType, Type, detail::ItemTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::ItemBaseType get(MLIRContext *Context,
                                      unsigned int Dimension, bool WithOffset,
                                      llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  bool getWithOffset() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class NdItemType
    : public Type::TypeBase<NdItemType, Type, detail::NdItemTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::NdItemType get(MLIRContext *Context,
                                    unsigned int Dimension,
                                    llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class GroupType
    : public Type::TypeBase<GroupType, Type, detail::GroupTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::GroupType get(MLIRContext *Context, unsigned int Dimension,
                                   llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  unsigned int getDimension() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

class VecType
    : public Type::TypeBase<VecType, Type, detail::VecTypeStorage,
                            mlir::MemRefElementTypeInterface::Trait,
                            mlir::LLVM::PointerElementTypeInterface::Trait> {
public:
  using Base::Base;

  static mlir::sycl::VecType get(MLIRContext *Context, mlir::Type DataT,
                                 int NumElements,
                                 llvm::SmallVector<mlir::Type, 4> Body);
  static mlir::Type parseType(mlir::DialectAsmParser &Parser);

  mlir::Type getDataType() const;
  int getNumElements() const;
  llvm::ArrayRef<mlir::Type> getBody() const;
};

/// Return true if the given \p Ty is a SYCL type.
inline bool isSYCLType(Type Ty) { return isa<SYCLDialect>(Ty.getDialect()); }

// TODO: Modify SYCLDialect::addType() to avoid ever having to modify this
// function when adding new types.

/// Return the list of types derived from the input type.
llvm::SmallVector<mlir::TypeID> getDerivedTypes(mlir::TypeID TypeID);
} // namespace sycl
} // namespace mlir

#endif // MLIR_SYCL_OPS_DIALECT_H_
