// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// @file
///
/// @brief Manipulation of memory operations like loads and stores.

#ifndef VECZ_MEMORY_OPERATIONS_H_INCLUDED
#define VECZ_MEMORY_OPERATIONS_H_INCLUDED

#include <inttypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <optional>

namespace llvm {
class CallInst;
class LoadInst;
class StoreInst;
class Argument;
class Function;
class Instruction;
class Value;
class Type;
}  // namespace llvm

namespace vecz {

class VectorizationContext;
struct UniformValueResult;

/// @brief Return or declare a masked memory operation builtin function.
///
/// @param[in] Ctx Context used to manipulate internal builtins.
/// @param[in] DataTy Loaded type or stored value type.
/// @param[in] PtrTy Pointer type. Must either be opaque or have its pointee
/// type match DataTy.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] IsLoad true if defined a masked load, false if a masked store.
/// @param[in] IsVP true if defining a vector-predicated operation
///
/// @return Masked builtin function.
llvm::Function *getOrCreateMaskedMemOpFn(VectorizationContext &Ctx,
                                         llvm::Type *DataTy,
                                         llvm::PointerType *PtrTy,
                                         unsigned Alignment, bool IsLoad,
                                         bool IsVP);

/// @brief Create a call to a masked load operation builtin function.
///
/// @param[in] Ctx Context used to retrieve the builtin function.
/// @param[in] Ty Type to load from memory.
/// @param[in] Ptr Pointer. Internally bitcast to point to Ty.
/// @param[in] Mask Mask.
/// @param[in] EVL vector length as i32, else null (full width operation).
/// @param[in] Alignment Alignment
/// @param[in] Name Name to give to the call instruction.
/// @param[in] InsertBefore Insertion point for the call instruction.
///
/// @return Call instruction or null on error.
llvm::CallInst *createMaskedLoad(VectorizationContext &Ctx, llvm::Type *Ty,
                                 llvm::Value *Ptr, llvm::Value *Mask,
                                 llvm::Value *EVL, unsigned Alignment,
                                 llvm::Twine Name = "",
                                 llvm::Instruction *InsertBefore = nullptr);

/// @brief Create a call to a masked store operation builtin function.
///
/// @param[in] Ctx Context used to retrieve the builtin function.
/// @param[in] Data Stored value.
/// @param[in] Ptr Pointer. Internally bitcast to pointer to Data's type.
/// @param[in] Mask Mask.
/// @param[in] EVL vector length as i32, else null (full width operation).
/// @param[in] Alignment Alignment
/// @param[in] Name Name to give to the call instruction.
/// @param[in] InsertBefore Insertion point for the call instruction.
///
/// @return Call instruction or null on error.
llvm::CallInst *createMaskedStore(VectorizationContext &Ctx, llvm::Value *Data,
                                  llvm::Value *Ptr, llvm::Value *Mask,
                                  llvm::Value *EVL, unsigned Alignment,
                                  llvm::Twine Name = "",
                                  llvm::Instruction *InsertBefore = nullptr);

/// @brief Return or declare a (masked) interleaved memory operation builtin
/// function.

/// @param[in] Ctx Context used to manipulate internal builtins.
/// @param[in] DataTy Loaded type or stored value type.
/// @param[in] PtrTy Pointer type. Must either be opaque or have its pointee
/// type match DataTy's element type.
/// @param[in] Stride The stride of the access. May be null in which case the
/// default stride is used.
/// @param[in] MaskTy The mask type. May be null for an unmasked operation.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] IsLoad true if defining a load, false if defining a store.
/// @param[in] IsVP true if defining a vector-predicated operation
///
/// @return (Masked) interleaved builtin function.
llvm::Function *getOrCreateInterleavedMemOpFn(
    VectorizationContext &Ctx, llvm::Type *DataTy, llvm::PointerType *PtrTy,
    llvm::Value *Stride, llvm::Type *MaskTy, unsigned Alignment, bool IsLoad,
    bool IsVP);

/// @brief Create a call to a (masked) interleaved load builtin function. Also
/// known as a strided load.
///
/// @param[in] Ctx Vectorization Context used to retrieve the builtin info.
/// @param[in] Ty Type to load from memory
/// @param[in] Ptr Pointer. Internally bitcast to a pointer to Ty's element
/// type.
/// @param[in] Stride The stride of the operation. May be null in which case
/// the default stride is used.
/// @param[in] Mask The mask controlling the operation. May be null in which
/// case an unmasked builtin is called.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] Name Name to give to the call instruction.
/// @param[in] InsertBefore Insertion point for the call instruction.
///
/// @return Call instruction or null on error.
llvm::CallInst *createInterleavedLoad(
    VectorizationContext &Ctx, llvm::Type *Ty, llvm::Value *Ptr,
    llvm::Value *Stride, llvm::Value *Mask, llvm::Value *EVL,
    unsigned Alignment, llvm::Twine Name = "",
    llvm::Instruction *InsertBefore = nullptr);

/// @brief Create a call to a (masked) interleaved store builtin function. Also
/// known as a strided store.
///
/// @param[in] Ctx Vectorization Context used to retrieve the builtin info.
/// @param[in] Data Data value to store to memory.
/// @param[in] Ptr Pointer. Internally bitcast to a pointer to Data's element
/// type.
/// @param[in] Stride The stride of the operation. May be null in which case
/// the default stride is used.
/// @param[in] Mask The mask controlling the operation. May be null in which
/// case an unmasked builtin is called.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] Name Name to give to the call instruction.
/// @param[in] InsertBefore Insertion point for the call instruction.
///
/// @return Call instruction or null on error.
llvm::CallInst *createInterleavedStore(
    VectorizationContext &Ctx, llvm::Value *Data, llvm::Value *Ptr,
    llvm::Value *Stride, llvm::Value *Mask, llvm::Value *EVL,
    unsigned Alignment, llvm::Twine Name = "",
    llvm::Instruction *InsertBefore = nullptr);

/// @brief Return or declare a (masked) scatter/gather memory operation builtin
/// function.
///
/// @param[in] Ctx Context used to manipulate internal builtins.
/// @param[in] DataTy Loaded type or stored value type.
/// @param[in] VecPtrTy Pointer type. Must be a vector of pointers, each of
/// which are either opaque or have a pointee type matching DataTy's element
/// type.
/// @param[in] MaskTy The mask type. May be null for an unmasked operation.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] IsGather true if defining a gather (load), false if defining a
/// scatter (store).
/// @param[in] IsVP true if defining a vector-predicated operation
///
/// @return Scatter/gather builtin function.
llvm::Function *getOrCreateScatterGatherMemOpFn(vecz::VectorizationContext &Ctx,
                                                llvm::Type *DataTy,
                                                llvm::VectorType *VecPtrTy,
                                                llvm::Type *MaskTy,
                                                unsigned Alignment,
                                                bool IsGather, bool IsVP);

/// @brief Create a call to a (masked) gather memory operation builtin
/// function.
///
/// @param[in] Ctx Context used to retrieve the builtin function.
/// @param[in] Ty Type to load from memory.
/// @param[in] VecPtr Pointer value. Must be a vector of pointers, each of
/// which are either opaque or have a pointee type matching DataTy's element
/// type.
/// @param[in] Mask The predicate of the masked instruction. May be null in
/// which case an unmasked builtin is created.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] EVL vector length as i32, else null (full width operation).
/// @param[in] Name Name to give to the call instruction.
/// @param[in] InsertBefore Insertion point for the call instruction.
///
/// @return Call instruction or null on error.
llvm::CallInst *createGather(VectorizationContext &Ctx, llvm::Type *Ty,
                             llvm::Value *VecPtr, llvm::Value *Mask,
                             llvm::Value *EVL, unsigned Alignment,
                             llvm::Twine Name = "",
                             llvm::Instruction *InsertBefore = nullptr);

/// @brief Create a call to a (masked) scatter memory operation builtin
/// function.
///
/// @param[in] Ctx Context used to retrieve the builtin function.
/// @param[in] VecData Value to store to memory.
/// @param[in] VecPtr Pointer value. Must be a vector of pointers, each of
/// which are either opaque or have a pointee type matching DataTy's element
/// type.
/// @param[in] Mask The predicate of the masked instruction. May be null in
/// which case an unmasked builtin is created.
/// @param[in] Alignment Alignment of the operation.
/// @param[in] EVL vector length as i32, else null (full width operation).
/// @param[in] Name Name to give to the call instruction.
/// @param[in] InsertBefore Insertion point for the call instruction.
///
/// @return Call instruction or null on error.
llvm::CallInst *createScatter(VectorizationContext &Ctx, llvm::Value *VecData,
                              llvm::Value *VecPtr, llvm::Value *Mask,
                              llvm::Value *EVL, unsigned Alignment,
                              llvm::Twine Name = "",
                              llvm::Instruction *InsertBefore = nullptr);

/// @brief an enum to distinguish between loads and stores, and between builtin
/// memop calls and native IR memop instructions.
enum class MemOpKind : int {
  /// @brief The object does not contain a valid memory operation.
  Invalid = 0,
  /// @brief The object contains a LLVM load instruction.
  LoadInstruction,
  /// @brief The object contains a LLVM store instruction.
  StoreInstruction,
  /// @brief The object contains a 'load-like' function call.
  LoadCall,
  /// @brief The object contains a 'store-like' function call.
  StoreCall,
};

/// @brief an enum to distinguish between different memory access patterns
enum class MemOpAccessKind : int {
  /// @brief The object does not represent a vecz memop call
  Native = 0,
  /// @brief The object represents a masked memory operation
  Masked,
  /// @brief The object represents an interleaved memory operation
  Interleaved,
  /// @brief The object represents a masked interleaved memory operation
  MaskedInterleaved,
  /// @brief The object represents a scatter/gather memory operation
  ScatterGather,
  /// @brief The object represents a masked scatter/gather memory operation
  MaskedScatterGather,
};

struct MemOp;

/// @brief Describes a memory operation such as a load or a store.
class MemOpDesc {
  /// @brief Type of the data operand for stores, or memory type for loads.
  llvm::Type *DataTy;
  /// @brief Type of the pointer used to access memory.
  llvm::Type *PtrTy;
  /// @brief In the case of masked operations, type of the mask operand.
  llvm::Type *MaskTy;
  /// @brief Identifies the kind of memory operation which is performed.
  MemOpKind Kind;
  /// @brief Idenfities the kind of memory access pattern
  MemOpAccessKind AccessKind;
  /// @brief Whether or not the memory access is vector-length predicated.
  bool IsVLOp;
  /// @brief Memory alignment.
  unsigned Alignment;
  /// @brief Distance between consecutive elements in memory, in number of
  /// elements. Zero means uniform access, one means sequential access.
  /// Negative values mean the access is done is reverse order.
  llvm::Value *Stride;
  /// @brief Index of the data operand, for stores, or negative value.
  int8_t DataOpIdx;
  /// @brief Index of the pointer operand.
  int8_t PtrOpIdx;
  /// @brief Index of the mask operand, for masked operations, or negative
  /// value.
  int8_t MaskOpIdx;
  /// @brief Index of vector length operand, or negative value.
  int8_t VLOpIdx;

  friend struct MemOp;

 public:
  /// @brief Create an invalid memory operation.
  MemOpDesc();

  bool isMaskedMemOp() const { return AccessKind == MemOpAccessKind::Masked; }
  bool isInterleavedMemOp() const {
    return AccessKind == MemOpAccessKind::Interleaved;
  }
  bool isMaskedInterleavedMemOp() const {
    return AccessKind == MemOpAccessKind::MaskedInterleaved;
  }
  bool isScatterGatherMemOp() const {
    return AccessKind == MemOpAccessKind::ScatterGather;
  }
  bool isMaskedScatterGatherMemOp() const {
    return AccessKind == MemOpAccessKind::MaskedScatterGather;
  }

  /// @brief In the case of stores, return the data element being stored.
  llvm::Value *getDataOperand(llvm::Function *F) const {
    return getOperand(F, DataOpIdx);
  }

  /// @brief Return the pointer used by the memory operation.
  llvm::Value *getPointerOperand(llvm::Function *F) const {
    return getOperand(F, PtrOpIdx);
  }

  /// @brief In the case of a masked memory operation, return the mask.
  llvm::Value *getMaskOperand(llvm::Function *F) const {
    return getOperand(F, MaskOpIdx);
  }

  /// @brief In the case of a masked memory operation, return the vector
  /// length.
  llvm::Value *getVLOperand(llvm::Function *F) const {
    return getOperand(F, VLOpIdx);
  }

  /// @brief Index of the data operand of the MemOp
  /// @return The index, or -1 if no data operand
  int8_t getDataOperandIndex() const { return DataOpIdx; }
  /// @brief Index of the pointer operand of the MemOp
  /// @return The index, or -1 if no pointer operand
  int8_t getPointerOperandIndex() const { return PtrOpIdx; }
  /// @brief Index of the mask operand of the MemOp
  /// @return The index, or -1 if no mask operand
  int8_t getMaskOperandIndex() const { return MaskOpIdx; }
  /// @brief Index of the vector-length operand of the MemOp
  /// @return The index, or -1 if no mask operand
  int8_t getVLOperandIndex() const { return VLOpIdx; }

  /// @brief Get what kind of memory operation this is.
  /// @return The kind of the memory operation
  MemOpKind getKind() const { return Kind; }

  /// @brief Get the alignment of the memory operation.
  /// @return The alignment in bytes
  unsigned getAlignment() const { return Alignment; }

  /// @brief In the case of a interleaved memory operation, return the stride.
  /// @return The Value determining the stride
  llvm::Value *getStride() const { return Stride; }
  /// @brief Determine if the stride is an integer whose value can be determined
  /// at compile time.
  /// @return True is the stride is a compile time integer constant
  bool isStrideConstantInt() const;
  /// @brief Get the stride as a constant int. It assumes that it is possible
  /// and valid to do so.
  /// @return The stride in elements
  int64_t getStrideAsConstantInt() const;

  /// @brief Return the type of data element being accessed in memory.
  /// @return The type of the data element being accessed in memory.
  llvm::Type *getDataType() const { return DataTy; }

  /// @brief Return the type of the pointer operand.
  /// @return The type the pointer operand
  llvm::Type *getPointerType() const { return PtrTy; }

  /// @brief Return the specified operand from the function.
  ///
  /// @param[in] F Function to retrieve the operand from.
  /// @param[in] OpIdx Index of the operand to retrieve.
  ///
  /// @return Operand or null.
  llvm::Argument *getOperand(llvm::Function *F, int OpIdx) const;

  /// @brief Determine whether the given function is a memory operation.
  /// If that's the case, the descriptor is populated and returned.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return A MemOpDesc if the given function is a memory operation.
  /// std::nullopt otherwise.
  static std::optional<MemOpDesc> analyzeMemOpFunction(llvm::Function &F);

  /// @brief Determine whether the given function is a masked memory operation.
  /// If that's the case, the descriptor is populated and returned.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return A MemOpDesc if the given function is a masked memory operation.
  /// std::nullopt otherwise.
  static std::optional<MemOpDesc> analyzeMaskedMemOp(llvm::Function &F);

  /// @brief Determine whether the given function is an interleaved memory
  /// operation or not. If that's the case, the descriptor is populated and
  /// returned.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return A MemOpDesc if the given function is an interleaved memory
  /// operation. std::nullopt otherwise.
  static std::optional<MemOpDesc> analyzeInterleavedMemOp(llvm::Function &F);

  /// @brief Determine whether the given function is a masked interleaved memory
  /// operation or not. If that's the case, the descriptor is populated and
  /// returned.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return A MemOpDesc if the given function is a masked interleaved memory
  /// operation. std::nullopt otherwise.
  static std::optional<MemOpDesc> analyzeMaskedInterleavedMemOp(
      llvm::Function &F);

  /// @brief Determine whether the given function is a scatter/gather memory
  /// operation or not. If that's the case, the descriptor is populated and
  /// returned.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return A MemOpDesc if the given function is a scatter/gather operation.
  /// std::nullopt otherwise.
  static std::optional<MemOpDesc> analyzeScatterGatherMemOp(llvm::Function &F);

  /// @brief Determine whether the given function is a scatter/gather memory
  /// operation or not. If that's the case, the descriptor is populated and
  /// returned.
  ///
  /// @param[in] F Function to analyze.
  ///
  /// @return A MemOpDesc if the given function is a masked scatter/gather
  /// operation. std::nullopt otherwise.
  static std::optional<MemOpDesc> analyzeMaskedScatterGatherMemOp(
      llvm::Function &F);

  /// @brief Determine whether the operation is a load or not.
  bool isLoad() const {
    switch (Kind) {
      default:
        return false;
      case MemOpKind::LoadInstruction:
      case MemOpKind::LoadCall:
        return true;
    }
  }

  /// @brief Determine whether the operation is a store or not.
  bool isStore() const {
    switch (Kind) {
      default:
        return false;
      case MemOpKind::StoreInstruction:
      case MemOpKind::StoreCall:
        return true;
    }
  }

  /// @brief Determine whether the operation is an instruction or not.
  bool isLoadStoreInst() const {
    switch (Kind) {
      default:
        return false;
      case MemOpKind::LoadInstruction:
      case MemOpKind::StoreInstruction:
        return true;
    }
  }

  bool isVLOp() const { return IsVLOp; }
};

/// @brief Wrapper that combines a memory operation descriptor and instruction.
/// This allows manipulating different kinds of memory operations (load and
/// store instructions, vecz builtins) in the same way.
struct MemOp {
  /// @brief Create an invalid memory operation.
  MemOp() {}
  /// @brief Create a memory operation from an instruction and an existing
  /// memory operation descriptor.
  ///
  /// @param[in] I Memory instruction.
  /// @param[in] Desc Memory operation descriptor.
  MemOp(llvm::Instruction *I, const MemOpDesc &Desc);
  /// @brief Create a memory operation from an instruction.
  /// @param[in] I Instruction that may be a memory operation.
  static std::optional<MemOp> get(llvm::Instruction *I);
  /// @brief Create a memory operation from an instruction and an existing
  /// memory operation descriptor.
  ///
  /// @param[in] CI Memory builtin call instruction.
  /// @param[in] AccessKind the kind of access to consider
  static std::optional<MemOp> get(llvm::CallInst *CI,
                                  MemOpAccessKind AccessKind);

  /// @brief Access the memory operation descriptor.
  const MemOpDesc &getDesc() const { return Desc; }

  /// @brief Access the memory operation descriptor.
  MemOpDesc &getDesc() { return Desc; }

  /// @brief Return the instruction that performs the memory operation.
  llvm::Instruction *getInstr() const { return Ins; }

  /// @brief Return the alignment in bytes.
  unsigned getAlignment() const { return Desc.getAlignment(); }

  /// @brief In the case of a interleaved memory operation, return the stride.
  llvm::Value *getStride() const { return Desc.getStride(); }

  /// @brief Return the type of data element being accessed in memory.
  llvm::Type *getDataType() const { return Desc.getDataType(); }

  /// @brief Return the type of the pointer operand.
  llvm::Type *getPointerType() const { return Desc.getPointerType(); }

  /// @brief Determine whether the operation is a load or not.
  bool isLoad() const { return Desc.isLoad(); }

  /// @brief Determine whether the operation is a store or not.
  bool isStore() const { return Desc.isStore(); }

  /// @brief Determine whether the operation is an instruction or not.
  bool isLoadStoreInst() const { return Desc.isLoadStoreInst(); }

  /// @brief Determine whether the operation is a masked memop call
  bool isMaskedMemOp() const { return Desc.isMaskedMemOp(); }

  /// @brief Determine whether the operation is a scatter/gather memop call
  bool isMaskedScatterGatherMemOp() const {
    return Desc.isMaskedScatterGatherMemOp();
  }

  /// @brief Determine whether the operation is a masked interleaved memop call
  bool isMaskedInterleavedMemOp() const {
    return Desc.isMaskedInterleavedMemOp();
  }

  /// @brief In the case of stores, return the data element being stored.
  /// @return Data operand or null.
  llvm::Value *getDataOperand() const;
  /// @brief Return the pointer used by the memory operation.
  /// @return Pointer used by the memory operation or null for invalid
  /// operations.
  llvm::Value *getPointerOperand() const;
  /// @brief In the case of a masked memory operation, return the mask.
  /// @return Mask operand or null.
  llvm::Value *getMaskOperand() const;

  /// @brief In the case of stores, set the data element being stored.
  /// @return true on success.
  bool setDataOperand(llvm::Value *V);
  /// @brief Set the pointer used by the memory operation.
  /// @return true on success.
  bool setPointerOperand(llvm::Value *V);
  /// @brief In the case of a masked memory operation, set the mask.
  /// @return true on success.
  bool setMaskOperand(llvm::Value *V);

  /// @brief In the case of a builtin memory operation, return the call.
  /// @return Call instruction or null.
  llvm::CallInst *getCall() const;

  /// @brief Determine if the stride is an integer whose value can be determined
  /// at compile time.
  /// @return True is the stride is a compile time integer constant
  bool isStrideConstantInt() const { return Desc.isStrideConstantInt(); }
  /// @brief Get the stride as a constant int. It assumes that it is possible
  /// and valid to do so.
  /// @return The stride in elements
  int64_t getStrideAsConstantInt() const {
    return Desc.getStrideAsConstantInt();
  }

 private:
  /// @brief Access an operand of the call instruction.
  ///
  /// @param[in] OpIdx Index of the operand to access.
  ///
  /// @return Specified operand of the call instruction.
  llvm::Value *getCallOperand(int OpIdx) const;

  /// @brief Set an operand of the call instruction.
  ///
  /// @param[in] OpIdx Index of the operand to access.
  /// @param[in] V the Value to set
  ///
  /// @return true on success.
  bool setCallOperand(int OpIdx, llvm::Value *V);

  /// @brief Instruction that performs the memory operation.
  llvm::Instruction *Ins;
  /// @brief Describes the memory operation.
  MemOpDesc Desc;
};

namespace {
inline llvm::ConstantInt *getSizeInt(llvm::IRBuilder<> &B, int64_t val) {
  if (B.GetInsertBlock()->getModule()->getDataLayout().getPointerSize() == 4) {
    return B.getInt32(val);
  }
  return B.getInt64(val);
}

inline llvm::IntegerType *getSizeTy(llvm::Module &M) {
  if (M.getDataLayout().getPointerSize() == 4) {
    return llvm::Type::getInt32Ty(M.getContext());
  }
  return llvm::Type::getInt64Ty(M.getContext());
}

inline llvm::IntegerType *getSizeTy(llvm::IRBuilder<> &B) {
  return getSizeTy(*(B.GetInsertBlock()->getModule()));
}
}  // namespace
}  // namespace vecz

#endif  // VECZ_MEMORY_OPERATIONS_H_INCLUDED
