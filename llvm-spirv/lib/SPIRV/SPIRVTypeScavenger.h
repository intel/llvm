//===- SPIRVTypeScavenger.h - Recover pointer types in opaque pointer IR --===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2022 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of The Khronos Group, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements the necessary logic to recover pointer types from LLVM
// IR for the output SPIR-V file after LLVM IR completes its transition to
// opaque pointers.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRVTYPESCAVENGER_H
#define SPIRVTYPESCAVENGER_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"

using namespace llvm;

/// This class allows for the recovery of pointer element types from LLVM
/// opaque pointer types.
class SPIRVTypeScavenger {
  /// A representation of a pointer whose type will be determined by uses. This
  /// will include every value that needs to be assigned the same type.
  struct DeferredType {
    std::vector<Value *> Values;
  };

  /// This is called when a deferred type is fixed to a known use.
  void fixType(DeferredType &Deferred, Type *AssignedType);

  /// This merges two deferred types into one deferred type.
  void mergeType(DeferredType *A, DeferredType *B);

  /// A representation of the possible states of a type internal to this pass:
  /// it may be either
  /// * Something with a fixed LLVM type (this is a Type *)
  /// * A type whose pointer element type is yet unknown (DeferredType *)
  /// * A multi-level pointer type (this is a Value *, whose type is what this
  ///   pointer type will point to). The latter should only exist for
  ///   pointer operands of memory operations that return ptr.
  typedef PointerUnion<Type *, DeferredType *, Value *> DeducedType;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, DeducedType Ty) {
    if (auto *AsTy = dyn_cast<Type *>(Ty))
      return OS << *AsTy;
    if (auto *AsDeferred = dyn_cast<DeferredType *>(Ty))
      return OS << "deferred type for " << *AsDeferred->Values[0];
    if (auto *AsValue = dyn_cast<Value *>(Ty))
      return OS << "points to " << *AsValue;
    return OS;
  }

  /// Compute the pointer element type of a value, solely based on its
  /// definition. The value must be pointer-valued.
  DeducedType computePointerElementType(Value *V);

  /// This stores the Value -> pointer element type mapping for the module.
  ValueMap<Value *, DeducedType> DeducedTypes;

  /// Enforce that the pointer element types of all operands of the instruction
  /// matches the type that the instruction itself requires. If a pointer
  /// element type of one of the operands is deferred, this will type the use
  /// correctly.
  void correctUseTypes(Instruction &I);

  /// This assigns known pointer element types for the parameters of a function.
  /// This method should be called for all functions before doing any type
  /// analysis on the module.
  void deduceFunctionType(Function &F);

  /// This assigns known pointer element types for parameters of LLVM
  /// intrinsics.
  void deduceIntrinsicTypes(Function &F, Intrinsic::ID Id);

  /// Compute pointer element types for all pertinent values in the module.
  void typeModule(Module &M);

public:
  explicit SPIRVTypeScavenger(Module &M) { typeModule(M); }

  /// This type represents the type that a pointer element type of a type. If it
  /// is a Type value, then the pointee type represents a pointer to that type.
  /// If it is a Value value, then the pointee type is the type of that value
  /// (which should be a pointer-typed value.)
  typedef PointerUnion<Type *, Value *> PointeeType;

  /// Get the pointer element type of the value.
  /// If the type is a multi-level pointer, then PointeeType will be a Value
  /// whose pointee type can be recursively queried through this method.
  /// Otherwise, it will be a pointer to the Type returned by this method.
  PointeeType getPointerElementType(Value *V);

  /// Get the pointer element type of an argument of the given function. Since
  /// this type is guaranteed to not be a multi-level pointer type, the result
  /// is an LLVM type instead of a PointeeType.
  Type *getArgumentPointerElementType(Function *F, unsigned ArgNo);
};

#endif // SPIRVTYPESCAVENGER_H
