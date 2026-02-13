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

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"

using namespace llvm;

/// This class allows for the recovery of typed pointer types from LLVM opaque
/// pointer types. A detailed description of how this algorithm works may be
/// found in the file comment of SPIRVTypeScavenger.cpp.
class SPIRVTypeScavenger {
  /// The mapping from type variables to concrete types.
  std::vector<Type *> TypeVariables;

  /// The structure storing which type variables have been unified.
  IntEqClasses UnifiedTypeVars;

  /// Replace all ptr types found within T with new type variables.
  Type *allocateTypeVariable(Type *T);

  /// Replace all type variables found within T with their concrete types. If
  /// the type variable doesn't have a concrete type yet, the type variable will
  /// be retained.
  Type *substituteTypeVariables(Type *T);

  /// Try to resolve all type variables into concrete types using knowledge that
  /// T1 and T2 have to be the same type. If T1 and T2 cannot be made the same
  /// type, return false (and callers will know they need to insert synthetic
  /// bitcasts to guarantee equality).
  bool unifyType(Type *T1, Type *T2);

  /// This stores the Value -> corrected type mapping for the module. It is
  /// expected that all instructions, arguments, and global values will appear
  /// in this mapping, while constants are not expected to be listed here.
  ValueMap<Value *, Type *> DeducedTypes;

  /// Store associated type variables for certain instructions. In the case
  /// where a return value has an association with an operand, it's necessary
  /// that the type variable used to generate type rules be the same for all
  /// invocations of getTypeRules. This variable allows storage of such
  /// variables.
  ValueMap<Value *, Type *> AssociatedTypeVariables;

  /// A type rule, which expresses that the given operand of a User must have
  /// the given type (which may contain type variables).
  struct TypeRule {
    unsigned OpNo;
    bool LhsIndirect;
    bool RhsIndirect;
    PointerUnion<Type *, Use *> Target;
    TypeRule(unsigned A, bool AIndirect, Type *B, bool BIndirect)
        : OpNo(A), LhsIndirect(AIndirect), RhsIndirect(BIndirect), Target(B) {}
    TypeRule(unsigned A, bool AIndirect, Use *B, bool BIndirect)
        : OpNo(A), LhsIndirect(AIndirect), RhsIndirect(BIndirect), Target(B) {}

    /// Establishes typeof(operand) == concrete type
    static TypeRule is(unsigned OpIndex, Type *Ty) {
      return TypeRule(OpIndex, false, Ty, false);
    }
    /// Establishes typeof(operand) == concrete type
    static TypeRule is(Use &U, Type *Ty) {
      return TypeRule::is(U.getOperandNo(), Ty);
    }
    /// Establishes typeof(operand) == typeof(operand)
    static TypeRule is(User &U, unsigned Op1, unsigned Op2) {
      return TypeRule(Op1, false, &U.getOperandUse(Op2), false);
    }
    /// Establishes typedptr(typeof(operand)) == typedptr(typeof(operand))
    /// (this is useful when the address spaces do not need to match).
    static TypeRule isIndirect(User &U, unsigned Op1, unsigned Op2) {
      return TypeRule(Op1, true, &U.getOperandUse(Op2), true);
    }
    /// Establishes typeof(operand) == typedptr(concrete type)
    static TypeRule pointsTo(Use &U, Type *Ty) {
      return TypeRule(U.getOperandNo(), false, Ty, true);
    }
    /// Establishes typeof(operand) == typedptr(concrete type)
    static TypeRule pointsTo(User &U, unsigned OpIndex, Type *Ty) {
      return TypeRule::pointsTo(U.getOperandUse(OpIndex), Ty);
    }
    /// Establishes typeof(mem operand) == typedptr(typeof(val operand))
    static TypeRule pointsTo(User &U, unsigned MemIndex, unsigned ValIndex) {
      return TypeRule(MemIndex, false, &U.getOperandUse(ValIndex), true);
    }
    /// Establishes typeof(operand) == typedptr(typeof(return))
    static TypeRule pointsToReturn(User &U, unsigned OpIndex) {
      return TypeRule(RETURN_OPERAND, true, &U.getOperandUse(OpIndex), false);
    }
    /// Establishes typeof(return) == concrete type
    static TypeRule returns(Type *Ty) {
      return TypeRule(RETURN_OPERAND, false, Ty, false);
    }
    /// Establishes typeof(return) == typedptr(concrete type)
    static TypeRule returnsPointerTo(Type *Ty) {
      return TypeRule(RETURN_OPERAND, false, Ty, true);
    }
    /// Establishes typeof(return) == typeof(operand)
    static TypeRule propagates(Use &U) {
      return TypeRule(RETURN_OPERAND, false, &U, false);
    }
    /// Establishes typeof(return) == typeof(operand)
    static TypeRule propagates(User &U, unsigned OpIndex) {
      return TypeRule::propagates(U.getOperandUse(OpIndex));
    }
    /// Establishes typedptr(typeof(return)) == typedptr(typeof(operand))
    static TypeRule propagatesIndirect(Use &U) {
      return TypeRule(RETURN_OPERAND, true, &U, true);
    }
    /// Establishes typedptr(typeof(return)) == typedptr(typeof(operand))
    static TypeRule propagatesIndirect(User &U, unsigned OpIndex) {
      return TypeRule::propagatesIndirect(U.getOperandUse(OpIndex));
    }
  };

  /// This is a value that allows the ability to express the type of a value as
  /// a whole in a typing rule.
  static constexpr unsigned RETURN_OPERAND = ~0U;

  /// Turn a type rule into an operand and a type to check for. If the type of
  /// the operand and the type to check against cannot be unified, then a
  /// bitcast will need to be inserted for the use.
  std::pair<Use &, Type *> getTypeCheck(Instruction &I, const TypeRule &Rule);

  /// Retrieve the list of typing rules for an instruction.
  void getTypeRules(Instruction &I, SmallVectorImpl<TypeRule> &Rules);

  /// Get the best guess for the type of the value, applying any type rules to
  /// the return value of an instruction that exist. The return type may refer
  /// to type variables that have yet to be resolved, if the type rules are
  /// insufficient to establish a typed pointer type for the instruction.
  Type *getTypeAfterRules(Value *V);

  /// Enforce that the pointer element types of all operands of the instruction
  /// matches the type that the instruction itself requires. If a pointer
  /// element type of one of the operands is deferred, this will type the use
  /// correctly.
  void correctUseTypes(Instruction &I);

  /// This assigns known pointer element types for the parameters of a function.
  /// This method should be called for all functions before doing any type
  /// analysis on the module.
  void deduceFunctionType(Function &F);

  /// This computes known type rules of a call to an LLVM intrinsic or specific
  /// well-known function name. Returns true if the call was known to this
  /// function.
  bool typeIntrinsicCall(CallBase &CB, SmallVectorImpl<TypeRule> &TypeRules);

  /// Get the type rules for checking argument and return value compatibility
  /// for the function type being called. This is meant to help unify cases
  /// for indirect function calls.
  void typeFunctionParams(CallBase &CB, FunctionType *FT, unsigned ArgStart,
                          bool IncludeRet,
                          SmallVectorImpl<TypeRule> &TypeRules);

  /// Compute the type of a global variable or global alias, based on the type
  /// of the initializer (which may be null for global variables).
  void typeGlobalValue(GlobalValue &GV, Constant *Init);

  /// Compute pointer element types for all pertinent values in the module.
  void typeModule(Module &M);

  /// This stores a list of instructions whose pointer element types are
  /// currently being investigated, to avoid the possibility of infinite cycles.
  std::vector<Value *> VisitStack;

public:
  explicit SPIRVTypeScavenger(Module &M) : UnifiedTypeVars(1024) {
    typeModule(M);
  }

  /// Get the type of the value, with pointer types replaced with
  /// TypedPointerType types instead.
  Type *getScavengedType(Value *V);

  /// Get the deduced function type for a function, with pointer types replaced
  /// with TypedPointerTypes (maybe including type variables).
  FunctionType *getFunctionType(Function *F);
};

#endif // SPIRVTYPESCAVENGER_H
