//===- SPIRVBuiltinHelper.h - Helpers for managing calls to builtins ------===//
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
// This file implements helper functions for adding calls to OpenCL or SPIR-V
// builtin functions, or for rewriting calls to one into calls to the other.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRVBUILTINHELPER_H
#define SPIRVBUILTINHELPER_H

#include "LLVMSPIRVLib.h"
#include "libSPIRV/SPIRVOpCode.h"
#include "libSPIRV/SPIRVType.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypedPointerType.h"

namespace SPIRV {
enum class ManglingRules { None, OpenCL, SPIRV };

namespace detail {
/// This is a helper for triggering the static_assert in mapArg.
template <typename> constexpr bool LegalFnType = false;
} // namespace detail

/// A helper class for changing OpenCL builtin function calls to SPIR-V function
/// calls, or vice versa. Most of the functions will return a reference to the
/// current instance, allowing calls to be chained together, for example:
///     mutateCallInst(CI, NewFuncName)
///       .removeArg(3)
///       .appendArg(translateScope());
///
/// Only when the destuctor of this object is called will the original CallInst
/// be destroyed and replaced with the new CallInst be created.
class BuiltinCallMutator {
  // Original call instruction
  llvm::CallInst *CI;
  // New unmangled function name
  std::string FuncName;
  // Return type mutator. This needs to be saved, because we can't call it until
  // the new instruction is created.
  std::function<llvm::Value *(llvm::IRBuilder<> &, llvm::CallInst *)> MutateRet;
  typedef decltype(MutateRet) MutateRetFuncTy;
  // The attribute list for the new called function.
  llvm::AttributeList Attrs;
  // The attribute list for the new call instruction.
  llvm::AttributeList CallAttrs;
  // The return type for the new call instruction.
  llvm::Type *ReturnTy;
  // The arguments for the new call instruction.
  llvm::SmallVector<llvm::Value *, 8> Args;
  // The pointer element types for the new call instruction.
  llvm::SmallVector<llvm::Type *, 8> PointerTypes;
  // The mangler rules to use for the new call instruction.
  ManglingRules Rules;

  friend class BuiltinCallHelper;
  BuiltinCallMutator(
      llvm::CallInst *CI, std::string FuncName, ManglingRules Rules,
      std::function<std::string(llvm::StringRef)> NameMapFn = nullptr);

  // This does the actual work of creating of the new call, and will return the
  // new instruction.
  llvm::Value *doConversion();

public:
  ~BuiltinCallMutator() {
    if (CI)
      doConversion();
  }
  BuiltinCallMutator(const BuiltinCallMutator &) = delete;
  BuiltinCallMutator &operator=(const BuiltinCallMutator &) = delete;
  BuiltinCallMutator &operator=(BuiltinCallMutator &&) = delete;
  BuiltinCallMutator(BuiltinCallMutator &&);

  /// The builder used to generate IR for this call.
  llvm::IRBuilder<> Builder;

  /// Return the resulting new instruction. It is not possible to use any
  /// method on this object after calling this function.
  llvm::Value *getMutated() { return doConversion(); }

  /// Return the number of arguments currently specified for the new call.
  unsigned arg_size() const { return Args.size(); }

  /// Get the corresponding argument for the new call.
  llvm::Value *getArg(unsigned Index) const { return Args[Index]; }

  llvm::Type *getType(unsigned Index) const { return PointerTypes[Index]; }

  /// Return the pointer element type of the corresponding index, or nullptr if
  /// it is not a pointer.
  llvm::Type *getPointerElementType(unsigned Index) const {
    if (auto *TPT = llvm::dyn_cast<llvm::TypedPointerType>(PointerTypes[Index]))
      return TPT->getElementType();
    return nullptr;
  }

  /// A pair representing both the LLVM value of an argument and its
  /// corresponding pointer element type. This type can be constructed from
  /// implicit conversion from an LLVM value object (but only if it is not of
  /// pointer type), or by the appropriate std::pair type.
  struct ValueTypePair : public std::pair<llvm::Value *, llvm::Type *> {
    ValueTypePair(llvm::Value *V) : pair(V, V->getType()) {
      assert(!V->getType()->isPointerTy() &&
             "Must specify a pointer element type if value is a pointer.");
    }
    ValueTypePair(std::pair<llvm::Value *, llvm::Type *> P) : pair(P) {}
    ValueTypePair(llvm::Value *V, llvm::Type *T) : pair(V, T) {}
    ValueTypePair() = delete;
    using pair::pair;
  };

  /// Use the following arguments as the arguments of the new call, replacing
  /// any previous arguments. This version may not be used if any argument is of
  /// pointer type.
  BuiltinCallMutator &setArgs(llvm::ArrayRef<llvm::Value *> Args);

  /// This will replace the return type of the call with a different return
  /// type. The second argument is a function that will be called with an
  /// IRBuilder parameter and the newly generated function, and will return the
  /// value to replace all uses of the original call instruction with. Example
  /// usage:
  ///
  ///     BuiltinCallMutator Mutator = /* ... */;
  ///     Mutator.changeReturnType(Int16Ty, [](IRBuilder<> &IRB, CallInst *CI) {
  ///       return IRB.CreateZExt(CI, Int16Ty);
  ///     });
  BuiltinCallMutator &changeReturnType(llvm::Type *ReturnTy,
                                       MutateRetFuncTy MutateFunc);

  /// Insert an argument before the given index.
  BuiltinCallMutator &insertArg(unsigned Index, ValueTypePair Arg);

  /// Add an argument to the end of the argument list.
  BuiltinCallMutator &appendArg(ValueTypePair Arg) {
    return insertArg(Args.size(), Arg);
  }

  /// Replace the argument at the given index with a new value.
  BuiltinCallMutator &replaceArg(unsigned Index, ValueTypePair Arg);

  /// Remove the argument at the given index.
  BuiltinCallMutator &removeArg(unsigned Index);

  /// Remove all arguments in a range.
  BuiltinCallMutator &removeArgs(unsigned Start, unsigned Len) {
    for (unsigned I = 0; I < Len; I++)
      removeArg(Start);
    return *this;
  }

  /// Move the argument from the given index to the new index.
  BuiltinCallMutator &moveArg(unsigned FromIndex, unsigned ToIndex) {
    if (FromIndex == ToIndex)
      return *this;
    ValueTypePair Pair(Args[FromIndex], getType(FromIndex));
    removeArg(FromIndex);
    insertArg(ToIndex, Pair);
    return *this;
  }

  /// Use a callback function or lambda to convert an argument to a new value.
  /// The expected return type of the lambda is anything that is convertible
  /// to ValueTypePair, which could be a single Value* (but only if it is not
  /// pointer-typed), or a std::pair<Value *, Type *>. The possible signatures
  /// of the function parameter are as follows:
  ///     ValueTypePair func(IRBuilder<> &Builder, Value *, Type *);
  ///     ValueTypePair func(IRBuilder<> &Builder, Value *);
  ///     ValueTypePair func(Value *, Type *);
  ///     ValueTypePair func(Value *);
  ///
  /// When present, the IRBuilder parameter corresponds to a builder that is set
  /// to insert immediately before the new call instruction. The Value parameter
  /// corresponds to the argument to be mutated. The Type parameter, when
  /// present, will be either a TypedPointerType representing the "true" type of
  /// the value, or the argument's type otherwise.
  template <typename FnType>
  BuiltinCallMutator &mapArg(unsigned Index, FnType Func) {
    using namespace llvm;
    using std::is_invocable;
    IRBuilder<> Builder(CI);
    Value *V = Args[Index];
    [[maybe_unused]] Type *T = getType(Index);

    // Dispatch the function call as appropriate, based on the types that the
    // function may be called with.
    if constexpr (is_invocable<FnType, IRBuilder<> &, Value *, Type *>::value)
      replaceArg(Index, Func(Builder, V, T));
    else if constexpr (is_invocable<FnType, IRBuilder<> &, Value *>::value)
      replaceArg(Index, Func(Builder, V));
    else if constexpr (is_invocable<FnType, Value *, Type *>::value)
      replaceArg(Index, Func(V, T));
    else if constexpr (is_invocable<FnType, Value *>::value)
      replaceArg(Index, Func(V));
    else {
      // We need a helper value that is always false, but is dependent on the
      // template parameter to prevent this static_assert from firing when one
      // of the if constexprs above fires.
      static_assert(detail::LegalFnType<FnType>,
                    "mapArg lambda signature is not satisfied");
    }
    return *this;
  }

  /// Map all arguments according to the given function, as if mapArg(i, Func)
  /// had been called for every argument i.
  template <typename FnType> BuiltinCallMutator &mapArgs(FnType Func) {
    for (unsigned I = 0, E = Args.size(); I < E; I++)
      mapArg(I, Func);
    return *this;
  }
};

/// A helper class for generating calls to SPIR-V builtins with appropriate name
/// mangling rules. It is expected that transformation passes inherit from this
/// class.
class BuiltinCallHelper {
  ManglingRules Rules;
  std::function<std::string(llvm::StringRef)> NameMapFn;

protected:
  llvm::Module *M = nullptr;
  bool UseTargetTypes = false;

public:
  /// Initialize details about how to mangle and demangle builtins correctly.
  /// The Rules argument selects which name mangler to use for mangling.
  /// The NameMapFn function will map type names during demangling; it defaults
  /// to the identity function.
  explicit BuiltinCallHelper(
      ManglingRules Rules,
      std::function<std::string(llvm::StringRef)> NameMapFn = nullptr)
      : Rules(Rules), NameMapFn(std::move(NameMapFn)) {}

  /// Initialize the module that will be operated on. This method must be called
  /// before future methods.
  void initialize(llvm::Module &M);

  /// Return a mutator that will replace the given call instruction with a call
  /// to the given function name. The function name will have its name mangled
  /// in accordance with the argument types provided to the mutator.
  BuiltinCallMutator mutateCallInst(llvm::CallInst *CI, std::string FuncName);

  /// Return a mutator that will replace the given call instruction with a call
  /// to the given SPIR-V opcode (whose name is used in the lookup map of
  /// getSPIRVFuncName).
  BuiltinCallMutator mutateCallInst(llvm::CallInst *CI, spv::Op Opcode);

  /// Create a call to a SPIR-V builtin function (specified via opcode).
  /// The return type and argument types may be TypedPointerType, if the actual
  /// LLVM type is a pointer type.
  llvm::Value *addSPIRVCall(llvm::IRBuilder<> &Builder, spv::Op Opcode,
                            llvm::Type *ReturnTy,
                            llvm::ArrayRef<llvm::Value *> Args,
                            llvm::ArrayRef<llvm::Type *> ArgTys,
                            const llvm::Twine &Name = "");

  /// Create a call to a SPIR-V builtin function, returning a value and type
  /// pair suitable for use in BuiltinCallMutator::replaceArg and similar
  /// functions.
  BuiltinCallMutator::ValueTypePair
  addSPIRVCallPair(llvm::IRBuilder<> &Builder, spv::Op Opcode,
                   llvm::Type *ReturnTy, llvm::ArrayRef<llvm::Value *> Args,
                   llvm::ArrayRef<llvm::Type *> ArgTys,
                   const llvm::Twine &Name = "") {
    llvm::Value *V =
        addSPIRVCall(Builder, Opcode, ReturnTy, Args, ArgTys, Name);
    return BuiltinCallMutator::ValueTypePair(V, ReturnTy);
  }

  /// Adapt the various SPIR-V image types, for example changing a "spirv.Image"
  /// type into a "spirv.SampledImage" type with identical parameters.
  ///
  /// The input type is expected to be a TypedPointerType to either a
  /// "spirv.*" or "opencl.*" struct type. In the case of "opencl.*" struct
  /// types, it will first convert it into the corresponding "spirv.Image"
  /// struct type.
  ///
  /// If the image type does not match OldImageKind, this method will abort.
  llvm::Type *adjustImageType(llvm::Type *T, llvm::StringRef OldImageKind,
                              llvm::StringRef NewImageKind);

  /// Create a new type representing a SPIR-V opaque type that takes no
  /// parameters (such as sampler types).
  ///
  /// If UseRealType is false, a typed pointer type may be returned; if it is
  /// true, a pointer type will be used instead.
  llvm::Type *getSPIRVType(spv::Op TypeOpcode, bool UseRealType = false);

  /// Create a new type representing a SPIR-V opaque type that takes only an
  /// access qualifier (such as pipe types).
  ///
  /// If UseRealType is false, a typed pointer type may be returned; if it is
  /// true, a pointer type will be used instead.
  llvm::Type *getSPIRVType(spv::Op TypeOpcode, spv::AccessQualifier Access,
                           bool UseRealType = false);

  /// Create a new type representing a SPIR-V opaque type that is an image type
  /// of some kind.
  ///
  /// If UseRealType is false, a typed pointer type may be returned; if it is
  /// true, a pointer type will be used instead.
  llvm::Type *getSPIRVType(spv::Op TypeOpcode, llvm::Type *InnerType,
                           SPIRVTypeImageDescriptor Desc,
                           std::optional<spv::AccessQualifier> Access,
                           bool UseRealType = false);

  /// Create a new type representing a SPIR-V opaque type that takes arbitrary
  /// parameters.
  ///
  /// If UseRealType is false, a typed pointer type may be returned; if it is
  /// true, a pointer type will be used instead.
  llvm::Type *getSPIRVType(spv::Op TypeOpcode, llvm::StringRef InnerTypeName,
                           llvm::ArrayRef<unsigned> Parameters,
                           bool UseRealType = false);

private:
  llvm::SmallVector<llvm::Type *, 4> CachedParameterTypes;
  llvm::Function *CachedFunc = nullptr;

public:
  BuiltinCallMutator::ValueTypePair getCallValue(llvm::CallInst *CI,
                                                 unsigned ArgNo);

  llvm::Type *getCallValueType(llvm::CallInst *CI, unsigned ArgNo) {
    return getCallValue(CI, ArgNo).second;
  }
};

} // namespace SPIRV

#endif // SPIRVBUILTINHELPER_H
