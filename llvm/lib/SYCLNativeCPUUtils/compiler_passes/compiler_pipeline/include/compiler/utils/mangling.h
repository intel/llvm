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
/// @brief Simple function mangling framework.

#ifndef COMPILER_UTILS_MANGLING_H_INCLUDED
#define COMPILER_UTILS_MANGLING_H_INCLUDED

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <optional>

namespace llvm {
class LLVMContext;
class Type;
class raw_ostream;
}  // namespace llvm

namespace compiler {
namespace utils {
/// @brief Describes type qualifiers, which are aspects that need to be taken
/// into account when mangling function names. Such aspects are not represented
/// in the LLVM type. This is why such qualifiers need to be used along types.
enum TypeQualifier : int32_t {
  /// @brief The type has no special qualifier.
  eTypeQualNone = 0,
  /// @brief The type is a signed integer.
  eTypeQualSignedInt = 1,
  /// @brief The type is a constant pointer.
  eTypeQualPointerConst = 2,
  /// @brief The type is a volatile pointer.
  eTypeQualPointerVolatile = 4,
  /// @brief The type is a restrict pointer.
  eTypeQualPointerRestrict = 8
};

/// @brief Contains a small hierarchical list of TypeQualifier.
///
/// This hierarchy maps to derived types such as pointers or vectors:
/// * First qualifier for the pointer type.
/// * Second qualifier for the pointed-to type.
class TypeQualifiers final {
  using StorageT = uint64_t;

 public:
  /// @brief Create a type qualifier list with no qualifiers.
  TypeQualifiers();
  /// @brief Create a type qualifier list with one qualifiers.
  ///
  /// @param[in] Qual First qualifier.
  TypeQualifiers(TypeQualifier Qual);
  /// @brief Create a type qualifier list with two qualifiers.
  ///
  /// @param[in] Qual1 First qualifier.
  /// @param[in] Qual2 Second qualifier.
  TypeQualifiers(TypeQualifier Qual1, TypeQualifier Qual2);

  /// @brief Create a type qualifier list with one qualifiers.
  /// @note Convenience function that allows bit manipulation of qualifiers.
  ///
  /// @param[in] Qual First qualifier.
  TypeQualifiers(unsigned Qual);
  /// @brief Create a type qualifier list with two qualifiers.
  /// @note Convenience function that allows bit manipulation of qualifiers.
  ///
  /// @param[in] Qual1 First qualifier.
  /// @param[in] Qual2 Second qualifier.
  TypeQualifiers(unsigned Qual1, unsigned Qual2);

  /// @brief Number of type qualifiers contained in the list.
  StorageT getCount() const;

  /// @brief Top-most qualifier from the list.
  TypeQualifier front() const;

  /// @brief Remove the top-most qualifier from the list and returns it.
  TypeQualifier pop_front();

  /// @brief Return the qualifier at the given index.
  TypeQualifier at(unsigned Idx) const;

  /// @brief Add a qualifier to the list, making it bottom-most.
  ///
  /// @param[in] Qual Qualifier to add to the list.
  ///
  /// @return true if there was enough space to add the qualifier, or false.
  bool push_back(TypeQualifier Qual);
  /// @brief Add a qualifier to the list, making it bottom-most.
  /// @note Convenience function that allows bit manipulation of qualifiers.
  ///
  /// @param[in] Qual Qualifier to add to the list.
  ///
  /// @return true if there was enough space to add the qualifier, or false.
  bool push_back(unsigned Qual);
  /// @brief Add qualifiers to the end of the list.
  ///
  /// @param[in] Quals Qualifiers to add to the list.
  ///
  /// @return true if there was enough space to add the qualifiers, or false.
  bool push_back(TypeQualifiers Quals);

  /// @brief Determine whether two qualifier lists are equal.
  bool operator==(const TypeQualifiers &other) {
    return storage_ == other.storage_;
  }

  /// @brief Determine whether two qualifier lists are different.
  bool operator!=(const TypeQualifiers &other) { return !(*this == other); }

 private:
  /// @brief Set the number of type qualifiers contained in the list.
  void setCount(StorageT newCount);

  /// @brief Bits that make up the list. Deliberately small to pass by value.
  StorageT storage_;

  /// @brief Number of bits used to encode the size of the list.
  static const unsigned NumCountBits = 4;

  /// @brief Number of bits used to encode one qualifier in the list.
  static const unsigned NumQualBits = 10;

  /// @brief Number of bits that can be used to store the list.
  static const unsigned NumStorageBits = sizeof(StorageT) * 8;

  /// @brief Maximum size of the list.
  static const unsigned MaxSize = (NumStorageBits - NumCountBits) / NumQualBits;

  static_assert(MaxSize < (1 << NumCountBits) - 1, "MaxSize cannot be encoded");
};

/// @brief Helps with light parsing such as demangling function names.
class Lexer final {
 public:
  /// @brief Create a new lexer with the given text.
  ///
  /// @param[in] text Text to lex.
  Lexer(llvm::StringRef text);

  /// @brief Number of characters left to lex.
  unsigned Left() const;
  /// @brief Current lexing position in the text.
  unsigned CurrentPos() const;
  /// @brief String containing the text remaining to be lexed.
  llvm::StringRef TextLeft() const;
  /// @brief Current character.
  /// @return Character or negative value if no text is left.
  int Current() const;

  /// @brief Consume one character, advancing to the next character in the
  /// string.
  /// @return true if a character was consumed, false if no text left.
  bool Consume();
  /// @brief Consume several characters, advancing through the string.
  ///
  /// @param[in] Size Number of characters to consume.
  ///
  /// @return true if Size characters were consumed, false otherwise.
  bool Consume(unsigned Size);
  /// @brief Consume a string, and skip past it.
  ///
  /// @param[in] Pattern String to consume.
  ///
  /// @return true if Pattern was found and consumed, false otherwise.
  bool Consume(llvm::StringRef Pattern);
  /// @brief Consume an unsigned integer, and skip past it.
  ///
  /// @param[out] Result Consumed unsigned integer.
  ///
  /// @return true if an unsigned integer was consumed, false otherwise.
  bool ConsumeInteger(unsigned &Result);
  /// @brief Consume a signed integer, and skip past it.
  ///
  /// @param[out] Result Consumed signed integer.
  ///
  /// @return true if a signed integer was consumed, false otherwise.
  bool ConsumeSignedInteger(int &Result);
  /// @brief Consume consecutive alphabetic characters and skip past them.
  ///
  /// @param[out] Result Consumed string.
  ///
  /// @return true if an alphabetic string was consumed, false otherwise.
  bool ConsumeAlpha(llvm::StringRef &Result);
  /// @brief Consume consecutive alphanumeric characters and skip past them.
  ///
  /// @param[out] Result Consumed string.
  ///
  /// @return true if an alphanumeric string was consumed, false otherwise.
  bool ConsumeAlphanumeric(llvm::StringRef &Result);
  /// @brief Consume all characters until C is found. C is not consumed.
  ///
  /// @param[in] C Delimiter character.
  /// @param[out] Result Consumed string.
  ///
  /// @return true if C was found, false otherwise.
  bool ConsumeUntil(char C, llvm::StringRef &Result);
  /// @brief Consume all whitespace characters
  ///
  /// @return true if any whitespace was consumed or false otherwise
  bool ConsumeWhitespace();

 private:
  /// @brief Text to lex.
  llvm::StringRef Text;
  /// @brief Current lexing position into the text.
  unsigned Pos;
};

/// @brief Converts between mangled and non-mangled function names.
class NameMangler final {
 public:
  /// @brief Create a new name mangler.
  ///
  /// @param[in] context LLVM context to use.
  NameMangler(llvm::LLVMContext *context);

  /// @brief Determine the mangled name of a function.
  ///
  /// @param[in] Name Non-mangled name of the function.
  /// @param[in] Tys List of types, one for each function argument.
  /// @param[in] Quals Qualifiers, one for each type in Tys..
  ///
  /// @return The mangled name of the function.
  std::string mangleName(llvm::StringRef Name, llvm::ArrayRef<llvm::Type *> Tys,
                         llvm::ArrayRef<TypeQualifiers> Quals);

  /// @brief Try to mangle the given qualified type.
  ///
  /// @param[in] O Output stream to write the mangled name to.
  /// @param[in] Type Type to mangle.
  /// @param[in] Quals Type qualifiers.
  ///
  /// @return true if the type name could be mangled.
  bool mangleType(llvm::raw_ostream &O, llvm::Type *Type, TypeQualifiers Quals);

  /// @brief Try to mangle the given qualified type, taking substitutions into
  /// account.
  ///
  /// @param[in] O Output stream to write the mangled name to.
  /// @param[in] Type Type to mangle.
  /// @param[in] Quals Type qualifiers.
  /// @param[in] PrevTys Previously mangled types.
  /// @param[in] PrevQuals Qualifiers for previously mangled types.
  ///
  /// @return true if the type name could be mangled.
  bool mangleType(llvm::raw_ostream &O, llvm::Type *Type, TypeQualifiers Quals,
                  llvm::ArrayRef<llvm::Type *> PrevTys,
                  llvm::ArrayRef<TypeQualifiers> PrevQuals);

  /// @brief Remove the mangling of a function name, retrieving argument types
  ///        and qualifiers in the process.
  ///
  /// @param[in] Name Mangled function name to demangle.
  /// @param[out] Types Vector that will receive LLVM types for the arguments.
  /// @param[out] Quals Vector that will receive type qualifiers for the
  /// arguments.
  ///
  /// @return Demangled name or an empty string on failure
  llvm::StringRef demangleName(llvm::StringRef Name,
                               llvm::SmallVectorImpl<llvm::Type *> &Types,
                               llvm::SmallVectorImpl<TypeQualifiers> &Quals);

  /// @brief Remove the mangling of a function name, retrieving argument types
  ///        and qualifiers in the process.
  ///
  /// @param[in] Name Mangled function name to demangle.
  /// @param[out] Types Vector that will receive LLVM types for the arguments.
  /// @param[out] PointerElementTypes Vector that will receive LLVM types for
  /// the *first level* of pointer element types.
  /// @param[out] Quals Vector that will receive type qualifiers for the
  /// arguments.
  ///
  /// For example:
  ///   _Z3fooPii
  ///     Types[0]               = PointerType
  ///     PointerElementTypes[0] = i32
  ///     Quals[0] = (PointerQual, SignedIntQual)
  ///
  ///     Types[1] = i32
  ///     PointerElementTypes[1] = nullptr
  ///     Quals[1] = (SignedIntQual)
  ///
  /// @return Demangled name or an empty string on failure
  llvm::StringRef demangleName(
      llvm::StringRef Name, llvm::SmallVectorImpl<llvm::Type *> &Types,
      llvm::SmallVectorImpl<llvm::Type *> &PointerElementTypes,
      llvm::SmallVectorImpl<TypeQualifiers> &Quals);

  /// @brief Remove the mangling of a function name.
  ///
  /// @param[in] Name Mangled function name to demangle.
  ///
  /// @return Demangled name or original name if not mangled.
  llvm::StringRef demangleName(llvm::StringRef Name);

 private:
  /// @brief Try to mangle the given qualified type. This only works for simple
  /// types that do not require string manipulation.
  ///
  /// @param[in] Ty Type to mangle.
  /// @param[in] Qual Type qualifier.
  ///
  /// @return Mangled name of the type or nullptr.
  const char *mangleSimpleType(llvm::Type *Ty, TypeQualifier Qual);
  /// @brief Try to mangle the given builtin type name. This only works for
  /// 'spirv' target extension types (LLVM 17+).
  ///
  /// @param[in] Ty type to mangle.
  ///
  /// @return string if builtin type could be mangled otherwise empty string.
  std::optional<std::string> mangleBuiltinType(llvm::Type *Ty);
  /// @brief Try to demangle the given type name. This only works for simple
  /// types that do not require string manipulation.
  ///
  /// @param[in,out] L Lexer for the mangled type name.
  /// @param[out] Ty Demangled type.
  /// @param[out] Qual Demangled type qualifier.
  ///
  /// @return true if the type name could be demangled.
  bool demangleSimpleType(Lexer &L, llvm::Type *&Ty, TypeQualifier &Qual);
  /// @brief Try to demangle the given type name. This only works for opencl
  /// builtin types.
  ///
  /// @param[in,out] L Lexer for the mangled type name.
  /// @param[out] Ty Demangled type.
  ///
  /// @return true if the type name could be demangled.
  bool demangleOpenCLBuiltinType(Lexer &L, llvm::Type *&Ty);
  /// @brief Try to demangle the given type.
  ///
  /// @param[in] L Lexer currently pointing at a type.
  /// @param[out] Ty Demangled type.
  /// @param[out] PointerEltTy If null, unchanged. Else, set to the demangled
  /// pointer element type, if Ty is a non-opaque pointer type. Else set to
  /// nulltpr.
  /// @param[out] Quals Demangled type qualifiers.
  /// @param[in] CtxTypes Previously demangled types, used for substitutions.
  /// @param[in] CtxQuals Previously demangled qualifiers.
  ///
  /// @return true if the type could be demangled, false otherwise.
  bool demangleType(Lexer &L, llvm::Type *&Ty, llvm::Type **PointerEltTy,
                    TypeQualifiers &Quals,
                    llvm::SmallVectorImpl<llvm::Type *> &CtxTypes,
                    llvm::SmallVectorImpl<TypeQualifiers> &CtxQuals);

  /// @brief Demangle a name.
  ///
  /// @param[in] L Lexer currently pointing at a mangled name.
  ///
  /// @return Demangled name or an empty string.
  llvm::StringRef demangleName(Lexer &L);
  /// @brief Determine the type 'index' the substitution refers to.
  ///
  /// @param[in] SubID Substitution ID.
  /// @param[in] Tys List of types.
  /// @param[in] Quals Qualifiers for the types.
  ///
  /// @return Resolved type index or negative value.
  int resolveSubstitution(unsigned SubID,
                          llvm::SmallVectorImpl<llvm::Type *> &Tys,
                          llvm::SmallVectorImpl<TypeQualifiers> &Quals);
  /// @brief Try to emit a substituion for the given type instead of mangling
  /// it.
  ///
  /// @param[in,out] O Stream to write the substitution to.
  /// @param[in] Ty Type to mangle
  /// @param[in] Quals Qualifiers for the type.
  /// @param[in] PrevTys Types that have previously been mangled.
  /// @param[in] PrevQuals Qualifiers for the previously mangled types.
  ///
  /// @return true if a substitution was emitted, false otherwise.
  bool emitSubstitution(llvm::raw_ostream &O, llvm::Type *Ty,
                        TypeQualifiers Quals,
                        llvm::ArrayRef<llvm::Type *> PrevTys,
                        llvm::ArrayRef<TypeQualifiers> PrevQuals);
  /// @brief Determine whether the type is a builtin type or not. Builtin types
  /// are not considered for substitutions.
  ///
  /// @param[in] Ty Type to analyze.
  /// @param[in] Quals Type qualifiers.
  ///
  /// @return true if the type is a builtin type, or false.
  bool isTypeBuiltin(llvm::Type *Ty, TypeQualifiers &Quals);

  /// @brief LLVM context used to access LLVM types.
  llvm::LLVMContext *Context;
};
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_MANGLING_H_INCLUDED
