//=== CompileTimePropertiesPass.h - SYCL Compile Time Props Pass-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which converts values of symbolic attributes
// (compile-time properties) to integer id-based ones to later map to LLVM IR
// metadata nodes. The header also contains a number of function templates to
// extract corresponding attributes and their values.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/PassManager.h"

#include <cassert>

namespace llvm {

class CompileTimePropertiesPass
    : public PassInfoMixin<CompileTimePropertiesPass> {
public:
  // Enriches the module with metadata that describes the found variables for
  // the SPIRV-LLVM Translator.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

namespace detail {

/// Converts the string into a boolean value. If the string is equal to "false"
/// we consider its value as /c false, /true otherwise.
///
/// @param Value [in] "boolean as string" value.
///
/// @returns \c false if the value of \c Value equals to "false", \c true
/// otherwise.
inline bool toBool(StringRef Value) { return !Value.equals("false"); }

} // namespace detail

/// Returns a boolean representation of an attribute's value.
///
/// Currently the function works for string attributes only
///
/// @param Attr [in] An attribute
/// @return \c true if the attribute is a string attribute and its value is not
/// equal to \c "false", \c false otherwise.
inline bool hasProperty(const Attribute &Attr) {
  return Attr.isStringAttribute() && detail::toBool(Attr.getValueAsString());
}

/// Returns the attribute @Attribute's value as an integer of type @Int.
///
/// Currently the function works for string attributes only
///
/// @param Attr [in] An attribute
/// @tparam Int [in] The deserved type of the result.
///
/// @returns \c the attribute's value as an @Int.
template <typename Int> Int getAttributeAsInteger(const Attribute &Attr) {
  assert(Attr.isStringAttribute() &&
         "The attribute Attr must be a string attribute");
  Int Value = 0;
  bool Error = Attr.getValueAsString().getAsInteger(10, Value);
  assert(!Error && "The attribute's value is not a number");
  (void)Error;
  return Value;
}

/// Checks whether the object @Object has the @AttributeName property.
/// The object has the property if the @AttributeName attribute is a string
/// attribute defined for the object and its value is not represented as
/// \c false.
///
/// @param Object        [in] Object that can have attributes.
/// @param AttributeName [in] Property name.
/// @tparam T            [in] Object's type, the type must be able to work with
///                           attributes.
///
/// @returns \c true if the object @Object has the @AttributeName property,
/// \c false otherwise.
template <typename T>
bool hasProperty(const T &Object, StringRef AttributeName) {
  return Object.hasAttribute(AttributeName) &&
         hasProperty(Object.getAttribute(AttributeName));
}

/// Returns the value of the string @AttributeName attribute of the object
/// @Object as an integer of type @Int.
///
/// @param Object        [in] Object that can have attributes.
/// @param AttributeName [in] Property name.
/// @tparam Int          [in] The deserved type of the result.
/// @tparam T            [in] Object's type, the type must be able to work with
///                           attributes.
///
/// @returns \c the attribute's value as an @Int.
template <typename Int, typename T>
Int getAttributeAsInteger(const T &Object, StringRef AttributeName) {
  assert(Object.hasAttribute(AttributeName) &&
         "The object Object must have the requested attribute");
  return getAttributeAsInteger<Int>(Object.getAttribute(AttributeName));
}

} // namespace llvm
