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
/// @brief Internal Vecz Choices header.

#ifndef VECZ_VECZ_CHOICES_H_INCLUDED
#define VECZ_VECZ_CHOICES_H_INCLUDED

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringRef.h>

// Forward declaration
namespace llvm {
class StringRef;
class Twine;
}  // namespace llvm

namespace vecz {

/// @brief Describes and holds various Vecz choices.
///
/// These choices can affect the code generated and are usually optimization
/// related. Since they are not always the best choice for a given target, they
/// are controlled at runtime by this class.
class VectorizationChoices {
 public:
  VectorizationChoices();
  ~VectorizationChoices() = default;

  /// @brief Enumeration with the available choices for Vecz.
  ///
  /// These are choices that can affect the code generated, often for
  /// optimization reasons. The Choices are prefixed by a `e<Category>` prefix,
  /// where `<Category>` is an arbitrary string to help document the intention
  /// of the Choice. For example, optimizations are prefixed with
  /// `eOptimization`.
  ///
  /// @note Each Choice has to be uniquely named without taking into account
  /// it's prefix, i.e. there shouldn't be any Choices sharing the same name
  /// but with different prefixes. Also, Choices names must not start with
  /// `"no"`, although different capitalizations (e.g. `"No"`) are allowed.
  /// Additionally, Choices' names should contain only alphanumeric characters.
  /// These restrictions are in place to allow for a `Choices` string to be
  /// parsable easily. See, for example, `parseChoicesString` . If you add a
  /// new Choice here, please also update the parseChoicesString function, as
  /// well as the two relevant `cl::opt` in `vectorizer.cpp`.
  enum Choice {
    /// @brief An invalid Choice ID, useful for error checking etc. Equals 0.
    eInvalid = 0,
    /// @brief Packetize uniform instructions instead of using a vector splat.
    ///
    /// When going through the packetization process, the default behaviour when
    /// encountering a uniform instruction is creating a vector splat
    /// with its value and stopping the packetization there. This option changes
    /// that behaviour, and instead makes the packetizer packetize even the
    /// uniform instructions, provided that they are used by a varying
    /// instruction.
    eOptimizationPacketizeUniform,
    /// @brief Packetize uniform instructions, but only in loops.
    ///
    /// This is similar to eOptimizationPacketizeUniform, with the difference
    /// that it only affects uniform values used inside loops.
    eOptimizationPacketizeUniformInLoops,
    /// @brief Emit loops for instantiated call instructions
    ///
    /// This will emit instantiated call instruction in loops instead of
    /// actually instantiating them. It only works when the call instruction has
    /// no users.
    eOptimizationInstantiateCallsInLoops,
    /// @brief Use the BOSCC linearization algorithm during Control-Flow
    //         Conversion.
    //
    //  @note This optimization retains uniform branches by duplicating pieces
    //  of the code.
    eLinearizeBOSCC,
    /// @brief Turn on full scalarization in the Scalarization pass
    //
    // This is useful for testing the scalarizer, and isn't intended to deliver
    // any performance benefits.
    eFullScalarization,
    /// @brief Treat division operations as being able to throw CPU exceptions
    ///
    /// @note This choice must be enabled for strict correctness on targets that
    /// support hardware exceptions on division by zero/division overflow, which
    /// require extra code to prevent traps on inactive vector lanes during
    /// linearization. However, any trapping behaviour of the input IR may be
    /// preserved (that is, on positively-executed code paths); it is left to
    /// the front end to conform to OpenCL spec in this regard.
    eDivisionExceptions,
    /// @brief Generate a vector-predicated kernel such that no work-items
    /// (vector elements) with side effects with IDs beyond the local workgroup
    /// size are enabled.
    ///
    /// @note The exact semantics concerning which operations are
    /// masked/unmasked are not defined. The guarantee is that the vectorized
    /// kernel will be safe to execute on workgroups with sizes smaller than
    /// the vector width. Some architectures may want to predicate beyond that
    /// remit for performance reasons, even if the vector-predicated operations
    /// are safe to execute on any input.
    eVectorPredication,
    /// @brief Force a default vectorization width, made without
    /// target-specific knowledge.
    ///
    /// @note This is most-commonly used in testing. Packetization may make
    /// decisions based on the target, which can make testing more difficult.
    /// This choice forces the default vector register width.
    eTargetIndependentPacketization,
  };

  /// @brief Check if a choice is enabled or not
  /// @param C The choice to check for
  /// @return true if the choice is enabled, false otherwise
  bool isEnabled(Choice C) const { return Enabled.count(C) > 0; }

  /// @brief Enable a choice
  /// @param C The choice to enable
  /// @return true if the choice was already enabled, false otherwise
  bool enable(Choice C) {
    auto res = Enabled.insert(C);
    return res.second;
  }

  /// @brief Disable a choice
  /// @param C The choice to disable
  /// @return true if the choice was enabled, false otherwise
  bool disable(Choice C) { return Enabled.erase(C); }

  /// @brief Parse a semicolon separated of Choices to enable or disable
  ///
  /// This functions accepts a string of Choices, separated by semicolon, and
  /// enables or disables them. The Choices are parsed according to the
  /// following rules:
  /// - The Choices are separated by a semicolon (';') character
  /// - Only one separator is allowed between each Choice.
  /// - Trailing separators are ignored (but only one is allowed still).
  /// - Choices are specified as they are in their enumerations, without the
  ///   "e<Category>" prefix.
  /// - Choices can be prefixed with the "no" prefix (without any whitespace),
  ///   which specifies that the Choice needs to be disabled instead of being
  ///   enabled.
  /// - The "no" prefix only applies to the Choice it is attached to and not to
  ///   any following Choices.
  /// - Whitespace between the Choices and the separators, as well as leading
  ///   and trailing whitespace at the beginning and end of the string, is
  ///   ignored.
  ///
  /// Examples:
  /// - "PacketizeUniform"
  /// - "PacketizeUniform;InstantiateCallsInLoops"
  /// - "PacketizeUniform ;   noInstantiateCallsInLoops \n"
  /// - " noPacketizeUniform;noInstantiateCallsInLoops; "
  ///
  /// @param[in] Str The string containing the Choices to enable/disable
  /// @return true on success, false if the parsing failed
  bool parseChoicesString(llvm::StringRef Str);

  /// @brief Convert a Choice name from a string to the matching Choice value
  ///
  /// The choices are matched without their e<Category> prefix.
  ///
  /// @param[in] Str The string with the Choice name
  /// @return The Choice name, or eInvalid in case of error
  static Choice fromString(llvm::StringRef Str);

  //
  // Specific getters and setters for the most commonly used choices
  //

  /// @brief Check if the eOptimizationPacketizeUniform choice is set
  /// @return true if the choice is set, false otherwise
  bool packetizeUniform() const {
    return isEnabled(eOptimizationPacketizeUniform);
  }
  /// @brief Enable the eOptimizationPacketizeUniform choice
  /// @return true if eOptimizationPacketizeUniform was already enabled
  bool enablePacketizeUniform() {
    return enable(eOptimizationPacketizeUniform);
  }
  /// @brief Disable the eOptimizationPacketizeUniform choice
  /// @return true if eOptimizationPacketizeUniform was enabled
  bool disablePacketizeUniform() {
    return disable(eOptimizationPacketizeUniform);
  }

  /// @brief Check if the eOptimizationPacketizeUniformInLoops choice is set
  /// @return true if the choice is set, false otherwise
  bool packetizeUniformInLoops() const {
    return isEnabled(eOptimizationPacketizeUniformInLoops);
  }
  /// @brief Enable the eOptimizationPacketizeUniformInLoops choice
  /// @return true if eOptimizationPacketizeUniformInLoops was already enabled
  bool enablePacketizeUniformInLoops() {
    return enable(eOptimizationPacketizeUniformInLoops);
  }
  /// @brief Disable the eOptimizationPacketizeUniformInLoops choice
  /// @return true if eOptimizationPacketizeUniformInLoops was enabled
  bool disablePacketizeUniformInLoops() {
    return disable(eOptimizationPacketizeUniformInLoops);
  }

  /// @brief Check if the eOptimizationInstantiateCallsInLoops choice is set
  /// @return true if the choice is set, false otherwise
  bool instantiateCallsInLoops() const {
    return isEnabled(eOptimizationInstantiateCallsInLoops);
  }
  /// @brief Enable the eOptimizationInstantiateCallsInLoops choice
  /// @return true if eOptimizationInstantiateCallsInLoops was already enabled
  bool enableInstantiateCallsInLoops() {
    return enable(eOptimizationInstantiateCallsInLoops);
  }
  /// @brief Disable the eOptimizationInstantiateCallsInLoops choice
  /// @return true if eOptimizationInstantiateCallsInLoops was enabled
  bool disableInstantiateCallsInLoops() {
    return disable(eOptimizationInstantiateCallsInLoops);
  }

  /// @brief Check if the eLinearizeBOSCC choice is set
  /// @return true if the choice is set, false otherwise
  bool linearizeBOSCC() const { return isEnabled(eLinearizeBOSCC); }
  /// @brief Enable the eLinearizeBOSCC choice
  /// @return true if eLinearizeBOSCC was already enabled
  bool enableLinearizeBOSCC() { return enable(eLinearizeBOSCC); }
  /// @brief Disable the eLinearizeBOSCC choice
  /// @return true if eLinearizeBOSCC was enabled
  bool disableLinearizeBOSCC() { return disable(eLinearizeBOSCC); }

  /// @brief Check if the eVectorPredication choice is set
  /// @return true if the choice is set, false otherwise
  bool vectorPredication() const { return isEnabled(eVectorPredication); }
  /// @brief Enable the eVectorPredication choice
  /// @return true if eVectorPredication was already enabled
  bool enableVectorPredication() { return enable(eVectorPredication); }
  /// @brief Disable the eVectorPredication choice
  /// @return true if eVectorPredication was enabled
  bool disableVectorPredication() { return disable(eVectorPredication); }

  /// @brief Check if the eTargetIndependentPacketization choice is set
  /// @return true if the choice is set, false otherwise
  bool targetIndependentPacketization() const {
    return isEnabled(eTargetIndependentPacketization);
  }
  /// @brief Enable the eTargetIndependentPacketization choice
  /// @return true if eTargetIndependentPacketization was already enabled
  bool enableTargetIndependentPacketization() {
    return enable(eTargetIndependentPacketization);
  }
  /// @brief Disable the eTargetIndependentPacketization choice
  /// @return true if eTargetIndependentPacketization was enabled
  bool disableTargetIndependentPacketization() {
    return disable(eTargetIndependentPacketization);
  }

  struct ChoiceInfo {
    llvm::StringLiteral name;
    Choice number;
    llvm::StringLiteral desc;
  };

  static llvm::ArrayRef<ChoiceInfo> queryAvailableChoices();

 private:
  /// @brief All the choices enabled
  llvm::SmallSet<Choice, 2> Enabled;

  /// @brief Print an error message, used by parseChoicesString
  ///
  /// The error message will contain the message given as well as the Choices
  /// string being parsed and the position that the error occured.
  //
  /// @param[in] Input The Choices string being parsed
  /// @param[in] Position The position where the parsin error occured
  /// @param[in] Msg The accompanying error message
  static void printChoicesParseError(llvm::StringRef Input, unsigned Position,
                                     llvm::Twine Msg);
};

}  // namespace vecz
#endif  // VECZ_VECZ_CHOICES_H_INCLUDED
