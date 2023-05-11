//===- llvm/IR/FPAccuracy.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Interfaces related to floating-point accuracy control.
///
//===----------------------------------------------------------------------===/

#ifndef LLVM_IR_FPACCURACY_H
#define LLVM_IR_FPACCURACY_H

namespace llvm {

class StringRef;
class Type;

namespace Intrinsic {
typedef unsigned ID;
}

namespace fp {

/// FP accuracy
///
/// Enumerates supported accuracy modes for fpbuiltin intrinisics. These
/// modes are used to lookup required accuracy in terms of ULP for known
/// math operations that are represented by the fpbuiltin intrinsics.
///
/// These values can also be used to set the default accuracy for an IRBuilder
/// object and the IRBuilder will automatically attach the corresponding
/// "fpbuiltin-max-error" attribute to any fpbuiltin intrinsics that are
/// created using the IRBuilder object.
///
enum class FPAccuracy { High, Medium, Low, SYCL, CUDA };

/// Returns the required accuracy, in terms of ULP, for an fpbuiltin intrinsic
/// given the intrinsic ID, the base type for the operation, and the required
/// accuracy level (as an enumerated identifier).
///
/// If the supplied intrinsic ID and type do not identify an operation for
/// which required accuracy is available, this function will not return a value.
StringRef getAccuracyForFPBuiltin(Intrinsic::ID, const Type *, FPAccuracy);

} // namespace fp

} // namespace llvm

#endif // LLVM_IR_FPACCURACY_H
