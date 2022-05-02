//===----- SpecConstants.h - SYCL Specialization Constants Pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which converts symbolic id-based specialization
// constant intrinsics to integer id-based ones to later map to SPIRV spec
// constant operations. The spec constant IDs are symbolic before linkage to
// make separate compilation possible. After linkage all spec constants are
// available to the pass, and it can assign consistent integer IDs.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include <vector>

namespace llvm {

class StringRef;

// Represents either an element of a composite specialization constant or a
// single scalar specialization constant - at SYCL RT level composite
// specialization constants are being represented as a single byte-array, while
// at SPIR-V level they are represented by a number of scalar specialization
// constants.
// The same representation is re-used for scalar specialization constants in
// order to unify they processing with composite ones.
struct SpecConstantDescriptor {
  // Encodes ID of a scalar specialization constants which is a leaf of some
  // composite specialization constant.
  unsigned ID;
  // Encodes offset from the beginning of composite, where scalar resides, i.e.
  // location of the scalar value within a byte-array containing the whole
  // composite specialization constant. If descriptor is used to represent a
  // whole scalar specialization constant instead of an element of a composite,
  // this field should be contain zero.
  unsigned Offset;
  // Encodes size of scalar specialization constant.
  unsigned Size;
};

using SpecIDMapTy = MapVector<StringRef, std::vector<SpecConstantDescriptor>>;

class SpecConstantsPass : public PassInfoMixin<SpecConstantsPass> {
public:
  // SetValAtRT parameter controls spec constant lowering mode:
  // - if true, it is lowered to SPIRV intrinsic which retrieves constant value
  // - if false, it is replaced with C++ default (used for AOT compilers)
  SpecConstantsPass(bool SetValAtRT = true) : SetValAtRT(SetValAtRT) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  // Searches given module for occurrences of specialization constant-specific
  // metadata and builds "spec constant name" -> vector<"spec constant int ID">
  // map
  static bool collectSpecConstantMetadata(const Module &M, SpecIDMapTy &IDMap);
  // Searches given module for occurrences of specialization constant-specific
  // metadata and builds vector of default values for every spec constant.
  static bool
  collectSpecConstantDefaultValuesMetadata(const Module &M,
                                           std::vector<char> &DefaultValues);

private:
  bool SetValAtRT;
};

} // namespace llvm
