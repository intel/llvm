//===-- GENDialect.td - GEN dialect op definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GEN_DIALECT
#define GEN_DIALECT

include "mlir/IR/OpBase.td"

def GEN_Dialect : Dialect {
  let name = "gen";
  let cppNamespace = "::mlir::GEN";
  let summary = "The GEN dialect.";

  let description = [{
    GEN is a dialect for representing operations on Intel GPUs.
  }];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate max work group size
    /// required for kernels.
    static StringRef getMaxWorkGroupSizeAttrName() {
      return "gen.max_work_group_size";
    }

    /// Get the name of the attribute used to annotate exact work group size
    /// required for kernels.
    static StringRef getReqdWorkGroupSizeAttrName() {
      return "gen.reqd_work_group_size";
    }

    /// Get the name for the attribute used to annotate the exact sub group
    /// size required for kernels.
    static StringRef getReqdSubGroupSizeAttrName() {
      return "gen.intel_reqd_sub_group_size";
    }
  }];
}

#endif // GEN_DIALECT
