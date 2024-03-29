//===-- GENOpsInterfaces.td - GEN op interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the GEN op interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_OPS_INTERFACES
#define GEN_OPS_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GENOpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::GEN";
}

def GEN3DNDRangeInterface : GENOpInterface<"GEN3DNDRangeInterface"> {
  let description = [{
    Interface for operations representing 3D ND-range queries.
  }];
  let verify = [{
    return detail::verify3DNDRangeOpInterface($_op);
  }];
}

#endif // GEN_OPS_INTERFACES
