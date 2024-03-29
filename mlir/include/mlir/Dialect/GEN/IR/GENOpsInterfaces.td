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
    // Check operation signature.
    static_assert(ConcreteOp::template hasTrait<OpTrait::OneResult>(),
                  "Expecting operation returning a single result");
    static_assert(ConcreteOp::template hasTrait<OpTrait::OneOperand>(),
                  "Expecting operation receiving a single operand");
    auto concreteOp = cast<ConcreteOp>($_op);
    static_assert(std::is_same_v<decltype(concreteOp.getRes().getType()),
                                      IndexType>,
                  "Expecting result index type");
    static_assert(std::is_same_v<decltype(concreteOp.getDim().getType()),
                                 IntegerType>,
                  "Expecting integer operand type");
    // Additional checks.
    return detail::verify3DNDRangeOpInterface($_op);
  }];
}

#endif // GEN_OPS_INTERFACES
