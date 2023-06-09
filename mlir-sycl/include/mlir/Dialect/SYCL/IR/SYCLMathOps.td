//===-- SYCLMathOps.td - SYCL-specific math operations -*- tablegen -*-----===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for SYCL-specific math operations.
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_MATH_OPS
#define SYCL_MATH_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"

////////////////////////////////////////////////////////////////////////////////
// BASE DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

// Generic floating-point types (§4.17.1).
// TODO: Add vectors and marrays.
def SYCLGenfloat : AnyTypeOf<[F32, F64]>;

// Base class for SYCL-specific math operations (§4.17.5), which have no side
// effects and can be applied to scalars, vectors and marrarys.
// TODO: Consider VectorUnrollOpInterface, ElementwiseMappable.
// TODO: Consider arith::FastMathFlags.
class SYCL_MathOp<string mnemonic, list<Trait> traits = []> :
    SYCL_Op<"math." # mnemonic, traits # [Pure]>;

class SYCL_FloatUnaryOp<string mnemonic, string synopsis = "",
                        list<Trait> traits = []> :
    SYCL_MathOp<mnemonic, traits # [SameOperandsAndResultType]> {
  let summary = synopsis # ".";
  let arguments = (ins SYCLGenfloat:$operand);
  let results = (outs SYCLGenfloat:$result);

  let assemblyFormat = [{ $operand attr-dict `:` type($result) }];
}

////////////////////////////////////////////////////////////////////////////////
// math.atan OPERATION
////////////////////////////////////////////////////////////////////////////////

def SYCLSqrtOp : SYCL_FloatUnaryOp<"sqrt", "Compute square root">;

#endif // SYCL_MATH_OPS
