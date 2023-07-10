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
// TODO: Add marrays.
def SYCL_Vgenfloat : DialectType<SYCL_Dialect,
    CPred<[{
      ::llvm::isa<::mlir::sycl::VecType>($_self) &&
      ::llvm::isa<::mlir::Float32Type, ::mlir::Float64Type,
                  ::mlir::sycl::HalfType>(
          ::llvm::cast<::mlir::sycl::VecType>($_self).getDataType())
    }]>,
    "a sycl::vec of float, double or sycl::half">;
def SYCL_Genfloat : AnyTypeOf<[F32, F64, SYCL_HalfType, SYCL_Vgenfloat]>;

// Trait to mark ops representing a SYCL math function (§4.17.5).
def SYCLMathFunc : SYCLOpTrait<"SYCLMathFunc">;

// Base class for SYCL-specific math operations, which have no side effects and
// can be applied to scalars and vectors.
// TODO: Support marrarys.
// TODO: Consider VectorUnrollOpInterface, ElementwiseMappable.
// TODO: Consider arith::FastMathFlags.
class SYCL_MathOp<string mnemonic, list<Trait> traits = []> :
    SYCL_Op<"math." # mnemonic, traits # [SYCLMathFunc, Pure]>;

class SYCL_FloatUnaryOp<string mnemonic, string synopsis = "",
                        list<Trait> traits = []> :
    SYCL_MathOp<mnemonic, traits # [AllTypesMatch<["x", "result"]>]> {
  let summary = synopsis # ".";
  let arguments = (ins SYCL_Genfloat:$x);
  let results = (outs SYCL_Genfloat:$result);
  let assemblyFormat = [{ $x attr-dict `:` type($result) }];
}

class SYCL_FloatBinaryOp<string mnemonic, string synopsis = "",
                         list<Trait> traits = []> :
    SYCL_MathOp<mnemonic, traits # [AllTypesMatch<["x", "y", "result"]>]> {
  let summary = synopsis # ".";
  let arguments = (ins SYCL_Genfloat:$x, SYCL_Genfloat:$y);
  let results = (outs SYCL_Genfloat:$result);
  let assemblyFormat = [{ $x `,` $y attr-dict `:` type($result) }];
}

class SYCL_FloatTernaryOp<string mnemonic, string synopsis = "",
                          list<Trait> traits = []> :
    SYCL_MathOp<mnemonic, traits # [AllTypesMatch<["a", "b", "c", "result"]>]> {
  let summary = synopsis # ".";
  let arguments = (ins SYCL_Genfloat:$a, SYCL_Genfloat:$b, SYCL_Genfloat:$c);
  let results = (outs SYCL_Genfloat:$result);
  let assemblyFormat = [{ $a `,` $b `,` $c attr-dict `:` type($result) }];
}

////////////////////////////////////////////////////////////////////////////////
// OPERATIONS
////////////////////////////////////////////////////////////////////////////////

def SYCLCeilOp : SYCL_FloatUnaryOp<"ceil",
  "Round to integral value using the round to positive infinity rounding mode">;
def SYCLCopySignOp : SYCL_FloatBinaryOp<"copysign",
  "Returns `x` with its sign changed to match the sign of `y`">;
def SYCLCosOp : SYCL_FloatUnaryOp<"cos",
  "Compute cosine">;
def SYCLExpOp : SYCL_FloatUnaryOp<"exp",
  "Compute the base-e exponential of `x`">;
def SYCLExp2Op : SYCL_FloatUnaryOp<"exp2",
  "Exponential base 2 function">;
def SYCLExpM1Op : SYCL_FloatUnaryOp<"expm1",
  "Compute `e^x - 1.0`">;
def SYCLFabsOp : SYCL_FloatUnaryOp<"fabs",
  "Compute absolute value of a floating-point number">;
def SYCLFloorOp : SYCL_FloatUnaryOp<"floor",
  "Round to integral value using the round to negative infinity rounding mode">;
def SYCLFmaOp : SYCL_FloatTernaryOp<"fma"> {
  let summary = "Computes fused multiply-add.";
  let description = [{
    Returns the correctly rounded floating-point representation of the sum of
    `c` with the infinitely precise product of `a` and `b`.
  }];
}
def SYCLLogOp : SYCL_FloatUnaryOp<"log",
  "Compute natural logarithm">;
def SYCLLog10Op : SYCL_FloatUnaryOp<"log10",
  "Compute a base 10 logarithm">;
def SYCLLog2Op : SYCL_FloatUnaryOp<"log2",
  "Compute a base 2 logarithm">;
def SYCLPowOp : SYCL_FloatBinaryOp<"pow",
  "Compute `x` to the power `y`">;
def SYCLRoundOp : SYCL_FloatUnaryOp<"round"> {
  let summary = "Round `x`.";
  let description = [{
    Return the integral value nearest to x rounding halfway cases away from
    zero, regardless of the current rounding direction.
  }];
}
def SYCLRsqrtOp : SYCL_FloatUnaryOp<"rsqrt",
  "Compute inverse square root">;
def SYCLSinOp : SYCL_FloatUnaryOp<"sin",
  "Compute sine">;
def SYCLSqrtOp : SYCL_FloatUnaryOp<"sqrt",
  "Compute square root">;
def SYCLTruncOp : SYCL_FloatUnaryOp<"trunc",
  "Round to integral value using the round to zero rounding mode">;

#endif // SYCL_MATH_OPS
