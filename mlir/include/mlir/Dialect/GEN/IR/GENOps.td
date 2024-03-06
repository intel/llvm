//===-- GENOps.td - GEN IR dialect op definition file ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the GEN IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_OPS
#define GEN_OPS

include "mlir/Dialect/GEN/IR/GENDialect.td"
include "mlir/Dialect/GEN/IR/GENAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// GEN op definitions
//===----------------------------------------------------------------------===//

class GEN_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<GEN_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Thread and Block index
//===----------------------------------------------------------------------===//

def IndexType : AnyTypeOf<[I32, I64]>;

def GEN_ThreadIdXOp : GEN_Op<"workitem.id.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_ThreadIdYOp : GEN_Op<"workitem.id.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_ThreadIdZOp : GEN_Op<"workitem.id.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

def GEN_BlockIdXOp : GEN_Op<"workgroup.id.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_BlockIdYOp : GEN_Op<"workgroup.id.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_BlockIdZOp : GEN_Op<"workgroup.id.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Block and Grid range
//===----------------------------------------------------------------------===//

def GEN_BlockDimXOp : GEN_Op<"workgroup.dim.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_BlockDimYOp : GEN_Op<"workgroup.dim.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_BlockDimZOp : GEN_Op<"workgroup.dim.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

def GEN_GridDimXOp : GEN_Op<"grid.dim.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_GridDimYOp : GEN_Op<"grid.dim.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def GEN_GridDimZOp : GEN_Op<"grid.dim.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def GEN_BarrierOp : GEN_Op<"barrier"> {
  let summary = "Workgroup barrier";

  string baseDescription = [{
    The `gen.barrier` operation performs a workgroup barrier and ensures all
    outstanding memory transaction using local or global memory are complete.
  }];

  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def IntegerOrFloatType : AnyTypeOf<[AnyInteger, AnyFloat]>;

def GEN_SubGroupShuffleOp : GEN_Op<"sub_group_shuffle", [
      TypesMatchWith<"result and value have the same type",
                     "res", "value", "$_self">]>,
  Results<(outs IntegerOrFloatType:$res)>,
  Arguments<(ins IntegerOrFloatType:$value,
                 I32:$mask,
                 GEN_ShflKindAttr:$kind)> {
  let summary = "Subgroup shuffle";
  string baseDescription = [{
    The `gen.sub_group_shuffle` operation is invoked by different work items
    with different values, given by $value. Different work items have different
    subgroup local IDs. The shuffle kind, $kind, is given to determine how to
    calculate the associated subgroup local ID. It returns the associated
    $value for the work item with subgroup local ID equal to:
    - $kind == xor, the current invocation’s subgroup local ID xor’ed with $mask.
    - $kind == up, the current invocation’s subgroup local ID - $mask.
    - $kind == down, the current invocation’s subgroup local ID + $mask.
    - $kind == idx, the subgroup local ID $mask.
  }];

  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($value) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

def GEN_FpToFpOp : GEN_Op<"conv.fptofp">,
  Results<(outs AnyFloat:$res)>,
  Arguments<(ins
    AnyFloat:$arg,
    OptionalAttr<GEN_RoundingModeAttr>:$roundingMode
  )> {
  let summary = "Convert between floating point types";

  string baseDescription = [{
    Convert `$arg` to the type of the result `$res`. If necessary, round the
    result according to `$roundingMode`.
  }];

  let assemblyFormat = [{
    $arg (` ` `{` `roundingMode` `=` $roundingMode^ `}`)? attr-dict `:` type($arg) `to` type($res)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

class FixedVectorOfRankAndType<list<int> allowedRanks,
                               list<Type> allowedTypes> : AllOfType<
  [FixedVectorOf<allowedTypes>, FixedVectorOfRank<allowedRanks>],
  FixedVectorOf<allowedTypes>.summary #
  FixedVectorOfRank<allowedRanks>.summary,
  "::mlir::VectorType">;

def GEN_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

def GEN_MatrixDPASOp : GEN_Op<"matrix.dpas">,
  Results<(outs FixedVectorOf<[GEN_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [GEN_MatrixElemType]>:$c,
    FixedVectorOfRankAndType<[1], [GEN_MatrixElemType]>:$a,
    FixedVectorOfRankAndType<[1], [GEN_MatrixElemType]>:$b,
    GEN_PrecisionTypeAttr:$pa,
    GEN_PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "GEN matrix multiply-add (for PVC)";

  string baseDescription = [{
    The 'gen.matrix.dpas' operation is a matrix multiply-add operation as follows:

       D = C + A x B

       where
          D : MxN
          C : MxN
          A : MxK
          B : KxN

          M : repeat count ($rc), must be 1, 2, 4, or 8
          N : fixed execution size, must be 16
          K : depth * OPS_PER_CHAN
              OPS_PER_CHAN
                 1 : for TF32
                 2 : for 16-bit precision(BF, HF)
                 4 : for 8-bit precision (FP8, UB, B)
                 8 : for less-then 8 bit precision (U4/S4, U2/S2).

              If depth is 8, K would be 8, 16, 32, or 64 (based on OPS_PER_CHAN).

    $a, $b, $c, $d - matrix A, B, C, D, respectively
    $pa, $pb - precision of matrix A and B resepectively
    $rc - repect count
  }];

  string llvmBuilder = [{
    $d = createGenISADPAS(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GEN_Matrix2DBlockLoadOp : GEN_Op<"matrix.2Dblockload">,
  Results<(outs FixedVectorOf<[GEN_MatrixElemType]>:$res)>,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform
  )> {

  let summary = "GEN 2D block load";

  string baseDescription = [{
    The 'gen.matrix.2Dblockload' operation loads a submatrix from an array in memory.
    $ptr - the base address of the memory array
    $base_width, $base_height, $base_pitch - the shape of the memory array
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of the submatrix to load
    $elem_size_in_bits - 32 for f32, bf32; 16 for f16, int16, bf16; 8 for int8, int4, int2 and etc
    $v_blocks - number of blocks to load
    $transpose - transpose the submatrix in vector register (useful for 32 bit element types)
    $vnni_transform - transpose and pack the submatrix in register (useful for < 32 bit element types)

    The $transpose and $vnni_transform are mutual exclusive. These transformations are used for
    the B matrix in MMA (DPAS) operations D = C + A * B, where A should have row-major layout in register
    and B should have column-major layout.

    If the submatrix contains out of bound elements of the memory array, they are filled with 0.
  }];

  string llvmBuilder = [{
    $res = createGenISA2DBlockRead(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GEN_Matrix2DBlockStoreOp : GEN_Op<"matrix.2Dblockstore">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    FixedVectorOf<[GEN_MatrixElemType]>:$stored_val
  )> {

  let summary = "GEN 2D block store";

  string baseDescription = [{
    The 'gen.matrix.2Dblockstore' operation stores to a submatrix from an array in memory.
    $ptr - the base address of the memory array
    $base_width, $base_height, $base_pitch - the shape of the memory array
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of the submatrix to load
    $elem_size_in_bits - 32 for f32, bf32; 16 for f16, int16, bf16; 8 for int8, int4, int2 and etc
    $v_blocks - number of blocks to store
    $transpose - transpose the submatrix in vector register (useful for 32 bit element types)
    $vnni_transform - transpose and pack the submatrix in register (useful for < 32 bit element types)
    $stored_val - block to store

    The $transpose and $vnni_transform are mutual exclusive. These transformations are used for
    the B matrix in MMA (DPAS) operations D = C + A * B, where A should have row-major layout in register
    and B should have column-major layout.

    If the submatrix contains out of bound elements of the memory array, they are filled with 0.
  }];

  string llvmBuilder = [{
    createGenISA2DBlockWrite(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

#endif // GEN_OPS
