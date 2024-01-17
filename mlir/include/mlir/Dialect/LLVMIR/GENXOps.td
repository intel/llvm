//===-- GENXOps.td - GENX IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the GENX IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef GENXIR_OPS
#define GENXIR_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// GENX dialect definitions
//===----------------------------------------------------------------------===//

def GENX_Dialect : Dialect {
  let name = "genx";
  let summary = "The GENX dialect in MLIR.";
  let description = [{
    GENX is a dialect for representing operations on Intel GPUs and extensions
    to the LLVM type system.
  }];
  let cppNamespace = "::mlir::GENX";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel functions.
    static StringRef getKernelFuncAttrName() { return "genx.kernel"; }

    /// Get the name of the attribute used to annotate max work group size
    /// required for kernel functions.
    static constexpr ::llvm::StringLiteral getMaxWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.max_work_group_size");
    }

    /// Get the name of the attribute used to annotate exact work group size
    /// required for kernel functions.
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.reqd_work_group_size");
    }

    /// Get the name for the attribute used to annotate the exact sub group
    /// size required for kernel functions.
    static constexpr ::llvm::StringLiteral getReqdSubGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.intel_reqd_sub_group_size");
    }
  }];
}

//===----------------------------------------------------------------------===//
// GENX op definitions
//===----------------------------------------------------------------------===//

class GENX_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<GENX_Dialect, mnemonic, traits> {
}

class GENX_DeviceFunctionOp<string mnemonic, string fnName,
                            int arg, list<Trait> traits = []>
  : GENX_Op<mnemonic, !listconcat(traits, [Pure])>,
    Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = [{
    llvm::Type *retType = nullptr;
    if (std::string("}] # fnName # [{") == std::string("__builtin_IB_get_global_size")) {
      retType = builder.getInt32Ty();
    } else {
      retType = builder.getInt64Ty();
    }
    llvm::Type *argType = builder.getInt32Ty();
    llvm::Value *val = llvm::ConstantInt::get(argType, }] # arg # [{);
    llvm::CallInst *ci = createDeviceFunctionCall(builder, "}] # fnName # [{",
        retType, { argType }, { val });
    llvm::Type *opRetType = cast<llvm::IntegerType>(
        moduleTranslation.convertType(op->getResultTypes()[0]));
    if (opRetType == retType)
      $res = ci;
    else {
      llvm::Instruction::CastOps opcode =
          llvm::CastInst::getCastOpcode(ci, false, opRetType, false);
      $res = builder.CreateCast(opcode, ci, opRetType);
    }
  }];

  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Thread index and Block index

def GENX_ThreadIdXOp : GENX_DeviceFunctionOp<"workitem.id.x",
                        "_Z12get_local_idj", 0>;
def GENX_ThreadIdYOp : GENX_DeviceFunctionOp<"workitem.id.y",
                        "_Z12get_local_idj", 1>;
def GENX_ThreadIdZOp : GENX_DeviceFunctionOp<"workitem.id.z",
                        "_Z12get_local_idj", 2>;
def GENX_BlockIdXOp : GENX_DeviceFunctionOp<"workgroup.id.x",
                        "_Z12get_group_idj", 0>;
def GENX_BlockIdYOp : GENX_DeviceFunctionOp<"workgroup.id.y",
                        "_Z12get_group_idj", 1>;
def GENX_BlockIdZOp : GENX_DeviceFunctionOp<"workgroup.id.z",
                        "_Z12get_group_idj", 2>;

//===----------------------------------------------------------------------===//
// Block and Grid range

def GENX_BlockDimXOp : GENX_DeviceFunctionOp<"workgroup.dim.x",
                        "_Z12get_local_sizej", 0>;
def GENX_BlockDimYOp : GENX_DeviceFunctionOp<"workgroup.dim.y",
                        "_Z12get_local_sizej", 1>;
def GENX_BlockDimZOp : GENX_DeviceFunctionOp<"workgroup.dim.z",
                        "_Z12get_local_sizej", 2>;
// opencl function is currently not being parsed properly by IGC. 
// Once IGC fixes this issue, we can switch back to the mangled function name or use 
// a GenISA intrinsic. 
def GENX_GridDimXOp : GENX_DeviceFunctionOp<"grid.dim.x",
                        "__builtin_IB_get_global_size", 0>;
def GENX_GridDimYOp : GENX_DeviceFunctionOp<"grid.dim.y",
                        "__builtin_IB_get_global_size", 1>;
def GENX_GridDimZOp : GENX_DeviceFunctionOp<"grid.dim.z",
                        "__builtin_IB_get_global_size", 2>;

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def GENX_BarrierOp : GENX_Op<"barrier"> {
  let summary = "Workgroup barrier";

  string baseDescription = [{
    The `genx.barrier` operation performs a workgroup barrier and ensures all
    outstanding memory transaction using local or global memory are complete.
  }];

  string llvmBuilder = [{
    llvm::Type *retType = builder.getVoidTy();
    llvm::Type *argType = builder.getInt32Ty();
    int memFence = 3; // local + global memory fence
    llvm::Value *arg = llvm::ConstantInt::get(argType, memFence);
    createDeviceFunctionCall(builder, "_Z7barrierj", retType, {argType}, {arg},
                             true /*convergent*/);
  }];

  let assemblyFormat = "attr-dict";
}

/// Enum attribute of the different shuffle kinds.
def ShflKind : I32EnumAttr<"ShflKind", "GENX shuffle kind",
  [
    I32EnumAttrCase<"XOR",  0>,
    I32EnumAttrCase<"UP",   1>,
    I32EnumAttrCase<"DOWN", 2>,
    I32EnumAttrCase<"IDX",  3>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def ShflKindAttr : EnumAttr<GENX_Dialect, ShflKind, "shfl_kind">;

def IntegerOrFloatType : AnyTypeOf<[AnyInteger, AnyFloat]>;

def GENX_SubGroupShuffleOp : GENX_Op<"sub_group_shuffle", [
      TypesMatchWith<"result and value have the same type",
                     "res", "value", "$_self">]>,
  Results<(outs IntegerOrFloatType:$res)>,
  Arguments<(ins IntegerOrFloatType:$value,
                 I32:$mask,
                 ShflKindAttr:$kind)> {
  let summary = "Subgroup shuffle";
  string baseDescription = [{
    The `genx.sub_group_shuffle` operation is invoked by different work items
    with different values, given by $value. Different work items have different
    subgroup local IDs. The shuffle kind, $kind, is given to determine how to
    calculate the associated subgroup local ID. It returns the associated
    $value for the work item with subgroup local ID equal to:
    - $kind == XOR, the current invocation’s subgroup local ID xor’ed with $mask.
    - $kind == UP, the current invocation’s subgroup local ID - $mask.
    - $kind == DOWN, the current invocation’s subgroup local ID + $mask.
    - $kind == IDX, the subgroup local ID $mask.
  }];
  string llvmBuilder = [{
    $res = createSubGroupShuffle(builder, $value, $mask, $kind);
  }];
  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($value) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

/// Enum attribute of the different floating-point rounding modes.
def RoundingMode : I32EnumAttr<"RoundingMode", "GENX floating-point rounding mode",
  [
    I32EnumAttrCase<"UNUSED", 0>,
    I32EnumAttrCase<"RTE",    1>, ///< Round to nearest, ties to even
    I32EnumAttrCase<"RTN",    2>, ///< Round toward negative
    I32EnumAttrCase<"RTP",    3>, ///< Round toward positive
    I32EnumAttrCase<"RTZ",    4>, ///< Round toward zero
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def RoundingModeAttr : EnumAttr<GENX_Dialect, RoundingMode, "rounding_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def GENX_FpToFpOp : GENX_Op<"conv.fptofp">,
  Results<(outs AnyFloat:$res)>,
  Arguments<(ins
    AnyFloat:$arg,
    RoundingModeAttr:$roundingMode
  )> {

  let summary = "Convert between floating point types";

  string baseDescription = [{
    Convert `$arg` to the type of the result `$res`. If necessary, round the
    result according to `$roundingMode`.
  }];

  string llvmBuilder = [{
    $res = createGenISAFpToFp(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    $arg attr-dict `:` type($arg) `to` type($res)
  }];
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

def GENX_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

/// Enum attribute of the different precision types.
def PrecisionType : I32EnumAttr<"PrecisionType", "GENX precision type",
  [
    I32EnumAttrCase<"PRECISION_UNUSED", 0>,
    I32EnumAttrCase<"U8",               1>,
    I32EnumAttrCase<"U4",               2>,
    I32EnumAttrCase<"U2",               3>,
    I32EnumAttrCase<"S8",               4>,
    I32EnumAttrCase<"S4",               5>,
    I32EnumAttrCase<"S2",               6>,
    I32EnumAttrCase<"BF8",              7>,
    I32EnumAttrCase<"TF32",             8>,
    I32EnumAttrCase<"BF16",             9>,
    I32EnumAttrCase<"FP16",             10>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def PrecisionTypeAttr : EnumAttr<GENX_Dialect, PrecisionType, "precision_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def GENX_MatrixDPASOp : GENX_Op<"matrix.dpas">,
  Results<(outs FixedVectorOf<[GENX_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [GENX_MatrixElemType]>:$c,
    FixedVectorOfRankAndType<[1], [GENX_MatrixElemType]>:$a,
    FixedVectorOfRankAndType<[1], [GENX_MatrixElemType]>:$b,
    PrecisionTypeAttr:$pa,
    PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "GENX matrix multiply-add (for PVC)";

  string baseDescription = [{
    The 'genx.matrix.dpas' operation is a matrix multiply-add operation as follows:

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

def GENX_Matrix2DBlockLoadOp : GENX_Op<"matrix.2Dblockload">,
  Results<(outs FixedVectorOf<[GENX_MatrixElemType]>:$res)>,
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

  let summary = "GENX 2D block load";

  string baseDescription = [{
    The 'genx.matrix.2Dblockload' operation loads a submatrix from an array in memory.
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

def GENX_Matrix2DBlockStoreOp : GENX_Op<"matrix.2Dblockstore">,
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
    FixedVectorOf<[GENX_MatrixElemType]>:$stored_val
  )> {

  let summary = "GENX 2D block store";

  string baseDescription = [{
    The 'genx.matrix.2Dblockstore' operation stores to a submatrix from an array in memory.
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

#endif // GENXIR_OPS
