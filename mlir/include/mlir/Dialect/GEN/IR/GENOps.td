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

def IntegerOrFloatType : AnyTypeOf<[AnySignlessInteger, AnyFloat]>;

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
    $kind $value `,` $mask attr-dict `:` type($res)
  }];
}

#endif // GEN_OPS
