//===- Polygeist.td - Polygeist dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_OPS
#define POLYGEIST_OPS

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"

def CacheLoad
    : Polygeist_Op<"cacheload"> {

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = memref.getType().cast<MemRefType>();
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];
  let summary = "load from cross barier cache";
  let description = [{}];
}

// HasParent<"ParallelOp">,
def BarrierOp
    : Polygeist_Op<"barrier",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins Variadic<Index>:$indices);
  let summary = "barrier for parallel loops";
  let description = [{}];
  let hasCanonicalizer = true;
}

//===----------------------------------------------------------------------===//
// SubIndexOp
//===----------------------------------------------------------------------===//

def SubIndexOp : Polygeist_Op<"subindex", [
  ViewLikeOpInterface, NoSideEffect
]> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source, Index : $index);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}


def StreamToTokenOp : Polygeist_Op<"stream2token", [
  NoSideEffect
]> {
  let summary = "Extract an async stream from a cuda stream";

  let arguments = (ins AnyType : $source);
  let results = (outs AnyType : $result);
}

//===----------------------------------------------------------------------===//
// Memref2PointerOp
//===----------------------------------------------------------------------===//

def Memref2PointerOp : Polygeist_Op<"memref2pointer", [
  ViewLikeOpInterface, NoSideEffect
]> {
  let summary = "Extract and LLVM pointer from a MemRef";

  let arguments = (ins AnyMemRef : $source);
  let results = (outs LLVM_AnyPointer : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def Pointer2MemrefOp : Polygeist_Op<"pointer2memref", [
  ViewLikeOpInterface, NoSideEffect
]> {
  let summary = "Upgrade a pointer to a memref";

  let arguments = (ins LLVM_AnyPointer : $source);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def TrivialUseOp : Polygeist_Op<"trivialuse"> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source);

}

def TypeSizeOp : Polygeist_Op<"typeSize", [NoSideEffect]> {
  let summary = "Get size of type";
  let arguments = (ins TypeAttr : $source);
  let results = (outs AnyType : $result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def TypeAlignOp : Polygeist_Op<"typeAlign", [NoSideEffect]> {
  let summary = "Get alignment of type";
  let arguments = (ins TypeAttr : $source);
  let results = (outs AnyType : $result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
#endif // POLYGEIST_OPS
