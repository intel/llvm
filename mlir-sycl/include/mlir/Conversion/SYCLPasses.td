//===-- SYCLPasses.td - Conversion pass definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLPASSES
#define MLIR_CONVERSION_SYCLPASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SYCLToLLVM
//===----------------------------------------------------------------------===//

def ConvertSYCLToLLVM : Pass<"convert-sycl-to-llvm", "ModuleOp"> {
  let summary = "Convert SYCL dialect to LLVM dialect";
  let description = [{
    See docs/SYCLToLLVMDialectConversion/ for more details.
    TODO: add docs referenced above.
  }];
  let constructor = "mlir::sycl::createConvertSYCLToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// SYCLToGPU
//===----------------------------------------------------------------------===//

def ConvertSYCLToGPU : Pass<"convert-sycl-to-gpu", "ModuleOp"> {
  let summary = "Convert SYCL dialect to GPU dialect";
  let description = [{
    Pass to convert supported operations to the GPU dialect.
  }];
  let constructor = "mlir::sycl::createConvertSYCLToGPUPass()";
  let dependentDialects = [
      "AffineDialect",
      "arith::ArithDialect",
      "gpu::GPUDialect",
      "memref::MemRefDialect",
  ];
}

#endif // MLIR_CONVERSION_SYCLPASSES
