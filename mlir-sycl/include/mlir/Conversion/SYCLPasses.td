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
  let dependentDialects = [
      "LLVM::LLVMDialect",
      "arith::ArithDialect",
      "memref::MemRefDialect",
      "spirv::SPIRVDialect",
      "vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SYCLToGPU
//===----------------------------------------------------------------------===//

def ConvertSYCLToGPU : Pass<"convert-sycl-to-gpu", "ModuleOp"> {
  let summary = "Convert SYCL dialect to GPU dialect";
  let description = [{
    Pass to convert supported operations to the GPU dialect.

    This pass converts `sycl.work_group_id`, `sycl.num_work_items`,
    `sycl.work_group_size`, `sycl.local_id`, `sycl.global_id`,
    `sycl.sub_group_id` `sycl.num_sub_groups`, and `sycl.sub_group_size`
    operations to `gpu.block_id` `gpu.grid_dim` `gpu.block_dim` `gpu.thread_id`
    `gpu.global_id` `gpu.sub_group_id` `gpu.num_sub_groups`
    `gpu.sub_group_size`, respectively.

    While operations querying subgroup information are lowered to their
    counterpart `gpu` dialect operations, the rest of the operations require a
    different treatment, as the operation might or might not be passed an
    argument querying for a specific dimension.
};
  }];
  let dependentDialects = [
      "arith::ArithDialect",
      "gpu::GPUDialect",
      "memref::MemRefDialect",
      "vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SYCLToSPIRV
//===----------------------------------------------------------------------===//

def ConvertSYCLToSPIRV : Pass<"convert-sycl-to-spirv", "ModuleOp"> {
  let summary = "Convert SYCL dialect to SPIRV dialect";
  let description = [{
    Pass to convert supported operations to the SPIRV dialect.

    This pass converts `sycl.global_offset`, `sycl.num_work_groups`,
    `sycl.sub_group_max_size` and `sycl.sub_group_local_id` operations to calls
    to the `GlobalOffset`, `NumWorkgroups`, `SubgroupMaxSize` and
    `SubgroupLocalInvocationId` SPIR-V builtins, respectively.

    While operations querying subgroup information are lowered to a single
    operation, the rest of the operations require a different treatment, as the
    operation might or might not be passed an argument querying for a specific
    dimension.
  }];
  let dependentDialects = [
      "arith::ArithDialect",
      "spirv::SPIRVDialect",
      "memref::MemRefDialect",
      "vector::VectorDialect"
  ];
}

#endif // MLIR_CONVERSION_SYCLPASSES
