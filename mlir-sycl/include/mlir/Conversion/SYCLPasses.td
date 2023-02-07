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
  let options = [Option<"useBarePtrCallConv", "use-bare-ptr-call-conv", "bool",
                        /*default=*/"false",
			"Replace MemRef values with bare pointers to the MemRef "
			"element types">];
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
  let constructor = "mlir::sycl::createConvertSYCLToGPUPass()";
  let dependentDialects = [
      "AffineDialect",
      "arith::ArithDialect",
      "gpu::GPUDialect",
      "memref::MemRefDialect",
  ];
}

#endif // MLIR_CONVERSION_SYCLPASSES
