//===- polygeist-opt.cpp - The polygeist-opt driver -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'polygeist-opt' tool, which is the polygeist analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/PolygeistPasses.h"
#include "mlir/Conversion/SYCLPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Polygeist/IR/Polygeist.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registerTransformsPasses();
  registerConversionPasses();
  registerAffinePasses();
  registerAsyncPasses();
  arith::registerArithPasses();
  func::registerFuncPasses();
  registerGPUPasses();
  LLVM::registerLLVMPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  spirv::registerSPIRVPasses();
  vector::registerVectorPasses();
  polygeist::registerPolygeistPasses();
  polygeist::registerConvertPolygeistToLLVM();
  sycl::registerSYCLPasses();
  sycl::registerConversionPasses();

  // Register MLIR stuff
  registry.insert<AffineDialect, func::FuncDialect, LLVM::LLVMDialect,
                  memref::MemRefDialect, async::AsyncDialect, func::FuncDialect,
                  arith::ArithDialect, scf::SCFDialect, gpu::GPUDialect,
                  NVVM::NVVMDialect, omp::OpenMPDialect, math::MathDialect,
                  DLTIDialect, polygeist::PolygeistDialect, sycl::SYCLDialect,
                  vector::VectorDialect, spirv::SPIRVDialect>();

  // TODO: We should not be using these extensions. Make sure we do not generate
  // invalid pointers/memrefs from codegen. Also present in driver.cc.
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMStructType::attachInterface<
        PtrElementModel<LLVM::LLVMStructType>>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMPointerType::attachInterface<
        PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
        *ctx);
  });

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Polygeist modular optimizer driver", registry));
}
