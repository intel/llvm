//===- SYCLFuncRegistry - SYCL functions registry --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement a registry of SYCL functions callable by the compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToLLVM/SYCLFuncRegistry.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SYCLToLLVM/DialectBuilder.h"
#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-func-registry"

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// SYCLFuncDescriptor
//===----------------------------------------------------------------------===//

void SYCLFuncDescriptor::declareFunction(ModuleOp &module, OpBuilder &b) {
  LLVMBuilder builder(b, module.getLoc());
  funcRef = builder.getOrInsertFuncDecl(name, outputTy, argTys, module);
}

Value SYCLFuncDescriptor::call(FuncId id, ValueRange args,
                               const SYCLFuncRegistry &registry, OpBuilder &b,
                               Location loc) {
  const SYCLFuncDescriptor &funcDesc = registry.getFuncDesc(id);
  LLVM_DEBUG(
      llvm::dbgs() << "Creating SYCLFuncDescriptor::call to funcDesc.funcRef: "
                   << funcDesc.funcRef << "\n");

  SmallVector<Type, 4> funcOutputTys;
  if (!funcDesc.outputTy.isa<LLVM::LLVMVoidType>())
    funcOutputTys.emplace_back(funcDesc.outputTy);

  LLVMBuilder builder(b, loc);
  LLVM::CallOp callOp = builder.genCall(funcDesc.funcRef, funcOutputTys, args);
  
  // TODO: we could check here the arguments against the function signature and
  // assert if there is a mismatch.
  assert(callOp.getNumResults() <= 1 && "expecting a single result");
  return callOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// SYCLFuncRegistry
//===----------------------------------------------------------------------===//

SYCLFuncRegistry *SYCLFuncRegistry::instance = nullptr;

const SYCLFuncRegistry SYCLFuncRegistry::create(
    ModuleOp &module, OpBuilder &builder) {
  if (!instance)
    instance = new SYCLFuncRegistry(module, builder);

  return *instance;
}

SYCLFuncRegistry::SYCLFuncRegistry(ModuleOp &module, OpBuilder &builder)
    : registry() {
  MLIRContext *context = module.getContext();
  LowerToLLVMOptions options(context);
  LLVMTypeConverter converter(context, options);
  populateSYCLToLLVMTypeConversion(converter);

  Type id1PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 1)));
  Type id2PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 2)));
  Type id3PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 3)));
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto i64Ty = IntegerType::get(context, 64);

  // Construct the SYCL functions descriptors (enum,
  // function name, signature).
  // clang-format off
  std::vector<SYCLFuncDescriptor> descriptors = {
      // cl::sycl::id<1>::id()
      SYCLFuncDescriptor(SYCLFuncDescriptor::FuncId::Id1CtorDefault,
                         "_ZN2cl4sycl2idILi1EEC2Ev", voidTy, {id1PtrTy}),
      // cl::sycl::id<2>::id()
      SYCLFuncDescriptor(SYCLFuncDescriptor::FuncId::Id2CtorDefault,
                         "_ZN2cl4sycl2idILi2EEC2Ev", voidTy, {id2PtrTy}),
      // cl::sycl::id<3>::id()
      SYCLFuncDescriptor(SYCLFuncDescriptor::FuncId::Id3CtorDefault,
                         "_ZN2cl4sycl2idILi3EEC2Ev", voidTy, {id3PtrTy}),

      // cl::sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id1CtorSizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE",
          voidTy, {id1PtrTy, i64Ty}),
      // cl::sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id2CtorSizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE",
          voidTy, {id2PtrTy, i64Ty}),
      // cl::sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id3CtorSizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE",
          voidTy, {id3PtrTy, i64Ty}),

      // cl::sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id1CtorRange,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm",
          voidTy, {id1PtrTy, i64Ty, i64Ty}),
      // cl::sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id2CtorRange,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm",
          voidTy, {id2PtrTy, i64Ty, i64Ty}),
      // cl::sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id3CtorRange,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm",
          voidTy, {id3PtrTy, i64Ty, i64Ty}),      

      // cl::sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id1CtorItem,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm",
          voidTy, {id1PtrTy, i64Ty, i64Ty, i64Ty}),
      // cl::sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id2CtorItem,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm",
          voidTy, {id2PtrTy, i64Ty, i64Ty, i64Ty}),
      // cl::sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id3CtorItem,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm",
          voidTy, {id3PtrTy, i64Ty, i64Ty, i64Ty}),
  };
  // clang-format on

  // Declare SYCL functions and add them to the registry.
  for (SYCLFuncDescriptor &funcDesc : descriptors) {
    funcDesc.declareFunction(module, builder);
    registry.emplace(funcDesc.id, funcDesc);
  }
}
