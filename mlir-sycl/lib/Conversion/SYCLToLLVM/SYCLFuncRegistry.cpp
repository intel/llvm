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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-func-registry"

using namespace mlir;
using namespace mlir::sycl;

// TODO: move in LLVMBuilder class when available.
static FlatSymbolRefAttr getOrInsertFuncDecl(ModuleOp module, OpBuilder &b,
                                             StringRef funcName,
                                             Type resultType,
                                             ArrayRef<Type> argsTypes,
                                             bool isVarArg = false) {
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFunctionType funcType =
        LLVM::LLVMFunctionType::get(resultType, argsTypes, isVarArg);
    b.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  }
  return SymbolRefAttr::get(b.getContext(), funcName);
}

//===----------------------------------------------------------------------===//
// SYCLFuncDescriptor
//===----------------------------------------------------------------------===//

void SYCLFuncDescriptor::declareFunction(ModuleOp &module, OpBuilder &b) {
  // TODO: use LLVMBuilder once available.
  funcRef = getOrInsertFuncDecl(module, b, name, outputTy, argTys);
}

Value SYCLFuncDescriptor::call(FuncId id, ArrayRef<Value> args,
                               const SYCLFuncRegistry &registry, OpBuilder &b,
                               Location loc) {
  SmallVector<Type, 1> funcOutputTys;
  const SYCLFuncDescriptor &funcDesc = registry.getFuncDesc(id);
  if (!funcDesc.outputTy.isa<LLVM::LLVMVoidType>())
    funcOutputTys.emplace_back(funcDesc.outputTy);

  // TODO: generate the call via LLVMBuilder here
  //  LLVMBuilder builder(b, loc);
  //   return builder.call(funcDesc.funcRef, ArrayRef<Type>(funcOutputsTys), args);
  auto callOp = b.create<LLVM::CallOp>(loc, ArrayRef<Type>(funcOutputTys),
                                       funcDesc.funcRef, args);
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
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto i8Ty = IntegerType::get(context, 8);
  auto opaquePtrTy = LLVM::LLVMPointerType::get(i8Ty);
  auto opaquePtrPtrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
  auto i64Ty = IntegerType::get(context, 64);
  auto i64PtrTy = LLVM::LLVMPointerType::get(i64Ty);

  // Construct the SYCL functions descriptors (enum, function name, signature).
  // clang-format off
  std::vector<SYCLFuncDescriptor> descriptors = {
      // cl::sycl::id<1>::id()
      SYCLFuncDescriptor(SYCLFuncDescriptor::FuncId::Id2CtorDefault,
                         "_ZN2cl4sycl2idILi1EEC2Ev", voidTy, {opaquePtrTy}),
      // cl::sycl::id<2>::id()
      SYCLFuncDescriptor(SYCLFuncDescriptor::FuncId::Id2CtorDefault,
                         "_ZN2cl4sycl2idILi2EEC2Ev", voidTy, {opaquePtrTy}),
      // cl::sycl::id<3>::id()
      SYCLFuncDescriptor(SYCLFuncDescriptor::FuncId::Id3CtorDefault,
                         "_ZN2cl4sycl2idILi3EEC2Ev", voidTy, {opaquePtrTy}),

      // cl::sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id1CtorSizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE",
          voidTy, {opaquePtrTy, i64Ty}),
      // cl::sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id2CtorSizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE",
          voidTy, {opaquePtrTy, i64Ty}),
      // cl::sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id3CtorSizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE",
          voidTy, {opaquePtrTy, i64Ty}),

      // cl::sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id1CtorRange,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm",
          voidTy, {opaquePtrTy, i64Ty, i64Ty}),
      // cl::sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id2CtorRange,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm",
          voidTy, {opaquePtrTy, i64Ty, i64Ty}),
      // cl::sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id3CtorRange,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm",
          voidTy, {opaquePtrTy, i64Ty, i64Ty}),      

      // cl::sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id1CtorItem,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm",
          voidTy, {opaquePtrTy, i64Ty, i64Ty, i64Ty}),
      // cl::sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id2CtorItem,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm",
          voidTy, {opaquePtrTy, i64Ty, i64Ty, i64Ty}),
      // cl::sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
      SYCLFuncDescriptor(
          SYCLFuncDescriptor::FuncId::Id3CtorItem,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm",
          voidTy, {opaquePtrTy, i64Ty, i64Ty, i64Ty}),
  };
  // clang-format on

  // Declare SYCL functions and add them to the registry.
  for (auto &funcDesc : descriptors) {
    funcDesc.declareFunction(module, builder);
    registry.emplace(funcDesc.id, funcDesc);
  }
}
