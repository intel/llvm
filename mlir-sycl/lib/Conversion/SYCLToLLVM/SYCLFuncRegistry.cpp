//===- SYCLFuncRegistry - SYCL functions registry -------------------------===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-func-registry"

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// SYCLFuncDescriptor::Id
//===----------------------------------------------------------------------===//
#pragma clang diagnostic ignored "-Wglobal-constructors"

std::map<SYCLFuncDescriptor::Kind, std::string>
    SYCLFuncDescriptor::Id::kindToName = {
        {Kind::Unknown, "unknown"},
};

std::map<std::string, SYCLFuncDescriptor::Kind>
    SYCLFuncDescriptor::Id::nameToKind = {
        {"unknown", Kind::Unknown},
};

//===----------------------------------------------------------------------===//
// SYCLFuncDescriptor
//===----------------------------------------------------------------------===//

Value SYCLFuncDescriptor::call(FuncId funcId, ValueRange args,
                               const SYCLFuncRegistry &registry, OpBuilder &b,
                               Location loc) {
  const SYCLFuncDescriptor &funcDesc = registry.getFuncDesc(funcId);
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

void SYCLFuncDescriptor::declareFunction(ModuleOp &module, OpBuilder &b) {
  LLVMBuilder builder(b, module.getLoc());
  funcRef = builder.getOrInsertFuncDecl(name, outputTy, argTys, module);
}

//===----------------------------------------------------------------------===//
// SYCLFuncRegistry
//===----------------------------------------------------------------------===//

SYCLFuncRegistry *SYCLFuncRegistry::instance = nullptr;

const SYCLFuncRegistry SYCLFuncRegistry::create(ModuleOp &module,
                                                OpBuilder &builder) {
  if (!instance)
    instance = new SYCLFuncRegistry(module, builder);

  return *instance;
}

SYCLFuncDescriptor::FuncId
SYCLFuncRegistry::getFuncId(SYCLFuncDescriptor::Kind kind, Type retType,
                            TypeRange argTypes) const {
  assert(kind != Kind::Unknown && "Invalid descriptor kind");
  LLVM_DEBUG(llvm::dbgs() << "Looking up function of kind: "
                          << SYCLFuncDescriptor::Id::kindToName.at(kind)
                          << "\n";);

  for (const auto &entry : registry) {
    const SYCLFuncDescriptor &desc = entry.second;
    LLVM_DEBUG(llvm::dbgs() << desc << "\n");

    // Skip through entries that do not match the requested kind.
    if (desc.descId.kind != kind) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip, kind does not match\n");
      continue;
    }
    // Ensure that the entry has return and arguments type that match the one
    // requested.
    if (desc.outputTy != retType) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip, return type does not match\n");
      continue;
    }
    if (desc.argTys.size() != argTypes.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tskip, number of arguments does not match\n");
      continue;
    }
    if (!std::equal(argTypes.begin(), argTypes.end(), desc.argTys.begin())) {
      LLVM_DEBUG({
        auto pair = std::mismatch(argTypes.begin(), argTypes.end(),
                                  desc.argTys.begin());
        llvm::dbgs() << "\tskip, arguments " << *pair.first << " and "
                     << *pair.second << " do not match\n";
      });
      continue;
    }

    return desc.descId.funcId;
  }

  llvm_unreachable("Could not find function id");
  return SYCLFuncDescriptor::FuncId::Unknown;
}

SYCLFuncRegistry::SYCLFuncRegistry(ModuleOp &module, OpBuilder &builder)
    : registry() {
  MLIRContext *context = module.getContext();
  LowerToLLVMOptions options(context);
  LLVMTypeConverter converter(context, options);
  populateSYCLToLLVMTypeConversion(converter);

  // Invoke declare derived function descriptors here.
}

void SYCLFuncRegistry::declareFuncDescriptors(
    std::vector<SYCLFuncDescriptor> &descriptors, ModuleOp &module,
    OpBuilder &builder) {
  // Declare function descriptors and add them to the registry.
  for (SYCLFuncDescriptor &desc : descriptors) {
    desc.declareFunction(module, builder);
    registry.emplace(desc.descId.funcId, desc);
  }
}
