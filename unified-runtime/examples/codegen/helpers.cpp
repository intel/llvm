/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "helpers.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/LLVMContext.h>

#include <LLVMSPIRVLib/LLVMSPIRVLib.h>

#include <sstream>

std::string generate_plus_one_spv() {
  using namespace llvm;
  LLVMContext ctx;
  std::unique_ptr<Module> module =
      std::make_unique<Module>("code_generated", ctx);
  module->setTargetTriple("spir64-unknown-unknown");
  IRBuilder<> builder(ctx);

  std::vector<Type *> args{Type::getInt32PtrTy(ctx, 1),
                           Type::getInt32PtrTy(ctx, 1)};
  FunctionType *f_type = FunctionType::get(Type::getVoidTy(ctx), args, false);
  Function *f =
      Function::Create(f_type, GlobalValue::LinkageTypes::ExternalLinkage,
                       "plus1", module.get());
  f->setCallingConv(CallingConv::SPIR_KERNEL);

  // get_global_id
  FunctionType *ggi_type =
      FunctionType::get(Type::getInt32Ty(ctx), {Type::getInt32Ty(ctx)}, false);
  Function *get_global_idj =
      Function::Create(ggi_type, GlobalValue::LinkageTypes::ExternalLinkage,
                       "_Z13get_global_idj", module.get());
  get_global_idj->setCallingConv(CallingConv::SPIR_FUNC);

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", f);

  builder.SetInsertPoint(entry);
  Constant *zero = ConstantInt::get(Type::getInt32Ty(ctx), 0);
  Constant *onei = ConstantInt::get(Type::getInt32Ty(ctx), 1);
  Value *idx = builder.CreateCall(get_global_idj, zero, "idx");
  auto argit = f->args().begin();
#if LLVM_VERSION_MAJOR > 15
  Value *firstElemSrc =
      builder.CreateGEP(argit->getType(), argit, idx, "src.idx");
  ++argit;
  Value *firstElemDst =
      builder.CreateGEP(argit->getType(), argit, idx, "dst.idx");
#elif LLVM_VERSION_MAJOR > 12
  Value *firstElemSrc = builder.CreateGEP(
      argit->getType()->getPointerElementType(), argit, idx, "src.idx");
  ++argit;
  Value *firstElemDst = builder.CreateGEP(
      argit->getType()->getPointerElementType(), argit, idx, "dst.idx");
#else
  Value *firstElemSrc = builder.CreateGEP(f->args().begin(), idx, "src.idx");
  Value *firstElemDst = builder.CreateGEP(++argit, idx, "dst.idx");
#endif
  Value *ldSrc = builder.CreateLoad(Type::getInt32Ty(ctx), firstElemSrc, "ld");
  Value *result = builder.CreateAdd(ldSrc, onei, "foo");
  builder.CreateStore(result, firstElemDst);
  builder.CreateRetVoid();

  // set metadata -- pretend we're opencl (see
  // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst#spir-v-instructions-mapped-to-llvm-metadata)
  Metadata *spirv_src_ops[] = {
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(ctx), 3 /*OpenCL_C*/)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(ctx), 102000 /*OpenCL ver 1.2*/))};
  NamedMDNode *spirv_src = module->getOrInsertNamedMetadata("spirv.Source");
  spirv_src->addOperand(MDNode::get(ctx, spirv_src_ops));

  module->print(errs(), nullptr);

  SPIRV::TranslatorOpts opts;
  opts.enableAllExtensions();
  opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);
  opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);

  std::ostringstream ss;
  std::string err;
  auto success = writeSpirv(module.get(), opts, ss, err);
  if (!success) {
    errs() << "Spirv translation failed with error: " << err << "\n";
  } else {
    errs() << "Spirv tranlsation success.\n";
  }
  errs() << "Code size: " << ss.str().size() << "\n";

  return ss.str();
}
