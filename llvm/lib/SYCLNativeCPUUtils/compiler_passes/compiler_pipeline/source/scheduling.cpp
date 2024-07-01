// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/metadata.h>
#include <compiler/utils/pass_functions.h>
#include <compiler/utils/scheduling.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <sys/types.h>

using namespace llvm;

namespace compiler {
namespace utils {

static constexpr const char *WorkItemParamName = "MuxWorkItemInfo";
static constexpr const char *WorkGroupParamName = "MuxWorkGroupInfo";

StructType *getWorkItemInfoStructTy(llvm::Module &M) {
  LLVMContext &ctx = M.getContext();
  // Check whether this struct has previously been defined.
  if (auto *ty = StructType::getTypeByName(ctx, WorkItemParamName)) {
    return ty;
  }
  auto *uint_type = Type::getInt32Ty(ctx);
  auto *size_type = getSizeType(M);
  auto *array_type = ArrayType::get(size_type, 3);

  SmallVector<Type *, WorkItemInfoStructField::total> elements(
      WorkItemInfoStructField::total);

  elements[WorkItemInfoStructField::local_id] = array_type;
  elements[WorkItemInfoStructField::sub_group_id] = uint_type;
  elements[WorkItemInfoStructField::num_sub_groups] = uint_type;
  elements[WorkItemInfoStructField::max_sub_group_size] = uint_type;

  return StructType::create(elements, WorkItemParamName);
}

StructType *getWorkGroupInfoStructTy(llvm::Module &M) {
  LLVMContext &ctx = M.getContext();
  // Check whether this struct has previously been defined.
  if (auto *ty = StructType::getTypeByName(ctx, WorkGroupParamName)) {
    return ty;
  }
  auto *uint_type = Type::getInt32Ty(ctx);
  auto *size_type = getSizeType(M);
  auto *array_type = ArrayType::get(size_type, 3);

  SmallVector<Type *, WorkGroupInfoStructField::total> elements(
      WorkGroupInfoStructField::total);

  elements[WorkGroupInfoStructField::group_id] = array_type;
  elements[WorkGroupInfoStructField::num_groups] = array_type;
  elements[WorkGroupInfoStructField::global_offset] = array_type;
  elements[WorkGroupInfoStructField::local_size] = array_type;
  elements[WorkGroupInfoStructField::work_dim] = uint_type;

  return StructType::create(elements, WorkGroupParamName);
}

void populateStructSetterFunction(Function &F, Argument &structPtrArg,
                                  StructType *const structTy,
                                  uint32_t structFieldIdx, bool hasRankArg) {
  assert(F.isDeclaration() && "Scrubbing existing function");

  F.addFnAttr(Attribute::AlwaysInline);
  F.setLinkage(GlobalValue::InternalLinkage);

  auto argIter = F.arg_begin();

  Value *const indexArg = hasRankArg ? argIter++ : nullptr;

  Value *const valueArg = argIter++;

  IRBuilder<> ir(BasicBlock::Create(F.getContext(), "", &F));

  SmallVector<Value *, 3> gep_indices{ir.getInt32(0),
                                      ir.getInt32(structFieldIdx)};

  if (hasRankArg) {
    gep_indices.push_back(indexArg);
  }

  assert(structPtrArg.getType()->isPointerTy() &&
         "Assuming a pointer type as the last argument");

  Value *gep = ir.CreateGEP(structTy, &structPtrArg, gep_indices);

  ir.CreateStore(valueArg, gep);

  ir.CreateRetVoid();
}

void populateStructGetterFunction(llvm::Function &F, Argument &structPtrArg,
                                  llvm::StructType *const structTy,
                                  uint32_t structFieldIdx, bool hasRankArg,
                                  size_t defaultValue) {
  assert(F.isDeclaration() && "Scrubbing existing function");
  F.addFnAttr(Attribute::AlwaysInline);
  F.setLinkage(GlobalValue::InternalLinkage);

  auto *indexArg = hasRankArg ? F.arg_begin() : nullptr;

  assert(structPtrArg.getType()->isPointerTy() &&
         "Assuming a pointer type as the last argument");

  IRBuilder<> ir(BasicBlock::Create(F.getContext(), "", &F));

  SmallVector<Value *, 3> gep_indices{ir.getInt32(0),
                                      ir.getInt32(structFieldIdx)};

  Value *ret = nullptr;
  Value *cmp = nullptr;

  if (hasRankArg) {
    // we have 3 dimensions; x, y & z
    auto *maxValidIndex = ir.getInt32(3);

    cmp = ir.CreateICmp(CmpInst::ICMP_ULT, indexArg, maxValidIndex);

    auto *sel = ir.CreateSelect(cmp, indexArg, ir.getInt32(0));

    gep_indices.push_back(sel);
  }

  auto gep = ir.CreateGEP(structTy, &structPtrArg, gep_indices);

  ret = ir.CreateLoad(F.getReturnType(), gep);

  if (hasRankArg) {
    ret = ir.CreateSelect(cmp, ret,
                          ConstantInt::get(F.getReturnType(), defaultValue));
  }

  ir.CreateRet(ret);
}

}  // namespace utils
}  // namespace compiler
