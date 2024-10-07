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

#include <compiler/utils/address_spaces.h>
#include <compiler/utils/attributes.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/pass_functions.h>
#include <compiler/utils/replace_local_module_scope_variables_pass.h>
#include <llvm/ADT/PriorityWorklist.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <multi_llvm/vector_type_helper.h>

#include <algorithm>
#include <cassert>
#include <functional>

using namespace llvm;

#define DEBUG_TYPE "replace-module-scope-vars"

namespace {
using AlignIntTy = uint64_t;

// Creates and returns a new GEP instruction, inserted before input parameter
// 'inst'. This GEP points to the element at 'index' of the struct living at
// the final argument of each function.
GetElementPtrInst *generateStructGEP(Instruction &inst,
                                     StructType *funcsStructTy,
                                     unsigned index) {
  // find the function the instruction is in
  auto func = inst.getFunction();

  // the local module-scope variables struct we added to each function
  auto funcsStruct = compiler::utils::getLastArgument(func);

  assert(funcsStruct->getType()->isPointerTy());

  // the type with which to index into our struct type
  auto indexTy = Type::getInt32Ty(inst.getModule()->getContext());

  // create a new GEP just before the instruction
  auto GEP = GetElementPtrInst::CreateInBounds(
      funcsStructTy, funcsStruct,
      {ConstantInt::get(indexTy, 0), ConstantInt::get(indexTy, index)}, "",
      &inst);
  return GEP;
}

// Given the type of a __local variable about to be added to the
// struct function calculates and returns the alignment of the type.
AlignIntTy calculateTypeAlign(Type *type, const DataLayout &layout) {
  // Get underlying type if variable is an array
  while (type->isArrayTy()) {
    type = type->getArrayElementType();
  }

  // 3 component wide vectors have the size of 4 components according to the
  // OpenCL spec section 6.1.5 'Alignment of Types'
  unsigned int vectorWidth =
      type->isVectorTy() ? multi_llvm::getVectorNumElements(type) : 1;
  if (3 == vectorWidth) {
    vectorWidth = 4;
  }

  // if we have a pointer type return the size of a pointer on the target
  if (type->isPointerTy()) {
    return layout.getPointerSize();
  }

  // size of member in bytes - at least 8 bits to avoid zero alignment on
  // integer types smaller than i8.
  const unsigned int vectorSize =
      (std::max(type->getScalarSizeInBits(), 8u) * vectorWidth) / 8;

  return vectorSize;
}

// Variables in the local address space not passed as arguments can only be
// declared in the outermost scope of a kernel function. Here we find the kernel
// function the local address space global resides in.
Function *determineKernel(GlobalVariable &global) {
  auto global_user = *(global.user_begin());
  if (auto instruction = dyn_cast<Instruction>(global_user)) {
    return instruction->getFunction();
  } else if (ConstantVector *cv = dyn_cast<ConstantVector>(global_user)) {
    User *cv_user = *(cv->user_begin());
    auto instruction = cast<Instruction>(cv_user);
    return instruction->getFunction();
  } else if (global_user) {
    global_user->print(errs());
    llvm_unreachable("Unknown user used the local module-scope variable!");
  }
  return nullptr;
}

// Information associated to with a local address space module scope variable
// that is needed to update it's debug info metadata
struct GlobalVarDebugInfoWrapper final {
  // Byte offset into struct of replacement variables
  unsigned offset;
  // Associated debug info metadata entry
  DIGlobalVariable *DIGlobal;
  // Kernel function variable was defined in
  Function *function;
};

// Check if a user is an instruction and if so add it to the Visited, Worklist
// and FuncsToClone. If it's not an instruction repeat for all its users
void checkUsersForInstructions(
    User *user, llvm::SmallPtrSet<llvm::Function *, 4> &Visited,
    llvm::SmallVector<llvm::Function *, 4> &FuncsToClone,
    llvm::SmallPriorityWorklist<llvm::Function *, 4> &Worklist) {
  if (auto *I = dyn_cast<Instruction>(user)) {
    auto *F = I->getFunction();
    if (Visited.insert(F).second) {
      Worklist.insert(F);
      FuncsToClone.push_back(F);
      LLVM_DEBUG(
          dbgs() << "Function '" << F->getName()
                 << "' requires additional local module struct parameter\n");
    }
  } else {
    for (auto *user_of_user : user->users()) {
      checkUsersForInstructions(user_of_user, Visited, FuncsToClone, Worklist);
    }
  }
}

/// @brief Clone all required functions in a module, appending an extra
/// parameter to them if they are part of the call graph required for access to
/// local variables.
///
/// @param module llvm module containing the functions
/// @param newParamType Type of the parameter to be added
/// @param newParamAttrs Parameter attributes of the parameter to be added
/// @return bool if the module has changed (currently always true)
///
/// This recurses through all the users of the local variables to look for any
/// functions which use them as well as assuming that the top level kernels must
/// have them.
bool addParamToAllRequiredFunctions(llvm::Module &module,
                                    llvm::Type *const newParamType,
                                    const llvm::AttributeSet &newParamAttrs) {
  llvm::SmallPtrSet<llvm::Function *, 4> Visited;
  llvm::SmallVector<llvm::Function *, 4> FuncsToClone;
  llvm::SmallPriorityWorklist<llvm::Function *, 4> Worklist;

  // Iterate through the top level functions checking if they are kernels.
  for (auto &F : module.functions()) {
    // Kernel entry points must present a consistent ABI to external users
    if (compiler::utils::isKernelEntryPt(F)) {
      Visited.insert(&F);
      Worklist.insert(&F);
      FuncsToClone.push_back(&F);
      LLVM_DEBUG(
          dbgs() << "Function '" << F.getName()
                 << "' requires additional local module struct parameter\n");
      continue;
    }
  }

  // Check each global's users if they are instructions or recurse up the user
  // chain if not. If an Instruction is found we add it to the functions to
  // clone.
  for (auto &global : module.globals()) {
    for (auto *user : global.users()) {
      checkUsersForInstructions(user, Visited, FuncsToClone, Worklist);
    }
  }

  // Iterate over the functions that require local struct parameters and
  // recursively register all callers of those functions as needing local struct
  // parameters too.
  while (!Worklist.empty()) {
    Function *F = Worklist.pop_back_val();
    for (auto *U : F->users()) {
      if (auto *CB = dyn_cast<CallBase>(U)) {
        auto *Caller = CB->getFunction();
        if (Visited.insert(Caller).second) {
          Worklist.insert(Caller);
          FuncsToClone.push_back(Caller);
          LLVM_DEBUG(dbgs() << "Function '" << Caller->getName()
                            << "' requires local struct parameters\n");
        }
      } else {
        report_fatal_error("unhandled user type");
      }
    }
  }

  // Ideally cloneFunctionsAddArg() would take a list of functions, but
  // currently takes a std::function so we search the created vector of
  // functions.
  return compiler::utils::cloneFunctionsAddArg(
      module,
      [newParamType, newParamAttrs](llvm::Module &) {
        return compiler::utils::ParamTypeAttrsPair{newParamType, newParamAttrs};
      },
      [&FuncsToClone](const llvm::Function &func, bool &ClonedWithBody,
                      bool &ClonedNoBody) {
        ClonedWithBody = llvm::is_contained(FuncsToClone, &func);
        ClonedNoBody = false;
      },
      nullptr /*updateMetaDataCallback*/);
}

}  // namespace

PreservedAnalyses compiler::utils::ReplaceLocalModuleScopeVariablesPass::run(
    Module &M, ModuleAnalysisManager &) {
  // the element types of the struct of replacement local module-scope
  // variables we are replacing
  SmallVector<Type *, 8> structElementTypes;

  // ordered list of kernel names which are used to find cached function
  // types. StringRef is safe here because the names will be taken over from
  // the old functions to the new ones.
  SmallVector<StringRef, 4> names;

  // unmodified function types of functions in the module
  DenseMap<StringRef, FunctionType *> functionTypes;

  for (auto &F : M.functions()) {
    if (isKernel(F)) {
      names.push_back(F.getName());
      functionTypes[F.getName()] = F.getFunctionType();
    }
  }

  // a map from the original global variable to the index into
  // structElementTypes
  ValueMap<GlobalVariable *, unsigned> index_map;

  // the global variables we need to process and remove
  SmallVector<GlobalVariable *, 8> globals;

  // maps variables in `globals` we're processing to helper information
  // needed for updating debug info
  DenseMap<GlobalVariable *, GlobalVarDebugInfoWrapper> debug_info_map;

  // __local address space automatic variables are represented in the LLVM
  // module as global variables with address space 3.
  //
  // This pass identifies these variables and places them into a struct
  // allocated in a newly created wrapper function. A pointer to the struct
  // is then passed via a parameter to the original kernel.
  for (auto &global : M.globals()) {
    // get the type of the global variable
    const auto type = global.getType();

    if (global.use_empty()) {
      continue;
    }

    if (type->isPointerTy() &&
        type->getPointerAddressSpace() == AddressSpace::Local) {
      // and save that this is a global we care about
      globals.push_back(&global);
    }
  }

  // if we found no local module-scope variables to be replaced...
  if (globals.empty()) {
    // ... then we're done!
    return PreservedAnalyses::all();
  }

  // Pad struct so that members are aligned.
  //
  // Unlike x86, ARM architecture alignment can be different from the
  // member size. So that __local alignment is OpenCL conformant
  // we need to manually pad our struct.
  //
  // To do this we keep track of each local module-scope elements
  // offset in the struct, and ensure that it is a multiple of
  // that elements alignment. Finally we then align the whole struct
  // to the largest alignment found out of all our __local members.

  // track largest member alignment found so far.
  unsigned int maxAlignment = 0;
  // byte offset in struct of current member
  unsigned int offset = 0;
  const auto &dl = M.getDataLayout();
  for (auto &global : globals) {
    auto memberType = global->getValueType();

    // alignment of the new struct member, in the case where we can't
    // calculate this, e.g. struct types, use the alignment of the llvm
    // global. This is also needed if '__attribute__(aligned)' was used to
    // set a specific alignment.
    const unsigned int alignment =
        std::max(global->getAlignment(), calculateTypeAlign(memberType, dl));
    assert(alignment > 0 && "'0' is an impossible alignment");

    // check if this is the largest alignment seen so far
    maxAlignment = std::max(alignment, maxAlignment);

    // check if member is not already aligned
    const unsigned int remainder = offset % alignment;
    if (0 != remainder) {
      // calculate number of padding bytes
      const unsigned int padding = alignment - remainder;

      // Use a byte array to pad struct rather than trying to create
      // an arbitrary intNTy, since this may not be supported by the backend.
      const auto padByteType = Type::getInt8Ty(M.getContext());
      const auto padByteArrayType = ArrayType::get(padByteType, padding);
      structElementTypes.push_back(padByteArrayType);

      // bump offset by padding size
      offset += padding;
    }

    // we need the byte-offset when generating debug info
    debug_info_map[global] = {offset, nullptr, nullptr};

    // map the global variable to its index in structElementTypes
    index_map[global] = structElementTypes.size();

    // then add our element type to the struct
    structElementTypes.push_back(memberType);

    // update the offset based on the type's size
    auto allocSize = dl.getTypeAllocSize(memberType);
    if (dl.getTypeAllocSize(memberType).isScalable()) {
      // Not an assert because this can happen in user-supplied IR
      report_fatal_error("Scalable types in local memory are not supported");
    }
    const unsigned int totalSize = allocSize.getFixedValue();
    offset += totalSize;
  }

  // create a struct containing all the local module-scope variables
  auto structTy = StructType::create(structElementTypes, "localVarTypes");

  // change all our functions to take a pointer to the new structTy we created
  const AttributeSet defaultAttrs;
  addParamToAllRequiredFunctions(M, structTy->getPointerTo(), defaultAttrs);

  // Check if we have debug info, if so we need to fix it up to turn global
  // variable entries into local variable ones.
  if (const auto NMD = M.getNamedMetadata("llvm.dbg.cu")) {
    const DIBuilder DIB(M, /*AllowUnresolved*/ false);

    for (auto *CUOp : NMD->operands()) {
      // Find module compilation unit
      DICompileUnit *CU = cast<DICompileUnit>(CUOp);

      // Check if there are any debug info global variables, as the DMA
      // pass can create global variables without debug metadata attached.
      auto DIGlobalVariables = CU->getGlobalVariables();
      if (DIGlobalVariables.empty()) {
        continue;
      }
      // Updated list of global debug info variables so that it no longer
      // contains entries we will later replace with DILocalVariable metadata
      SmallVector<Metadata *, 2> CU_DIExprs;
      for (auto &global : M.globals()) {
        // Get debug info expression for global variable
        SmallVector<DIGlobalVariableExpression *, 1> Global_DIExprs;
        global.getDebugInfo(Global_DIExprs);

        if (Global_DIExprs.empty()) {
          continue;
        }

        if (globals.end() == find(globals, &global)) {
          // This is not a __local address space variable we will
          // replace, so retain its debug info in the CU MDNode
          CU_DIExprs.append(Global_DIExprs.begin(), Global_DIExprs.end());
        } else {
          // We will replace this debug info variable later
          assert(Global_DIExprs.size() == 1 &&
                 "Only expecting a single debug info variable");
          debug_info_map[&global].DIGlobal = Global_DIExprs[0]->getVariable();
        }
      }
      CU->replaceGlobalVariables(MDTuple::get(M.getContext(), CU_DIExprs));
    }
  }

  for (auto &global : globals) {
    const SmallVector<User *, 8> users(global->users());

    for (auto *user : users) {
      // if we have a constant expression, we need to force it back to a
      // normal instruction, as we are removing the constant that the
      // constant expression was associated with (we are removing the global
      // variable), we can't use a constant expression to calculate the
      // result.
      if (auto *constant = dyn_cast<ConstantExpr>(user)) {
        replaceConstantExpressionWithInstruction(constant);
      }
    }
  }

  for (auto &global : globals) {
    if (debug_info_map[global].DIGlobal) {
      // If global variable has debug info, find out what kernel the __local
      // variable was defined in so we can use that information later.
      debug_info_map[global].function = determineKernel(*global);
      assert(debug_info_map[global].function);
    }

    // For each user that matches a specific kind of instruction, we do 3
    // different things:
    // 1) Create a GEP instruction to retrieve the address of the local
    // version of 'global' in the newly created local struct.
    // 2) We create a cast instruction to cast the type of the GEP created
    // in 1) to the type of the global instruction.
    // 3) Replace the use of the global instruction with the instruction
    // created in 2).
    const SmallVector<User *, 4> users(global->users());
    for (auto *user : users) {
      // if we have a GEP instruction...
      if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(user)) {
        auto local = generateStructGEP(*gep, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", gep);

        gep->setOperand(0, castedLocal);
        gep->setIsInBounds();
      } else if (CastInst *cast = dyn_cast<CastInst>(user)) {
        auto local = generateStructGEP(*cast, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", cast);

        cast->setOperand(0, castedLocal);
      } else if (LoadInst *load = dyn_cast<LoadInst>(user)) {
        auto local = generateStructGEP(*load, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", load);

        load->setOperand(0, castedLocal);
      } else if (StoreInst *store = dyn_cast<StoreInst>(user)) {
        auto local = generateStructGEP(*store, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", store);
        // global could be pointer or value operand of the store
        if (store->getValueOperand() == global) {
          store->setOperand(0, castedLocal);
        } else {
          store->setOperand(1, castedLocal);
        }
      } else if (ConstantVector *cv = dyn_cast<ConstantVector>(user)) {
        // Because 'cv' is not an instruction, we have to iterate over all its
        // users and do the work for all of them individually.
        for (auto cvIt = cv->user_begin(); cvIt != cv->user_end();) {
          auto cvUser = *cvIt++;
          auto inst = ::cast<Instruction>(cvUser);
          auto local = generateStructGEP(*inst, structTy, index_map[global]);

          auto castedLocal =
              CastInst::CreatePointerCast(local, global->getType(), "", inst);

          auto indexTy = Type::getInt32Ty(M.getContext());
          Value *newCv = UndefValue::get(cv->getType());

          // We can't simply 'setOperand' in a 'ConstantVector'. We have to
          // recreate it from scratch.
          for (unsigned i = 0; i < cv->getNumOperands(); ++i) {
            if (cv->getOperand(i) == global) {
              newCv = InsertElementInst::Create(
                  newCv, castedLocal, ConstantInt::get(indexTy, i), "", inst);
            } else {
              newCv = InsertElementInst::Create(newCv, cv->getOperand(i),
                                                ConstantInt::get(indexTy, i),
                                                "", inst);
            }
          }

          // And don't forget to replace 'cv' by 'newCv'.
          inst->replaceUsesOfWith(cv, newCv);
        }
      } else if (PHINode *phi = dyn_cast<PHINode>(user)) {
        // Because we can't create 1) before a phi node, we have to create it
        // before the terminator of the incoming block.
        for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
          if (phi->getIncomingValue(i) == global) {
            auto incomingBlock = phi->getIncomingBlock(i);
            auto incomingBlockT = incomingBlock->getTerminator();
            auto local =
                generateStructGEP(*incomingBlockT, structTy, index_map[global]);

            auto castedLocal = CastInst::CreatePointerCast(
                local, global->getType(), "", incomingBlockT);

            phi->setIncomingValue(i, castedLocal);
          }
        }
      } else if (AtomicRMWInst *atomic = dyn_cast<AtomicRMWInst>(user)) {
        auto local = generateStructGEP(*atomic, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", atomic);

        // global could be pointer or value operand of the atomic
        if (atomic->getPointerOperand() == global) {
          atomic->setOperand(0, castedLocal);
        } else {
          atomic->setOperand(1, castedLocal);
        }
      } else if (auto *atomic = dyn_cast<AtomicCmpXchgInst>(user)) {
        const auto local =
            generateStructGEP(*atomic, structTy, index_map[global]);
        const auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", atomic);

        // global could be the pointer
        if (atomic->getPointerOperand() == global) {
          atomic->setOperand(0, castedLocal);
        }
        // the comparison value
        if (atomic->getCompareOperand() == global) {
          atomic->setOperand(1, castedLocal);
        }
        // the new value
        if (atomic->getNewValOperand() == global) {
          atomic->setOperand(2, castedLocal);
        }
      } else if (SelectInst *select = dyn_cast<SelectInst>(user)) {
        auto local = generateStructGEP(*select, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", select);

        // global could be the true or false value of the select
        if (select->getTrueValue() == global) {
          select->setOperand(1, castedLocal);
        } else {
          select->setOperand(2, castedLocal);
        }
      } else if (CallInst *call = dyn_cast<CallInst>(user)) {
        auto local = generateStructGEP(*call, structTy, index_map[global]);

        auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", call);

        unsigned i = 0;
        for (; i < call->getNumOperands(); ++i) {
          if (call->getOperand(i) == global) {
            call->setOperand(i, castedLocal);
          }
        }
      } else if (InsertElementInst *insertIns =
                     dyn_cast<InsertElementInst>(user)) {
        auto local = generateStructGEP(*insertIns, structTy, index_map[global]);
        auto castedLocal = CastInst::CreatePointerCast(local, global->getType(),
                                                       "", insertIns);
        // Update middle operand as the others are the vector and index
        insertIns->setOperand(1, castedLocal);
      } else if (auto *cmpIns = dyn_cast<CmpInst>(user)) {
        const auto local =
            generateStructGEP(*cmpIns, structTy, index_map[global]);
        const auto castedLocal =
            CastInst::CreatePointerCast(local, global->getType(), "", cmpIns);
        // global could be either side of the compare
        if (cmpIns->getOperand(0) == global) {
          cmpIns->setOperand(0, castedLocal);
        }
        if (cmpIns->getOperand(1) == global) {
          cmpIns->setOperand(1, castedLocal);
        }
      } else {
        user->print(errs());
        llvm_unreachable("Unknown user used the local module-scope variable!");
      }
    }
  }

  // lastly, we create a wrapper function with the original kernel signature
  // of each kernel, which will alloca the struct for the remapped local
  // module-scope variables
  for (const auto &name : names) {
    // the original kernel function
    auto *kernelFunc = M.getFunction(name);

    // the original kernel function type, saved earlier
    auto kernelFuncTy = functionTypes[name];

    auto newFunc =
        Function::Create(kernelFuncTy, kernelFunc->getLinkage(), "", &M);

    // copy over function parameter names
    for (unsigned i = 0, e = newFunc->arg_size(); i != e; i++) {
      newFunc->getArg(i)->setName(kernelFunc->getArg(i)->getName());
    }
    // copy over function/parameter/ret attributes
    copyFunctionAttrs(*kernelFunc, *newFunc, newFunc->arg_size());

    auto baseName = getOrSetBaseFnName(*newFunc, *kernelFunc);
    newFunc->setName(baseName + ".mux-local-var-wrapper");

    // copy over function metadata
    copyFunctionMetadata(*kernelFunc, *newFunc);
    // drop the old function's kernel information - we've stolen it.
    dropIsKernel(*kernelFunc);

    // copy the calling convention too
    newFunc->setCallingConv(kernelFunc->getCallingConv());

    // we don't use exceptions
    newFunc->addFnAttr(Attribute::NoUnwind);

    // next, set the function to always inline unless it has a noinline
    // attribute.
    if (!kernelFunc->hasFnAttribute(Attribute::NoInline)) {
      kernelFunc->addFnAttr(Attribute::AlwaysInline);
    }

    // lastly set the linkage to internal
    kernelFunc->setLinkage(GlobalValue::InternalLinkage);

    // move debug info for function over
    newFunc->setSubprogram(kernelFunc->getSubprogram());
    kernelFunc->setSubprogram(nullptr);

    // create an irbuilder and basic block for our new function
    IRBuilder<> ir(BasicBlock::Create(newFunc->getContext(), "", newFunc));

    // stack allocate the local module-scope variables struct
    auto alloca = ir.CreateAlloca(structTy);
    alloca->setAlignment(MaybeAlign(maxAlignment).valueOrOne());

    // Generate debug info metadata for the globals we have replaced
    // which previously had debug info attached
    for (auto global : globals) {
      auto debug_info_wrapper = debug_info_map[global];
      auto DIGlobal = debug_info_wrapper.DIGlobal;
      if (!DIGlobal) {
        // No debug info for GlobalVariable
        continue;
      }

      // Expression for byte offset in newly allocated struct where our
      // replacement variable lives
      const unsigned offset = debug_info_wrapper.offset;
      const uint64_t dwPlusOp = dwarf::DW_OP_plus_uconst;
      DIBuilder DIB(M, /*AllowUnresolved*/ false);
      auto offset_expr =
          DIB.createExpression(ArrayRef<uint64_t>{dwPlusOp, offset});

      // enqueued_kernel_scope is true if the variable was originally defined
      // in kernelFunc, the kernel being enqueued by the user, rather than
      // another kernel function being called by kernelFunc.
      auto func = debug_info_wrapper.function;
      const bool enqueued_kernel_scope = !func->getSubprogram();
      auto DISubprogram = enqueued_kernel_scope ? newFunc->getSubprogram()
                                                : func->getSubprogram();

      // We can't guarantee a subprogram for all functions.
      // FIXME: Should we be able to? Do we need to clone subprograms somehow?
      // See CA-4241.
      if (!DISubprogram) {
        continue;
      }

      // Create replacement debug metadata entry representing the global
      // as a DILocalVariable in the kernel function scope.
      auto DILocal = DIB.createAutoVariable(
          DISubprogram, DIGlobal->getName(), DIGlobal->getFile(),
          DIGlobal->getLine(), dyn_cast<DIType>(DIGlobal->getType()));

      // Insert debug declare intrinsic pointing to the location of
      // the variable in our allocated struct
      auto *location =
          DILocation::get(DISubprogram->getContext(), DIGlobal->getLine(),
                          /*Column*/ 0, DISubprogram);
      if (enqueued_kernel_scope) {
        DIB.insertDeclare(alloca, DILocal, offset_expr, location,
                          alloca->getParent());
      } else {
        // A pointer to our struct is passed as the last argument to each
        // function, use this argument if the global came from another kernel
        // function which is called by kernelFunc.
        auto last_arg = func->arg_end() - 1;
        DIB.insertDeclare(last_arg, DILocal, offset_expr, location,
                          func->getEntryBlock().getFirstNonPHIOrDbg());
      }
    }

    // create a buffer for our args
    SmallVector<Value *, 8> args;

    for (auto &arg : newFunc->args()) {
      args.push_back(&arg);
    }

    // add the new alloca for the local module-scope variables struct
    args.push_back(alloca);

    // call the original function
    auto ci = ir.CreateCall(kernelFunc, args);
    ci->setCallingConv(kernelFunc->getCallingConv());
    ci->setAttributes(getCopiedFunctionAttrs(*kernelFunc));

    // and return void
    ir.CreateRetVoid();
  }

  // erase all the global variables that we have removed all uses for
  for (auto global : globals) {
    // Vecz generates constant vector with global variable with local scope.
    // In this case, if we try to remove the global variable, llvm generates
    // assert because there are still uses with constant vector in
    // LLVMContext. As a result, if constant vector uses global variable with
    // local scope, keep it.
    bool keepIt = false;
    for (auto *user : global->users()) {
      if (isa<ConstantVector>(user)) {
        keepIt = true;
        break;
      }
    }

    if (!keepIt) {
      global->eraseFromParent();
    }
  }

  return PreservedAnalyses::none();
}
