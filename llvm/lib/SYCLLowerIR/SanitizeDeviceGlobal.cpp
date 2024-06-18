//===-- SanitizeDeviceGlobal.cpp - instrument device global for sanitizer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass adds red zone to each image scope device global and record the
// information like size, red zone size and beginning address. The information
// will be used by address sanitizer.
// TODO: Do this in AddressSanitizer pass when urProgramGetGlobalVariablePointer
// is implemented.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SanitizeDeviceGlobal.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"

#define DEBUG_TYPE "SanitizeDeviceGlobal"

using namespace llvm;

namespace {

// Add extra red zone to each image scope device globals if the module has been
// instrumented by sanitizer pass. And record their infomation like size, red
// zone size, beginning address.
static bool instrumentDeviceGlobal(Module &M) {
  auto &DL = M.getDataLayout();
  IRBuilder<> IRB(M.getContext());
  SmallVector<GlobalVariable *, 8> GlobalsToRemove;
  SmallVector<GlobalVariable *, 8> NewDeviceGlobals;
  SmallVector<Constant *, 8> DeviceGlobalMetadata;

  constexpr uint64_t MaxRZ = 1 << 18;
  constexpr uint64_t MinRZ = 32;

  Type *IntTy = Type::getIntNTy(M.getContext(), DL.getPointerSizeInBits());

  // Device global meta data is described by a structure
  //  size_t device_global_size
  //  size_t device_global_size_with_red_zone
  //  size_t beginning address of the device global
  StructType *StructTy = StructType::get(IntTy, IntTy, IntTy);

  for (auto &G : M.globals()) {
    // Non image scope device globals are implemented by device USM, and the
    // out-of-bounds check for them will be done by sanitizer USM part. So we
    // exclude them here.
    if (!isDeviceGlobalVariable(G) || !hasDeviceImageScopeProperty(G))
      continue;

    Type *Ty = G.getValueType();
    const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
    const uint64_t RightRedzoneSize = [&] {
      // The algorithm for calculating red zone size comes from
      // llvm/lib/Transforms/Instrumentation/AddressSanitizer.cpp
      uint64_t RZ = 0;
      if (SizeInBytes <= MinRZ / 2) {
        // Reduce redzone size for small size objects, e.g. int, char[1].
        // Optimize when SizeInBytes is less than or equal to half of MinRZ.
        RZ = MinRZ - SizeInBytes;
      } else {
        // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 *
        // SizeInBytes.
        RZ = std::clamp((SizeInBytes / MinRZ / 4) * MinRZ, MinRZ, MaxRZ);

        // Round up to multiple of MinRZ.
        if (SizeInBytes % MinRZ)
          RZ += MinRZ - (SizeInBytes % MinRZ);
      }

      assert((RZ + SizeInBytes) % MinRZ == 0);
      return RZ;
    }();
    Type *RightRedZoneTy = ArrayType::get(IRB.getInt8Ty(), RightRedzoneSize);
    StructType *NewTy = StructType::get(Ty, RightRedZoneTy);
    Constant *NewInitializer = ConstantStruct::get(
        NewTy, G.getInitializer(), Constant::getNullValue(RightRedZoneTy));

    // Create a new global variable with enough space for a redzone.
    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G.isConstant(), G.getLinkage(), NewInitializer, "", &G,
        G.getThreadLocalMode(), G.getAddressSpace());
    NewGlobal->copyAttributesFrom(&G);
    NewGlobal->setComdat(G.getComdat());
    NewGlobal->setAlignment(Align(MinRZ));
    NewGlobal->copyMetadata(&G, 0);

    Value *Indices2[2];
    Indices2[0] = IRB.getInt32(0);
    Indices2[1] = IRB.getInt32(0);

    G.replaceAllUsesWith(
        ConstantExpr::getGetElementPtr(NewTy, NewGlobal, Indices2, true));
    NewGlobal->takeName(&G);
    GlobalsToRemove.push_back(&G);
    NewDeviceGlobals.push_back(NewGlobal);
    DeviceGlobalMetadata.push_back(ConstantStruct::get(
        StructTy, ConstantInt::get(IntTy, SizeInBytes),
        ConstantInt::get(IntTy, SizeInBytes + RightRedzoneSize),
        ConstantExpr::getPointerCast(NewGlobal, IntTy)));
  }

  if (GlobalsToRemove.empty())
    return false;

  // Create global to record number of device globals
  GlobalVariable *NumOfDeviceGlobals = new GlobalVariable(
      M, IntTy, false, GlobalValue::ExternalLinkage,
      ConstantInt::get(IntTy, NewDeviceGlobals.size()),
      "__AsanDeviceGlobalCount", nullptr, GlobalValue::NotThreadLocal, 1);
  NumOfDeviceGlobals->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  // Create meta data global to record device globals' information
  ArrayType *ArrayTy = ArrayType::get(StructTy, NewDeviceGlobals.size());
  Constant *MetadataInitializer =
      ConstantArray::get(ArrayTy, DeviceGlobalMetadata);
  GlobalVariable *AsanDeviceGlobalMetadata = new GlobalVariable(
      M, MetadataInitializer->getType(), false, GlobalValue::ExternalLinkage,
      MetadataInitializer, "__AsanDeviceGlobalMetadata", nullptr,
      GlobalValue::NotThreadLocal, 1);
  AsanDeviceGlobalMetadata->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  for (auto *G : GlobalsToRemove)
    G->eraseFromParent();

  return true;
}

}

namespace llvm {

PreservedAnalyses SanitizeDeviceGlobalPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  bool Modified = false;

  Modified |= instrumentDeviceGlobal(M);

  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}
