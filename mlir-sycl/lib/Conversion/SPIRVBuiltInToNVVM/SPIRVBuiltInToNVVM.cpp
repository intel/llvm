//===- SPIRVBuiltInToNVVM.cpp - SPIRVBuiltIn to NVVM Patterns
//---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SPIRVBuiltIn access to NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Regex.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSPIRVBUILTINTONVVM
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSPIRVBUILTINTONVVM
} // namespace mlir

using namespace mlir;
using namespace mlir::sycl;

namespace {
/// A pass converting MLIR SPIRVBuiltIn operations into NVVM calls.
class ConvertSPIRVBuiltInToNVVMPass
    : public impl::ConvertSPIRVBuiltInToNVVMBase<
          ConvertSPIRVBuiltInToNVVMPass> {
  void runOnOperation() override;
};
} // namespace

/// This function checks if the function with the given name already exists in
/// the module. If it does, it returns the existing function. If not, it creates
/// a new function with the specified name, return type, and argument types, and
/// inserts it into the module.
LLVM::LLVMFuncOp getOrInsertNVVMIntrinsic(OpBuilder &builder, ModuleOp module,
                                          StringRef name, Type retType,
                                          ArrayRef<Type> argTypes) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;

  auto funcType = LLVM::LLVMFunctionType::get(retType, argTypes, false);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto newFunc =
      builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, funcType);

  newFunc.setLinkage(LLVM::Linkage::External);

  return newFunc;
}

/// Marks NVVM kernel functions in the module.
void markNVVMKernelFunctions(ModuleOp module) {
  module.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (funcOp->hasAttr("gpu.kernel")) {
      if (!funcOp->hasAttr("nvvm.kernel")) {
        funcOp->setAttr("nvvm.kernel", UnitAttr::get(funcOp->getContext()));
      }
    }
  });
}

using RewriteHandlerFn =
    std::function<void(OpBuilder &, LLVM::LLVMFuncOp funcOp, ModuleOp module)>;

/// Rewrites SPIRV built-in functions to NVVM intrinsics.
/// This function creates a set of handlers for each SPIRV built-in function
/// that needs to be rewritten.
void rewriteSPIRVBuiltinFunctions(ModuleOp module, MLIRContext *context) {
  OpBuilder builder(context);

  llvm::StringMap<RewriteHandlerFn> handlers;

  static const std::array<std::string, 3> dims = {"x", "y", "z"};
  static const std::array<std::string, 3> suffixes = {"xv", "yv", "zv"};

  for (size_t i = 0; i < 3; ++i) {

    const std::string &dimStr = dims[i];
    const std::string &suffixStr = suffixes[i];

    std::string globalSizeKey = "__spirv_GlobalSize_" + suffixStr;
    std::string globalInvocationIdKey =
        "__spirv_GlobalInvocationId_" + suffixStr;
    std::string globalOffsetKey = "__spirv_GlobalOffset_" + suffixStr;

    std::string nctaidName = "llvm.nvvm.read.ptx.sreg.nctaid." + dimStr;
    std::string ntidName = "llvm.nvvm.read.ptx.sreg.ntid." + dimStr;
    std::string tidName = "llvm.nvvm.read.ptx.sreg.tid." + dimStr;
    std::string ctaidName = "llvm.nvvm.read.ptx.sreg.ctaid." + dimStr;

    handlers[globalSizeKey] = [=](OpBuilder &builder, LLVM::LLVMFuncOp,
                                  ModuleOp module) {
      auto loc = builder.getUnknownLoc();
      auto i32Ty = builder.getI32Type();
      auto i64Ty = builder.getI64Type();

      auto nctaidFunc =
          getOrInsertNVVMIntrinsic(builder, module, nctaidName, i32Ty, {});

      auto ntidFunc =
          getOrInsertNVVMIntrinsic(builder, module, ntidName, i32Ty, {});

      auto nctaid = builder.create<LLVM::CallOp>(
          loc, i32Ty,
          FlatSymbolRefAttr::get(builder.getContext(), nctaidFunc.getName()),
          ValueRange{});
      auto ntid = builder.create<LLVM::CallOp>(
          loc, i32Ty,
          FlatSymbolRefAttr::get(builder.getContext(), ntidFunc.getName()),
          ValueRange{});

      auto nctaid64 =
          builder.create<LLVM::ZExtOp>(loc, i64Ty, nctaid.getResult());
      auto ntid64 = builder.create<LLVM::ZExtOp>(loc, i64Ty, ntid.getResult());

      auto globalSize =
          builder.create<LLVM::MulOp>(loc, i64Ty, nctaid64, ntid64);
      builder.create<LLVM::ReturnOp>(loc, globalSize.getResult());
    };

    handlers[globalInvocationIdKey] = [=](OpBuilder &builder, LLVM::LLVMFuncOp,
                                          ModuleOp module) {
      auto loc = builder.getUnknownLoc();
      auto i32Ty = builder.getI32Type();
      auto i64Ty = builder.getI64Type();

      auto tidFunc =
          getOrInsertNVVMIntrinsic(builder, module, tidName, i32Ty, {});

      auto bdimFunc =
          getOrInsertNVVMIntrinsic(builder, module, ntidName, i32Ty, {});

      auto bidFunc =
          getOrInsertNVVMIntrinsic(builder, module, ctaidName, i32Ty, {});

      auto tid = builder.create<LLVM::CallOp>(
          loc, i32Ty,
          FlatSymbolRefAttr::get(builder.getContext(), tidFunc.getName()),
          ValueRange{});
      auto bdim = builder.create<LLVM::CallOp>(
          loc, i32Ty,
          FlatSymbolRefAttr::get(builder.getContext(), bdimFunc.getName()),
          ValueRange{});
      auto bid = builder.create<LLVM::CallOp>(
          loc, i32Ty,
          FlatSymbolRefAttr::get(builder.getContext(), bidFunc.getName()),
          ValueRange{});

      auto tid64 = builder.create<LLVM::ZExtOp>(loc, i64Ty, tid.getResult());
      auto bdim64 = builder.create<LLVM::ZExtOp>(loc, i64Ty, bdim.getResult());
      auto bid64 = builder.create<LLVM::ZExtOp>(loc, i64Ty, bid.getResult());

      auto mul = builder.create<LLVM::MulOp>(loc, i64Ty, bid64, bdim64);
      auto gid = builder.create<LLVM::AddOp>(loc, i64Ty, mul, tid64);

      builder.create<LLVM::ReturnOp>(loc, gid.getResult());
    };

    handlers[globalOffsetKey] = [=](OpBuilder &builder, LLVM::LLVMFuncOp,
                                    ModuleOp module) {
      auto loc = builder.getUnknownLoc();
      auto i64Ty = builder.getI64Type();
      auto zero = builder.create<LLVM::ConstantOp>(
          loc, i64Ty, builder.getI64IntegerAttr(0));
      builder.create<LLVM::ReturnOp>(loc, zero);
    };
  }

  static const std::array<std::string, 4> builtinBases = {
      "__spirv_LocalInvocationId_", "__spirv_WorkgroupId_",
      "__spirv_NumWorkgroups_", "__spirv_WorkgroupSize_"};

  static const std::array<std::string, 4> nvvmPrefixes = {
      "llvm.nvvm.read.ptx.sreg.tid.", "llvm.nvvm.read.ptx.sreg.ctaid.",
      "llvm.nvvm.read.ptx.sreg.nctaid.", "llvm.nvvm.read.ptx.sreg.ntid."};

  static const std::array<std::string, 3> dimSuffixes = {"xv", "yv", "zv"};
  static const std::array<std::string, 3> dimLetters = {"x", "y", "z"};

  for (size_t i = 0; i < builtinBases.size(); ++i) {
    for (size_t d = 0; d < 3; ++d) {
      std::string funcName = builtinBases[i] + dimSuffixes[d];
      std::string nvvmName = nvvmPrefixes[i] + dimLetters[d];

      handlers[funcName] = [=](OpBuilder &builder, LLVM::LLVMFuncOp,
                               ModuleOp module) {
        auto loc = builder.getUnknownLoc();
        auto i32Ty = builder.getI32Type();
        auto i64Ty = builder.getI64Type();

        auto regFunc =
            getOrInsertNVVMIntrinsic(builder, module, nvvmName, i32Ty, {});
        auto reg = builder.create<LLVM::CallOp>(
            loc, i32Ty,
            FlatSymbolRefAttr::get(builder.getContext(), regFunc.getName()),
            ValueRange{});
        auto reg64 = builder.create<LLVM::ZExtOp>(loc, i64Ty, reg.getResult());
        builder.create<LLVM::ReturnOp>(loc, reg64);
      };
    }
  }

  for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
    auto name = funcOp.getName();
    for (const auto &kv : handlers) {
      if (name.contains(kv.getKey()) && funcOp.empty()) {
        builder.setInsertionPointToStart(funcOp.addEntryBlock());
        kv.getValue()(builder, funcOp, module);
        break;
      }
    }
  }
}

/// Replace accesses to SPIRVBuiltIn variables with calls to the rewritten
/// functions. This function scans through the module for
/// `LLVM::ExtractElementOp` operations that extract elements from vectors
/// representing SPIRVBuiltIn variables. It replaces these operations with calls
/// to the corresponding NVVM functions.
void replaceBuiltinAccessWithCalls(ModuleOp module, MLIRContext *context) {
  OpBuilder builder(context);

  llvm::StringMap<StringRef> builtinMap = {
      {"__spirv_BuiltInGlobalInvocationId", "__spirv_GlobalInvocationId_"},
      {"__spirv_BuiltInGlobalSize", "__spirv_GlobalSize_"},
      {"__spirv_BuiltInGlobalOffset", "__spirv_GlobalOffset_"},
      {"__spirv_BuiltInLocalInvocationId", "__spirv_LocalInvocationId_"},
      {"__spirv_BuiltInWorkgroupId", "__spirv_WorkgroupId_"},
      {"__spirv_BuiltInNumWorkgroups", "__spirv_NumWorkgroups_"},
      {"__spirv_BuiltInWorkgroupSize", "__spirv_WorkgroupSize_"},
  };

  const char *dimSuffixes[3] = {"xv", "yv", "zv"};

  for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
    for (auto &block : func.getBody()) {
      for (auto it = block.begin(), end = block.end(); it != end;) {
        Operation *op = &*it++;
        auto extractOp = dyn_cast<LLVM::ExtractElementOp>(op);
        if (!extractOp)
          continue;

        Value vec = extractOp.getVector();
        Value index = extractOp.getOperand(1);

        auto constIdxOp =
            dyn_cast_or_null<LLVM::ConstantOp>(index.getDefiningOp());
        if (!constIdxOp)
          continue;

        auto intAttr = constIdxOp.getValue().dyn_cast<IntegerAttr>();
        if (!intAttr)
          continue;

        int64_t indexVal = intAttr.getInt();
        if (indexVal < 0 || indexVal > 2)
          continue;

        auto loadOp = vec.getDefiningOp<LLVM::LoadOp>();
        if (!loadOp)
          continue;

        auto addrOfOp = loadOp.getAddr().getDefiningOp<LLVM::AddressOfOp>();
        if (!addrOfOp)
          continue;

        SymbolTableCollection symbolTable;
        auto global = addrOfOp.getGlobal(symbolTable);
        if (!global)
          continue;

        auto globalName = global.getSymName();

        auto iter = builtinMap.find(globalName);
        if (iter == builtinMap.end())
          continue;

        StringRef baseName = iter->second;
        StringRef suffix = dimSuffixes[indexVal];

        LLVM::LLVMFuncOp targetFunc = nullptr;
        for (auto candidate : module.getOps<LLVM::LLVMFuncOp>()) {
          StringRef fname = candidate.getName();
          if (fname.contains(baseName) && fname.endswith(suffix)) {
            targetFunc = candidate;
            break;
          }
        }

        if (!targetFunc) {
          extractOp.emitError("Failed to find existing function for builtin ")
              << baseName << suffix;
          continue;
        }

        builder.setInsertionPoint(extractOp);
        auto loc = extractOp.getLoc();
        auto i64Ty = builder.getI64Type();

        auto call = builder.create<LLVM::CallOp>(
            loc, i64Ty, FlatSymbolRefAttr::get(context, targetFunc.getName()),
            ValueRange{});

        extractOp.replaceAllUsesWith(call.getResult());
        extractOp.erase();
      }
    }
  }
}

/// Create a pass to convert SPIRVBuiltIn operations to NVVM calls.
/// This pass will
/// 1.Mark the nvvm kernel function.
/// 2.rewrite SPIRVBuiltIn functions to NVVM intrinsic calls
/// 3.replace accesses to SPIRVBuiltIn variables with calls to the rewritten
/// functions.
void ConvertSPIRVBuiltInToNVVMPass::runOnOperation() {
  auto *context = &getContext();
  auto module = getOperation();

  markNVVMKernelFunctions(module);

  rewriteSPIRVBuiltinFunctions(module, context);
  replaceBuiltinAccessWithCalls(module, context);
}
