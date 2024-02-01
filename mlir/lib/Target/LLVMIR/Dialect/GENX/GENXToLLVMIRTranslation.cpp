//===- GENXToLLVMIRTranslation.cpp - Translate GENX to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR GENX dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/GENX/GENXToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/Dialect/GENX/GenIntrinsics.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

// Create a call to SPIR device function.
static llvm::CallInst *createDeviceFunctionCall(llvm::IRBuilderBase &builder,
                                                StringRef fnName,
                                                llvm::Type *retType,
                                                ArrayRef<llvm::Type *> argTypes,
                                                ArrayRef<llvm::Value *> args,
                                                bool convergent = false) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  auto *functionType =
      llvm::FunctionType::get(retType, argTypes, /*isVarArg*/ false);
  auto *fn = dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fnName, functionType).getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  if (convergent)
    fn->setConvergent();
  auto *ci = builder.CreateCall(fn, args);
  if (convergent)
    ci->setConvergent();
  return ci;
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

static llvm::CallInst *createSubGroupShuffle(llvm::IRBuilderBase &builder,
                                             llvm::Value *value,
                                             llvm::Value *mask,
                                             GENX::ShflKind kind) {
  assert(mask->getType()->isIntegerTy(32) && "Expecting mask type to be i32");

  std::string fnName = "";
  switch (kind) {
  case GENX::ShflKind::XOR:
    fnName = "_Z21sub_group_shuffle_xor";
    break;
  case GENX::ShflKind::UP:
    fnName = "_Z20sub_group_shuffle_up";
    break;
  case GENX::ShflKind::DOWN:
    fnName = "_Z22sub_group_shuffle_down";
    break;
  case GENX::ShflKind::IDX:
    fnName = "_Z17sub_group_shuffle";
    break;
  }

  llvm::Type *ty = value->getType();
  if (ty->isHalfTy())
    fnName += "Dh";
  else if (ty->isFloatTy())
    fnName += "f";
  else if (ty->isDoubleTy())
    fnName += "d";
  else if (ty->isIntegerTy(8))
    fnName += "c";
  else if (ty->isIntegerTy(16))
    fnName += "s";
  else if (ty->isIntegerTy(32))
    fnName += "i";
  else if (ty->isIntegerTy(64))
    fnName += "l";
  else
    llvm_unreachable("unhandled type");

  fnName += "j";

  return createDeviceFunctionCall(builder, fnName, value->getType(),
                                  {value->getType(), mask->getType()},
                                  {value, mask}, true /*convergent*/);
}

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

static llvm::CallInst *
createGenISAFpToFp(GENX::FpToFpOp op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) {
  // TODO: Remove GENX::RoundingMode and use llvm::RoundingMode directly.
  llvm::RoundingMode rounding;
  switch (op.getRoundingMode()) {
  case GENX::RoundingMode::RTE:
    rounding = llvm::RoundingMode::NearestTiesToEven;
    break;
  case GENX::RoundingMode::RTN:
    rounding = llvm::RoundingMode::TowardNegative;
    break;
  case GENX::RoundingMode::RTP:
    rounding = llvm::RoundingMode::TowardPositive;
    break;
  case GENX::RoundingMode::RTZ:
    rounding = llvm::RoundingMode::TowardZero;
    break;
  default:
    llvm_unreachable("Unhandled rounding mode");
  }

  SmallVector<llvm::Value *> args = {
      moduleTranslation.lookupValue(op.getArg())};

  llvm::Type *resTy = moduleTranslation.convertType(op->getResultTypes()[0]);
  unsigned resTySizeInBits = resTy->getScalarSizeInBits();
  unsigned srcTySizeInBits = args[0]->getType()->getScalarSizeInBits();
  // TODO: Add verifier.
  assert(srcTySizeInBits != resTySizeInBits &&
         "Expecting first argument and result size to be different");
  llvm::Intrinsic::ID id;
  if (srcTySizeInBits > resTySizeInBits)
    id = llvm::Intrinsic::experimental_constrained_fptrunc;
  else
    id = llvm::Intrinsic::experimental_constrained_fpext;

  return builder.CreateConstrainedFPCast(id, args[0], resTy, nullptr, "",
                                         nullptr, rounding);
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

static llvm::CallInst *
createGenISADPAS(GENX::MatrixDPASOp op, llvm::IRBuilderBase &builder,
                 LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  TypeRange opTypes = op->getOperandTypes();

  llvm::Value *a = moduleTranslation.lookupValue(op.getA());
  auto *aTy = llvm::FixedVectorType::get(builder.getInt16Ty(), op.getRc());
  if (a->getType() != aTy)
    a = builder.CreateBitCast(a, aTy);

  llvm::Value *b = moduleTranslation.lookupValue(op.getB());
  auto *bTy = llvm::FixedVectorType::get(builder.getInt32Ty(), 8);
  if (b->getType() != bTy)
    b = builder.CreateBitCast(b, bTy);

  llvm::Function *fn = llvm::GenISAIntrinsic::getDeclaration(
      module, llvm::GenISAIntrinsic::GenISA_sub_group_dpas,
      {moduleTranslation.convertType(op->getResultTypes()[0]),
       moduleTranslation.convertType(opTypes[0]), aTy, bTy});
  assert(fn && "GenISAIntrinsic::getDeclaration() returns NULL");

  SmallVector<llvm::Value *> args;
  args.push_back(moduleTranslation.lookupValue(op.getC()));
  args.push_back(a);
  args.push_back(b);

  auto *int32Ty = builder.getInt32Ty();
  auto *int1Ty = builder.getInt1Ty();
  args.push_back(llvm::ConstantInt::get(int32Ty, static_cast<int>(op.getPa())));
  args.push_back(llvm::ConstantInt::get(int32Ty, static_cast<int>(op.getPb())));
  args.push_back(llvm::ConstantInt::get(int32Ty, 8 /* systolic depth */));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getRc()));
  args.push_back(llvm::ConstantInt::get(int1Ty, false));

  return builder.CreateCall(fn, args);
}

static llvm::CallInst *
createGenISA2DBlockRead(GENX::Matrix2DBlockLoadOp op,
                        llvm::IRBuilderBase &builder,
                        LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::GenISAIntrinsic::getDeclaration(
      module, llvm::GenISAIntrinsic::GenISA_LSC2DBlockRead,
      {moduleTranslation.convertType(op->getResultTypes()[0])});
  assert(fn && "GenISAIntrinsic::getDeclaration() returns NULL");

  SmallVector<llvm::Value *> args(
      moduleTranslation.lookupValues(op.getOperands()));

  // The IGC intrinsic requires the first argument be int64
  assert(isa<llvm::PointerType>(args[0]->getType()) &&
         "Expecting a pointer type");
  args[0] = builder.CreatePointerCast(args[0], builder.getInt64Ty());

  auto *int32Ty = builder.getInt32Ty();
  auto *int1Ty = builder.getInt1Ty();
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getElemSizeInBits()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileWidth()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileHeight()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getVBlocks()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getTranspose()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getVnniTransform()));
  // FIXME: Add argument to control cache.
  args.push_back(llvm::ConstantInt::get(int32Ty, 0));

  return builder.CreateCall(fn, args);
}

static llvm::CallInst *
createGenISA2DBlockWrite(GENX::Matrix2DBlockStoreOp op,
                         llvm::IRBuilderBase &builder,
                         LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  TypeRange opTypes = op->getOperandTypes();
  llvm::Function *fn = llvm::GenISAIntrinsic::getDeclaration(
      module, llvm::GenISAIntrinsic::GenISA_LSC2DBlockWrite,
      {moduleTranslation.convertType(opTypes[opTypes.size() - 1])});
  assert(fn && "GenISAIntrinsic::getDeclaration() returns NULL");

  SmallVector<llvm::Value *> args(
      moduleTranslation.lookupValues(op.getOperands()));

  // The IGC intrinsic requires the first argument be int64
  assert(isa<llvm::PointerType>(args[0]->getType()) &&
         "Expecting a pointer type");
  args[0] = builder.CreatePointerCast(args[0], builder.getInt64Ty());
  llvm::Value *lastOperand = args.pop_back_val();

  auto *int32Ty = builder.getInt32Ty();
  auto *int1Ty = builder.getInt1Ty();
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getElemSizeInBits()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileWidth()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getTileHeight()));
  args.push_back(llvm::ConstantInt::get(int32Ty, op.getVBlocks()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getTranspose()));
  args.push_back(llvm::ConstantInt::get(int1Ty, op.getVnniTransform()));
  // FIXME: Add argument to control cache.
  args.push_back(llvm::ConstantInt::get(int32Ty, 0));
  args.push_back(lastOperand);

  return builder.CreateCall(fn, args);
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the GENX dialect to LLVM IR.
class GENXDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/GENXConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    StringAttr attrName = attribute.getName();
    Attribute attrVal = attribute.getValue();

    // Set calling convention for kernel
    if (attrName == GENX::GENXDialect::getKernelFuncAttrName())
      llvmFunc->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

    auto attachMetadata = [&](StringRef name) {
      SmallVector<llvm::Metadata *, 3> metadata;
      llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
      for (int64_t i : extractFromIntegerArrayAttr<int64_t>(attrVal)) {
        llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
        metadata.push_back(llvm::ConstantAsMetadata::get(constant));
      }
      llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
      llvmFunc->setMetadata(name, node);
    };

    // Set max_work_group_size metadata.
    if (attrName == GENX::GENXDialect::getMaxWorkGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("max_work_group_size");
    }

    // Set reqd_work_group_size metadata.
    if (attrName == GENX::GENXDialect::getReqdWorkGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("reqd_work_group_size");
    }

    // Set intel_reqd_sub_group_size metadata.
    if (attrName == GENX::GENXDialect::getReqdSubGroupSizeAttrName()) {
      if (!attrVal.dyn_cast<ArrayAttr>())
        return failure();

      attachMetadata("intel_reqd_sub_group_size");
    }

    return success();
  }
};
} // namespace

void mlir::registerGENXDialectTranslation(DialectRegistry &registry) {
  registry.insert<GENX::GENXDialect>();
  registry.addExtension(+[](MLIRContext *ctx, GENX::GENXDialect *dialect) {
    dialect->addInterfaces<GENXDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerGENXDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerGENXDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
