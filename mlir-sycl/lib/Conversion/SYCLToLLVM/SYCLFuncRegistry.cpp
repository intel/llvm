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
// SYCLFuncDescriptor::Id
//===----------------------------------------------------------------------===//

std::map<SYCLFuncDescriptor::FuncKind, std::string>
    SYCLFuncDescriptor::Id::funcKindToName = {
        {FuncKind::IdCtor, "idCtor"},
        {FuncKind::RangeCtor, "rangeCtor"},
        {FuncKind::Unknown, "unknown"},
};

std::map<std::string, SYCLFuncDescriptor::FuncKind>
    SYCLFuncDescriptor::Id::nameToFuncKind = {
        {"idCtor", FuncKind::IdCtor},
        {"rangeCtor", FuncKind::RangeCtor},
        {"unknown", FuncKind::Unknown},
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
// SYCLIdCtorDescriptor
//===----------------------------------------------------------------------===//

bool SYCLIdCtorDescriptor::isValid(SYCLFuncDescriptor::FuncId funcId) const {
  switch (funcId) {
  case FuncId::Id1CtorDefault:
  case FuncId::Id2CtorDefault:
  case FuncId::Id3CtorDefault:
  case FuncId::Id1CtorSizeT:
  case FuncId::Id2CtorSizeT:
  case FuncId::Id3CtorSizeT:
  case FuncId::Id1Ctor2SizeT:
  case FuncId::Id2Ctor2SizeT:
  case FuncId::Id3Ctor2SizeT:
  case FuncId::Id1Ctor3SizeT:
  case FuncId::Id2Ctor3SizeT:
  case FuncId::Id3Ctor3SizeT:
  case FuncId::Id1CopyCtor:
  case FuncId::Id2CopyCtor:
  case FuncId::Id3CopyCtor:
    return true;  
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// SYCLRangeCtorDescriptor
//===----------------------------------------------------------------------===//

bool SYCLRangeCtorDescriptor::isValid(FuncId funcId) const {
  switch (funcId) {
  case FuncId::Range1CtorDefault:
  case FuncId::Range2CtorDefault:
  case FuncId::Range3CtorDefault:
  case FuncId::Range1CtorSizeT:
  case FuncId::Range2CtorSizeT:
  case FuncId::Range3CtorSizeT:
  case FuncId::Range1Ctor2SizeT:
  case FuncId::Range2Ctor2SizeT:
  case FuncId::Range3Ctor2SizeT:
  case FuncId::Range1Ctor3SizeT:
  case FuncId::Range2Ctor3SizeT:
  case FuncId::Range3Ctor3SizeT:
  case FuncId::Range1CopyCtor:
  case FuncId::Range2CopyCtor:
  case FuncId::Range3CopyCtor:
    return true;
  default:
    return false;
  }
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
SYCLFuncRegistry::getFuncId(SYCLFuncDescriptor::FuncKind funcKind, Type retType,
                            TypeRange argTypes) const {
  assert(funcKind != SYCLFuncDescriptor::FuncKind::Unknown &&
         "Invalid funcKind");
  LLVM_DEBUG(llvm::dbgs() << "Looking up function of kind: "
                          << SYCLFuncDescriptor::Id::funcKindToName[funcKind]
                          << "\n";);

  for (const auto &entry : registry) {
    const SYCLFuncDescriptor &desc = entry.second;
    LLVM_DEBUG(llvm::dbgs() << desc << "\n");

    // Skip through entries that do not match the requested funcIdKind.
    if (desc.descId.funcKind != funcKind) {
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
      LLVM_DEBUG(llvm::dbgs() << "\tskip, number of arguments does not match\n");
      continue;
    }
    if (!std::equal(argTypes.begin(), argTypes.end(), desc.argTys.begin())) {
      LLVM_DEBUG(llvm::dbgs() << "\tskip, arguments types do not match\n");      
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

  Type id1PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 1)));
  Type id2PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 2)));
  Type id3PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 3)));
  Type range1PtrTy =
      converter.convertType(MemRefType::get(-1, RangeType::get(context, 1)));
  Type range2PtrTy =
      converter.convertType(MemRefType::get(-1, RangeType::get(context, 2)));
  Type range3PtrTy =
      converter.convertType(MemRefType::get(-1, RangeType::get(context, 3)));

  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto i64Ty = IntegerType::get(context, 64);

  // Construct the SYCL functions descriptors for the sycl::id<n> type.
  // Descriptor format: (enum, function name, signature).
  // clang-format off
  std::vector<SYCLIdCtorDescriptor> idDescriptors = {
      // sycl::id<1>::id()
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id1CtorDefault,
          "_ZN2cl4sycl2idILi1EEC2Ev", voidTy, {id1PtrTy}),
      // sycl::id<2>::id()
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id2CtorDefault,
          "_ZN2cl4sycl2idILi2EEC2Ev", voidTy, {id2PtrTy}),
      // sycl::id<3>::id()
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id3CtorDefault,
          "_ZN2cl4sycl2idILi3EEC2Ev", voidTy, {id3PtrTy}),

      // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id1CtorSizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE",
          voidTy, {id1PtrTy, i64Ty}),
      // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id2CtorSizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE",
          voidTy, {id2PtrTy, i64Ty}),
      // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id3CtorSizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE",
          voidTy, {id3PtrTy, i64Ty}),

      // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id1Ctor2SizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm",
          voidTy, {id1PtrTy, i64Ty, i64Ty}),
      // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id2Ctor2SizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm",
          voidTy, {id2PtrTy, i64Ty, i64Ty}),
      // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id3Ctor2SizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm",
          voidTy, {id3PtrTy, i64Ty, i64Ty}),      

      // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id1Ctor3SizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm",
          voidTy, {id1PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id2Ctor3SizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm",
          voidTy, {id2PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id3Ctor3SizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm",
          voidTy, {id3PtrTy, i64Ty, i64Ty, i64Ty}),

      // sycl::id<1>::id(sycl::id<1> const&)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id1CopyCtor,
          "_ZN2cl4sycl2idILi1EEC1ERKS2_", voidTy, {id1PtrTy, id1PtrTy}),
      // sycl::id<2>::id(sycl::id<2> const&)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id2CopyCtor,
          "_ZN2cl4sycl2idILi2EEC1ERKS2_", voidTy, {id2PtrTy, id2PtrTy}),
      // sycl::id<3>::id(sycl::id<3> const&)
      SYCLIdCtorDescriptor(SYCLFuncDescriptor::FuncId::Id3CopyCtor,
          "_ZN2cl4sycl2idILi3EEC1ERKS2_", voidTy, {id3PtrTy, id3PtrTy}),
  };
  // clang-format on

  // Declare sycl::id<n> ctors and add them to the registry.
  for (SYCLIdCtorDescriptor &idCtor : idDescriptors) {
    idCtor.declareFunction(module, builder);
    registry.emplace(idCtor.descId.funcId, idCtor);
  }

  // Construct the SYCL functions descriptors for the sycl::range<n> type.
  // clang-format off  
  std::vector<SYCLRangeCtorDescriptor> rangeDescriptors = {
      // sycl::range<1>::range()
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range1CtorDefault,
          "_ZN2cl4sycl5rangeILi1EEC2Ev", voidTy, {range1PtrTy}),
      // sycl::range<2>::range()
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range2CtorDefault,
          "_ZN2cl4sycl5rangeILi2EEC2Ev", voidTy, {range2PtrTy}),
      // sycl::range<3>::range()
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range3CtorDefault,
          "_ZN2cl4sycl5rangeILi3EEC2Ev", voidTy, {range3PtrTy}),

      // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range1CtorSizeT,
          "_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE", 
          voidTy, {range1PtrTy, i64Ty}),
      // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range2CtorSizeT,
          "_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE", 
          voidTy, {range2PtrTy, i64Ty}),
      // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range3CtorSizeT,
          "_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE", 
          voidTy, {range3PtrTy, i64Ty}),

      // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range1Ctor2SizeT,
          "_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm", 
          voidTy, {range1PtrTy, i64Ty, i64Ty}),
      // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)                         
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range2Ctor2SizeT,
          "_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm", 
          voidTy, {range2PtrTy, i64Ty, i64Ty}),
      // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range3Ctor2SizeT,
          "_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm", 
          voidTy, {range3PtrTy, i64Ty, i64Ty}),

      // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range1Ctor3SizeT,
          "_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm", 
          voidTy, {range1PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range2Ctor3SizeT,
          "_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm", 
          voidTy, {range2PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range3Ctor3SizeT,
          "_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm", 
          voidTy, {range3PtrTy, i64Ty, i64Ty, i64Ty}),

      // sycl::range<1>::range(sycl::range<1> const&)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range1CopyCtor,
          "_ZN2cl4sycl5rangeILi1EEC1ERKS2_", voidTy, {range1PtrTy, range1PtrTy}),
      // sycl::range<2>::range(sycl::range<2> const&)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range2CopyCtor,
          "_ZN2cl4sycl5rangeILi2EEC1ERKS2_", voidTy, {range2PtrTy, range2PtrTy}),
      // sycl::range<3>::range(sycl::range<3> const&)
      SYCLRangeCtorDescriptor(SYCLFuncDescriptor::FuncId::Range3CopyCtor,
          "_ZN2cl4sycl5rangeILi3EEC1ERKS2_", voidTy, {range3PtrTy, range3PtrTy}),
  };
  // clang-format on

  // Declare sycl::range<n> ctors and add them to the registry.
  for (SYCLRangeCtorDescriptor &rangeCtor : rangeDescriptors) {
    rangeCtor.declareFunction(module, builder);
    registry.emplace(rangeCtor.descId.funcId, rangeCtor);
  }  
}
