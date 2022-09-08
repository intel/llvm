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
        {Kind::Accessor, "accessor"},
        {Kind::Id, "id"},
        {Kind::Range, "range"},
        {Kind::Unknown, "unknown"},
};

std::map<std::string, SYCLFuncDescriptor::Kind>
    SYCLFuncDescriptor::Id::nameToKind = {
        {"accessor", Kind::Accessor},
        {"id", Kind::Id},
        {"range", Kind::Range},
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
// SYCLAccessorFuncDescriptor
//===----------------------------------------------------------------------===//

bool SYCLAccessorFuncDescriptor::isValid(FuncId funcId) const {
  switch (funcId) {
  case FuncId::AccessorInt1ReadWriteGlobalBufferFalseCtorDefault:
  case FuncId::AccessorInt1ReadWriteGlobalBufferFalseInit:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// SYCLIdFuncDescriptor
//===----------------------------------------------------------------------===//

bool SYCLIdFuncDescriptor::isValid(SYCLFuncDescriptor::FuncId funcId) const {
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
// SYCLRangeFuncDescriptor
//===----------------------------------------------------------------------===//

bool SYCLRangeFuncDescriptor::isValid(FuncId funcId) const {
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

  declareAccessorFuncDescriptors(converter, module, builder);
  declareIdFuncDescriptors(converter, module, builder);
  declareRangeFuncDescriptors(converter, module, builder);
}

void SYCLFuncRegistry::declareAccessorFuncDescriptors(
    LLVMTypeConverter &converter, ModuleOp &module, OpBuilder &builder) {
  MLIRContext *context = module.getContext();
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto i32Ty = IntegerType::get(context, 32);
  auto i32PtrTy = converter.convertType(MemRefType::get(-1, i32Ty));

  constexpr unsigned int dim = 1;
  Type id1SYCLTy = IDType::get(context, dim);
  Type range1SYCLTy = RangeType::get(context, dim);
  Type accessorImplDeviceSYCLTy = AccessorImplDeviceType::get(
      context, dim, {id1SYCLTy, range1SYCLTy, range1SYCLTy});
  Type accessorInt1ReadWriteGlobalBufferSYCLTy = AccessorType::get(
      context, i32Ty, dim, MemoryAccessMode::ReadWrite,
      MemoryTargetMode::GlobalBuffer, {accessorImplDeviceSYCLTy});
  Type accessorInt1ReadWriteGlobalBufferPtrTy = converter.convertType(
      MemRefType::get(-1, accessorInt1ReadWriteGlobalBufferSYCLTy));

  Type id1Ty = converter.convertType(id1SYCLTy);
  Type range1Ty = converter.convertType(range1SYCLTy);

  // Construct the SYCL functions descriptors for the sycl::accessor<n> type.
  // Descriptor format: (enum, function name, signature).
  // clang-format off
  std::vector<SYCLFuncDescriptor> descriptors = {
      // sycl::accessor<int, 1, read_write, global_buffer, (placeholder)0>::
      //   accessor()
      SYCLAccessorFuncDescriptor(
          FuncId::AccessorInt1ReadWriteGlobalBufferFalseCtorDefault,
          "_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_"
          "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_"
          "property_listIJEEEEC2Ev",
          voidTy, {accessorInt1ReadWriteGlobalBufferPtrTy}),
      // sycl::accessor<int, 1, read_write, global_buffer, (placeholder)0>::
      //   __init(int AS1*, sycl::range<1>, sycl::range<1>, sycl::id<1>)
      SYCLAccessorFuncDescriptor(
          FuncId::AccessorInt1ReadWriteGlobalBufferFalseInit,
          "_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_"
          "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_"
          "property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_"
          "2idILi1EEE",
          voidTy, {accessorInt1ReadWriteGlobalBufferPtrTy, i32PtrTy, range1Ty, range1Ty, id1Ty}),
  };
  // clang-format on

  // Declare sycl::id<n> function descriptors and add them to the registry.
  declareFuncDescriptors(descriptors, module, builder);
}

void SYCLFuncRegistry::declareIdFuncDescriptors(LLVMTypeConverter &converter,
                                                ModuleOp &module,
                                                OpBuilder &builder) {
  MLIRContext *context = module.getContext();
  Type id1PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 1)));
  Type id2PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 2)));
  Type id3PtrTy =
      converter.convertType(MemRefType::get(-1, IDType::get(context, 3)));
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto i64Ty = IntegerType::get(context, 64);

  // Construct the SYCL functions descriptors for the sycl::id<n> type.
  // Descriptor format: (enum, function name, signature).
  // clang-format off
  std::vector<SYCLFuncDescriptor> descriptors = {
      // sycl::id<1>::id()
      SYCLIdFuncDescriptor(FuncId::Id1CtorDefault,
          "_ZN2cl4sycl2idILi1EEC2Ev", voidTy, {id1PtrTy}),
      // sycl::id<2>::id()
      SYCLIdFuncDescriptor(FuncId::Id2CtorDefault,
          "_ZN2cl4sycl2idILi2EEC2Ev", voidTy, {id2PtrTy}),
      // sycl::id<3>::id()
      SYCLIdFuncDescriptor(FuncId::Id3CtorDefault,
          "_ZN2cl4sycl2idILi3EEC2Ev", voidTy, {id3PtrTy}),

      // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type)
      SYCLIdFuncDescriptor(FuncId::Id1CtorSizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE",
          voidTy, {id1PtrTy, i64Ty}),
      // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type)
      SYCLIdFuncDescriptor(FuncId::Id2CtorSizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE",
          voidTy, {id2PtrTy, i64Ty}),
      // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type)
      SYCLIdFuncDescriptor(FuncId::Id3CtorSizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE",
          voidTy, {id3PtrTy, i64Ty}),

      // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
      SYCLIdFuncDescriptor(FuncId::Id1Ctor2SizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm",
          voidTy, {id1PtrTy, i64Ty, i64Ty}),
      // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)
      SYCLIdFuncDescriptor(FuncId::Id2Ctor2SizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm",
          voidTy, {id2PtrTy, i64Ty, i64Ty}),
      // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
      SYCLIdFuncDescriptor(FuncId::Id3Ctor2SizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm",
          voidTy, {id3PtrTy, i64Ty, i64Ty}),      

      // sycl::id<1>::id<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
      SYCLIdFuncDescriptor(FuncId::Id1Ctor3SizeT,
          "_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm",
          voidTy, {id1PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::id<2>::id<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
      SYCLIdFuncDescriptor(FuncId::Id2Ctor3SizeT,
          "_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm",
          voidTy, {id2PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::id<3>::id<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
      SYCLIdFuncDescriptor(FuncId::Id3Ctor3SizeT,
          "_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm",
          voidTy, {id3PtrTy, i64Ty, i64Ty, i64Ty}),
      
      // sycl::id<1>::id(sycl::id<1> const&)
      SYCLIdFuncDescriptor(FuncId::Id1CopyCtor,
          "_ZN2cl4sycl2idILi1EEC1ERKS2_", voidTy, {id1PtrTy, id1PtrTy}),
      // sycl::id<2>::id(sycl::id<2> const&)
      SYCLIdFuncDescriptor(FuncId::Id2CopyCtor,
          "_ZN2cl4sycl2idILi2EEC1ERKS2_", voidTy, {id2PtrTy, id2PtrTy}),
      // sycl::id<3>::id(sycl::id<3> const&)
      SYCLIdFuncDescriptor(FuncId::Id3CopyCtor,
          "_ZN2cl4sycl2idILi3EEC1ERKS2_", voidTy, {id3PtrTy, id3PtrTy}),
  };
  // clang-format on

  // Declare sycl::id<n> function descriptors and add them to the registry.
  declareFuncDescriptors(descriptors, module, builder);
}

void SYCLFuncRegistry::declareRangeFuncDescriptors(LLVMTypeConverter &converter,
                                                   ModuleOp &module,
                                                   OpBuilder &builder) {
  MLIRContext *context = module.getContext();
  Type range1PtrTy =
      converter.convertType(MemRefType::get(-1, RangeType::get(context, 1)));
  Type range2PtrTy =
      converter.convertType(MemRefType::get(-1, RangeType::get(context, 2)));
  Type range3PtrTy =
      converter.convertType(MemRefType::get(-1, RangeType::get(context, 3)));
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto i64Ty = IntegerType::get(context, 64);

  // Construct the SYCL functions descriptors for the sycl::range<n> type.
  // clang-format off
  std::vector<SYCLFuncDescriptor> descriptors = {
      // sycl::range<1>::range()
      SYCLRangeFuncDescriptor(FuncId::Range1CtorDefault,
          "_ZN2cl4sycl5rangeILi1EEC2Ev", voidTy, {range1PtrTy}),
      // sycl::range<2>::range()
      SYCLRangeFuncDescriptor(FuncId::Range2CtorDefault,
          "_ZN2cl4sycl5rangeILi2EEC2Ev", voidTy, {range2PtrTy}),
      // sycl::range<3>::range()
      SYCLRangeFuncDescriptor(FuncId::Range3CtorDefault,
          "_ZN2cl4sycl5rangeILi3EEC2Ev", voidTy, {range3PtrTy}),

      // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type)
      SYCLRangeFuncDescriptor(FuncId::Range1CtorSizeT,
          "_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE", 
          voidTy, {range1PtrTy, i64Ty}),
      // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type)
      SYCLRangeFuncDescriptor(FuncId::Range2CtorSizeT,
          "_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE", 
          voidTy, {range2PtrTy, i64Ty}),
      // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type)
      SYCLRangeFuncDescriptor(FuncId::Range3CtorSizeT,
          "_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE", 
          voidTy, {range3PtrTy, i64Ty}),

      // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long)
      SYCLRangeFuncDescriptor(FuncId::Range1Ctor2SizeT,
          "_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm", 
          voidTy, {range1PtrTy, i64Ty, i64Ty}),
      // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long)                         
      SYCLRangeFuncDescriptor(FuncId::Range2Ctor2SizeT,
          "_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm", 
          voidTy, {range2PtrTy, i64Ty, i64Ty}),
      // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long)
      SYCLRangeFuncDescriptor(FuncId::Range3Ctor2SizeT,
          "_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm", 
          voidTy, {range3PtrTy, i64Ty, i64Ty}),

      // sycl::range<1>::range<1>(std::enable_if<(1)==(1), unsigned long>::type, unsigned long, unsigned long)
      SYCLRangeFuncDescriptor(FuncId::Range1Ctor3SizeT,
          "_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm", 
          voidTy, {range1PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::range<2>::range<2>(std::enable_if<(2)==(2), unsigned long>::type, unsigned long, unsigned long)
      SYCLRangeFuncDescriptor(FuncId::Range2Ctor3SizeT,
          "_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm", 
          voidTy, {range2PtrTy, i64Ty, i64Ty, i64Ty}),
      // sycl::range<3>::range<3>(std::enable_if<(3)==(3), unsigned long>::type, unsigned long, unsigned long)
      SYCLRangeFuncDescriptor(FuncId::Range3Ctor3SizeT,
          "_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm", 
          voidTy, {range3PtrTy, i64Ty, i64Ty, i64Ty}),

      // sycl::range<1>::range(sycl::range<1> const&)
      SYCLRangeFuncDescriptor(FuncId::Range1CopyCtor,
          "_ZN2cl4sycl5rangeILi1EEC1ERKS2_", voidTy, {range1PtrTy, range1PtrTy}),
      // sycl::range<2>::range(sycl::range<2> const&)
      SYCLRangeFuncDescriptor(FuncId::Range2CopyCtor,
          "_ZN2cl4sycl5rangeILi2EEC1ERKS2_", voidTy, {range2PtrTy, range2PtrTy}),
      // sycl::range<3>::range(sycl::range<3> const&)
      SYCLRangeFuncDescriptor(FuncId::Range3CopyCtor,
          "_ZN2cl4sycl5rangeILi3EEC1ERKS2_", voidTy, {range3PtrTy, range3PtrTy}),
  };
  // clang-format on

  // Declare sycl::range<n> function descriptors and add them to the registry.
  declareFuncDescriptors(descriptors, module, builder);
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
