#include "Builtins.h"

#include "Kernel.h"
#include "NDRangesHelper.h"
#include "target/TargetFusionInfo.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace jit_compiler;

unsigned Remapper::getDefaultValue(BuiltinKind K) const {
  switch (K) {
  case BuiltinKind::GlobalSizeRemapper:
  case BuiltinKind::LocalSizeRemapper:
  case BuiltinKind::NumWorkGroupsRemapper:
    return 1;
  case BuiltinKind::GlobalIDRemapper:
  case BuiltinKind::LocalIDRemapper:
  case BuiltinKind::GroupIDRemapper:
  case BuiltinKind::GlobalOffsetRemapper:
    return 0;
  }
  llvm_unreachable("Unhandled kind");
}

static raw_ostream &operator<<(raw_ostream &Os, const Indices &I) {
  return Os << I[0] << "_" << I[1] << "_" << I[2];
}

static raw_ostream &operator<<(raw_ostream &Os, const NDRange &ND) {
  return Os << ND.getDimensions() << "_" << ND.getGlobalSize() << "_"
            << ND.getLocalSize();
}

std::string Remapper::getFunctionName(BuiltinKind K, const NDRange &SrcNDRange,
                                      const NDRange &FusedNDRange,
                                      uint32_t Idx) {
  std::string Res;
  raw_string_ostream S{Res};
  S << "__" <<
      [K]() {
        switch (K) {
        case BuiltinKind::GlobalSizeRemapper:
          return "global_size";
        case BuiltinKind::LocalSizeRemapper:
          return "local_size";
        case BuiltinKind::NumWorkGroupsRemapper:
          return "num_work_groups";
        case BuiltinKind::GlobalIDRemapper:
          return "global_id";
        case BuiltinKind::LocalIDRemapper:
          return "local_id";
        case BuiltinKind::GroupIDRemapper:
          return "group_id";
        case BuiltinKind::GlobalOffsetRemapper:
          return "global_offset";
        }
        llvm_unreachable("Unhandled kind");
      }()
    << "_remapper_" << SrcNDRange << "_" << FusedNDRange;
  if (Idx != (uint32_t)-1)
    S << "_" << static_cast<char>('x' + Idx);
  return S.str();
}

/// Mirrors getters arguments depending on the input dimension.
static uint32_t mirror(int Dimensions, uint32_t I) {
  switch (Dimensions) {
  case 1:
    // No change
    assert(I < 3 && "Invalid index");
    return I;
  case 2:
    // X and Y are swapped
    switch (I) {
    case 0:
      return 1;
    case 1:
      return 0;
    case 2:
      return 2;
    }
    llvm_unreachable("Invalid index");
  case 3:
    // X and Z are swapped
    switch (I) {
    case 0:
      return 2;
    case 1:
      return 1;
    case 2:
      return 0;
    }
    llvm_unreachable("Invalid index");
  }
  llvm_unreachable("Invalid number of dimensions");
}

static std::string
getGetGlobalLinearIDFunctionName(const NDRange &FusedNDRange) {
  std::string Res;
  raw_string_ostream S{Res};
  S << "__global_linear_id_" << FusedNDRange;
  return S.str();
}

static Function *
getOrCreateGetGlobalLinearIDFunction(const TargetFusionInfo &TargetInfo,
                                     const NDRange &FusedNDRange, Module *M) {
  const auto Name = getGetGlobalLinearIDFunctionName(FusedNDRange);

  auto *F = M->getFunction(Name);
  if (F) {
    // Already created function.
    return F;
  }

  auto &Context = M->getContext();
  IRBuilder<> Builder{Context};
  const auto N = TargetInfo.getIndexSpaceBuiltinBitwidth();
  auto *Ty = FunctionType::get(Builder.getIntNTy(N), /*isVarArg*/ false);
  F = Function::Create(Ty, Function::LinkageTypes::InternalLinkage, Name, M);
  TargetInfo.setMetadataForGeneratedFunction(F);
  F->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);

  auto *EntryBlock = BasicBlock::Create(Context, "entry", F);
  Builder.SetInsertPoint(EntryBlock);

  // See:
  // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:multi-dim-linearization
  auto *Res = [&Builder, &TargetInfo, &FusedNDRange, N] {
    const auto Dimensions = FusedNDRange.getDimensions();
    const auto GetGS = [&FusedNDRange](uint32_t I) {
      return FusedNDRange.getGlobalSize()[I];
    };
    const auto GetID = [&Builder, &TargetInfo, &FusedNDRange,
                        Dimensions](uint32_t I) {
      return TargetInfo.getGlobalIDWithoutOffset(Builder, FusedNDRange,
                                                 mirror(Dimensions, I));
    };
    const auto GetConst = [&Builder, N](uint64_t C) {
      return Builder.getIntN(N, C);
    };
    switch (Dimensions) {
    case 1:
      // gid = id0
      return GetID(0);
    case 2: {
      // gid = id1 + (id0 * r1)
      auto *ID1 = GetID(1);
      return Builder.CreateAdd(ID1,
                               Builder.CreateMul(GetID(0), GetConst(GetGS(1))));
    }
    case 3: {
      // gid = (id0 * r1 * r2) + (id1 * r2) + id2
      auto *C0 = Builder.CreateMul(GetID(0), GetConst(GetGS(1) * GetGS(2)));
      auto *C1 = Builder.CreateAdd(
          Builder.CreateMul(GetID(1), GetConst(GetGS(2))), C0);
      return Builder.CreateAdd(GetID(2), C1);
    }
    default:
      llvm_unreachable("Invalid number of dimensions");
    }
  }();

  Builder.CreateRet(Res);

  return F;
}

Value *jit_compiler::getGlobalLinearID(IRBuilderBase &Builder,
                                       const TargetFusionInfo &TargetInfo,
                                       const NDRange &FusedNDRange) {
  auto *F = getOrCreateGetGlobalLinearIDFunction(
      TargetInfo, FusedNDRange,
      Builder.GetInsertBlock()->getParent()->getParent());
  auto *C = Builder.CreateCall(F);
  C->setAttributes(F->getAttributes());
  C->setCallingConv(F->getCallingConv());
  return C;
}

static Value *generateGetSizeCase(IRBuilderBase &Builder,
                                  const TargetFusionInfo &TargetInfo,
                                  const Indices &SrcIndices, int Dimensions,
                                  uint32_t Index) {
  auto N = TargetInfo.getIndexSpaceBuiltinBitwidth();
  return Builder.getIntN(N, SrcIndices[mirror(Dimensions, Index)]);
}

/// Input argument
static Value *generateGetGlobalSizeCase(IRBuilderBase &Builder,
                                        const TargetFusionInfo &TargetInfo,
                                        const NDRange &SrcNDRange,
                                        uint32_t Index) {
  return generateGetSizeCase(Builder, TargetInfo, SrcNDRange.getGlobalSize(),
                             SrcNDRange.getDimensions(), Index);
}

/// Input argument
static Value *generateGetGlobalOffsetCase(IRBuilderBase &Builder,
                                          const TargetFusionInfo &TargetInfo,
                                          const NDRange &SrcNDRange,
                                          uint32_t Index) {
  return generateGetSizeCase(Builder, TargetInfo, SrcNDRange.getOffset(),
                             SrcNDRange.getDimensions(), Index);
}

/// Input argument
static Value *generateGetLocalSizeCase(IRBuilderBase &Builder,
                                       const TargetFusionInfo &TargetInfo,
                                       const NDRange &SrcNDRange,
                                       uint32_t Index) {
  return generateGetSizeCase(Builder, TargetInfo, SrcNDRange.getLocalSize(),
                             SrcNDRange.getDimensions(), Index);
}

/// num_work_groups(x) = global_size(x) / local_size(x)
static Value *generateNumWorkGroupsCase(IRBuilderBase &Builder,
                                        const TargetFusionInfo &TargetInfo,
                                        const NDRange &SrcNDRange,
                                        uint32_t Index) {
  assert(SrcNDRange.hasSpecificLocalSize());
  auto N = TargetInfo.getIndexSpaceBuiltinBitwidth();
  Index = mirror(SrcNDRange.getDimensions(), Index);
  return Builder.getIntN(N, SrcNDRange.getGlobalSize()[Index] /
                                SrcNDRange.getLocalSize()[Index]);
}

static Value *remapGetGlobalID(IRBuilderBase &Builder,
                               const TargetFusionInfo &TargetInfo,
                               const NDRange &SrcNDRange,
                               const NDRange &FusedNDRange, uint32_t Index) {
  auto N = TargetInfo.getIndexSpaceBuiltinBitwidth();
  auto *GlobalLinearID = getGlobalLinearID(Builder, TargetInfo, FusedNDRange);
  const auto GetGS = [&Indices = SrcNDRange.getGlobalSize(),
                      Dimensions = SrcNDRange.getDimensions()](auto I) {
    return Indices[mirror(Dimensions, I)];
  };
  const auto GetConst = [&Builder, N](uint64_t C) {
    return Builder.getIntN(N, C);
  };
  switch (Index) {
  case 0:
    return Builder.CreateUDiv(GlobalLinearID, GetConst(GetGS(1) * GetGS(2)));
  case 1:
    return Builder.CreateURem(
        Builder.CreateUDiv(GlobalLinearID, GetConst(GetGS(2))),
        GetConst(GetGS(1)));
  case 2:
    return Builder.CreateURem(GlobalLinearID, GetConst(GetGS(2)));
  default:
    llvm_unreachable("Invalid index");
  }
}

/// global_id(0) = global_linear_id(x) / (global_size(1) * global_size(2))
/// global_id(1) = (global_linear_id(x) / global_size(2)) % global_size(1)
/// global_id(2) = global_linear_id(x) % global_size(2)
static Value *generateGetGlobalIDCase(IRBuilderBase &Builder,
                                      const TargetFusionInfo &TargetInfo,
                                      const NDRange &SrcNDRange,
                                      const NDRange &FusedNDRange,
                                      uint32_t Index) {
  // Note: This method ignores the global offset.
  assert(SrcNDRange.getOffset() == NDRange::AllZeros);
  return remapGetGlobalID(Builder, TargetInfo, SrcNDRange, FusedNDRange, Index);
}

/// local_id(x) = global_id(x) % local_size(x)
static Value *generateGetLocalIDCase(IRBuilderBase &Builder,
                                     const TargetFusionInfo &TargetInfo,
                                     const NDRange &SrcNDRange,
                                     const NDRange &FusedNDRange,
                                     uint32_t Index) {
  assert(SrcNDRange.hasSpecificLocalSize());
  auto N = TargetInfo.getIndexSpaceBuiltinBitwidth();
  auto *GlobalID =
      remapGetGlobalID(Builder, TargetInfo, SrcNDRange, FusedNDRange, Index);
  return Builder.CreateURem(
      GlobalID, Builder.getIntN(N, SrcNDRange.getLocalSize()[mirror(
                                       SrcNDRange.getDimensions(), Index)]));
}

/// group_id(x) = global_id(x) / local_size(x)
static Value *generateGetGroupIDCase(IRBuilderBase &Builder,
                                     const TargetFusionInfo &TargetInfo,
                                     const NDRange &SrcNDRange,
                                     const NDRange &FusedNDRange,
                                     uint32_t Index) {
  auto N = TargetInfo.getIndexSpaceBuiltinBitwidth();
  auto *GlobalID =
      remapGetGlobalID(Builder, TargetInfo, SrcNDRange, FusedNDRange, Index);
  return Builder.CreateUDiv(
      GlobalID, Builder.getIntN(N, SrcNDRange.getLocalSize()[mirror(
                                       SrcNDRange.getDimensions(), Index)]));
}

Value *Remapper::remap(BuiltinKind K, IRBuilderBase &Builder,
                       const NDRange &SrcNDRange, const NDRange &FusedNDRange,
                       uint32_t Index) const {
  switch (K) {
  case BuiltinKind::GlobalSizeRemapper:
    return generateGetGlobalSizeCase(Builder, TargetInfo, SrcNDRange, Index);
  case BuiltinKind::LocalSizeRemapper:
    return generateGetLocalSizeCase(Builder, TargetInfo, SrcNDRange, Index);
  case BuiltinKind::NumWorkGroupsRemapper:
    return generateNumWorkGroupsCase(Builder, TargetInfo, SrcNDRange, Index);
  case BuiltinKind::GlobalIDRemapper:
    return generateGetGlobalIDCase(Builder, TargetInfo, SrcNDRange,
                                   FusedNDRange, Index);
  case BuiltinKind::LocalIDRemapper:
    return generateGetLocalIDCase(Builder, TargetInfo, SrcNDRange, FusedNDRange,
                                  Index);
  case BuiltinKind::GroupIDRemapper:
    return generateGetGroupIDCase(Builder, TargetInfo, SrcNDRange, FusedNDRange,
                                  Index);
  case BuiltinKind::GlobalOffsetRemapper:
    return generateGetGlobalOffsetCase(Builder, TargetInfo, SrcNDRange, Index);
  }
  llvm_unreachable("Unhandled kind");
}

Expected<Function *>
jit_compiler::Remapper::remapBuiltins(Function *F, const NDRange &SrcNDRange,
                                      const NDRange &FusedNDRange) {
  auto &Cached = Cache[decltype(Cache)::key_type{F, SrcNDRange, FusedNDRange}];
  if (Cached) {
    // Cache hit. Return cached function.
    return Cached;
  }

  if (F->isDeclaration()) {
    if (auto OptK = TargetInfo.getBuiltinKind(F)) {
      auto K = OptK.value();
      if (!TargetInfo.shouldRemap(K, SrcNDRange, FusedNDRange))
        // If the builtin should not be remapped, return the original function.
        return F;

      return Cached = TargetInfo.createRemapperFunction(
                 *this, K, F, F->getParent(), SrcNDRange, FusedNDRange);
    }
    if (TargetInfo.isSafeToNotRemapBuiltin(F)) {
      // No need to remap.
      return Cached = F;
    }
    // Unknown builtin
    return make_error<StringError>(Twine("Cannot remap unknown builtin: \"")
                                       .concat(Twine{F->getName()})
                                       .concat("\""),
                                   inconvertibleErrorCode());
  }

  // As we clone the called function and remap the clone, we can have
  // more than one callee to the same function being remapped and a different
  // remapping will be performed each time.
  ValueToValueMapTy Map;
  ClonedCodeInfo CodeInfo;
  auto *Clone = CloneFunction(F, Map, &CodeInfo);

  if (!CodeInfo.ContainsCalls) {
    // No need to perform any remapping as the function has no calls.
    // We can erase the cloned function from the parent and return the
    // original.
    Clone->eraseFromParent();
    return Cached = F;
  }

  // Set Cached to support recursive functions.
  Cached = Clone;
  if (auto Err = TargetInfo.scanForBuiltinsToRemap(Clone, *this, SrcNDRange,
                                                   FusedNDRange)) {
    return Err;
  }
  return Clone;
}
