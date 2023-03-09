#include "Builtins.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace jit_compiler;

enum class BuiltinKind : uint8_t {
  GlobalSizeRemapper,
  LocalSizeRemapper,
  NumWorkGroupsRemapper,
  GlobalOffsetRemapper,
  GlobalIDRemapper,
  LocalIDRemapper,
  GroupIDRemapper,
};

static constexpr StringLiteral RemapperCommonName{"__remapper"};
static constexpr size_t NumBuiltins{11};
static constexpr size_t NumBuiltinsToRemap{7};

template <typename ForwardIt, typename KeyTy>
static ForwardIt mapArrayLookup(ForwardIt Begin, ForwardIt End,
                                const KeyTy &Key) {
  return std::lower_bound(
      Begin, End, Key,
      [](const auto &Entry, const auto &Key) { return Entry.first < Key; });
}

template <typename ForwardIt, typename KeyTy>
static auto mapArrayLookupValue(ForwardIt Begin, ForwardIt End,
                                const KeyTy &Key) -> decltype(Begin->second) {
  const auto Iter = mapArrayLookup(Begin, End, Key);
  assert(Iter != End && Iter->first == Key && "Invalid key");
  return Iter->second;
}

static BuiltinKind getBuiltinKind(Function *F) {
  constexpr std::array<std::pair<StringLiteral, BuiltinKind>,
                       NumBuiltinsToRemap>
      Map{{{GetGlobalSizeName, BuiltinKind::GlobalSizeRemapper},
           {GetGroupIDName, BuiltinKind::GroupIDRemapper},
           {GetGlobalOffsetName, BuiltinKind::GlobalOffsetRemapper},
           {GetNumWorkGroupsName, BuiltinKind::NumWorkGroupsRemapper},
           {GetLocalSizeName, BuiltinKind::LocalSizeRemapper},
           {GetLocalIDName, BuiltinKind::LocalIDRemapper},
           {GetGlobalIDName, BuiltinKind::GlobalIDRemapper}}};
  return mapArrayLookupValue(Map.begin(), Map.end(), F->getName());
}

/// 0 for IDs/offset and 1 for sizes.
static uint64_t getDefaultValue(BuiltinKind K) {
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

/// Will generate a unique function name so that it can be reused in further
/// stages.
static std::string getFunctionName(BuiltinKind K, const NDRange &SrcNDRange,
                                   const NDRange &FusedNDRange) {
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
  return S.str();
}

static bool shouldRemap(BuiltinKind K, const NDRange &SrcNDRange,
                        const NDRange &FusedNDRange) {
  switch (K) {
  case BuiltinKind::GlobalSizeRemapper:
  case BuiltinKind::GlobalOffsetRemapper:
    return true;
  case BuiltinKind::NumWorkGroupsRemapper:
  case BuiltinKind::LocalSizeRemapper:
    // Do not remap when the local size is not specified.
    return SrcNDRange.hasSpecificLocalSize();
  case BuiltinKind::GlobalIDRemapper:
  case BuiltinKind::LocalIDRemapper:
  case BuiltinKind::GroupIDRemapper: {
    // No need to remap when all but the dimensions and the left-most components
    // of the global size range are equal.
    const auto &GS0 = SrcNDRange.getGlobalSize();
    const auto &GS1 = FusedNDRange.getGlobalSize();
    return SrcNDRange.getDimensions() != FusedNDRange.getDimensions() ||
           !std::equal(GS0.begin() + 1, GS0.end(), GS1.begin() + 1);
  }
  }
  llvm_unreachable("Unhandled kind");
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

static Value *generateGetSizeCase(IRBuilderBase &Builder,
                                  const NDRange &SrcNDRange,
                                  const NDRange &FusedNDRange,
                                  const Indices &SrcIndices, uint32_t Index) {
  return Builder.getInt64(
      SrcIndices[mirror(SrcNDRange.getDimensions(), Index)]);
}

/// Input argument
static Value *generateGetGlobalSizeCase(IRBuilderBase &Builder,
                                        const NDRange &SrcNDRange,
                                        const NDRange &FusedNDRange,
                                        uint32_t Index) {
  return generateGetSizeCase(Builder, SrcNDRange, FusedNDRange,
                             SrcNDRange.getGlobalSize(), Index);
}

/// Input argument
static Value *generateGetGlobalOffsetCase(IRBuilderBase &Builder,
                                          const NDRange &SrcNDRange,
                                          const NDRange &FusedNDRange,
                                          uint32_t Index) {
  return generateGetSizeCase(Builder, SrcNDRange, FusedNDRange,
                             SrcNDRange.getOffset(), Index);
}

/// Input argument
static Value *generateGetLocalSizeCase(IRBuilderBase &Builder,
                                       const NDRange &SrcNDRange,
                                       const NDRange &FusedNDRange,
                                       uint32_t Index) {
  return generateGetSizeCase(Builder, SrcNDRange, FusedNDRange,
                             SrcNDRange.getLocalSize(), Index);
}

/// num_work_groups(x) = global_size(x) / local_size(x)
static Value *generateNumWorkGroupsCase(IRBuilderBase &Builder,
                                        const NDRange &SrcNDRange,
                                        const NDRange &FusedNDRange,
                                        uint32_t Index) {
  Index = mirror(SrcNDRange.getDimensions(), Index);
  return Builder.getInt64(SrcNDRange.getGlobalSize()[Index] /
                          SrcNDRange.getLocalSize()[Index]);
}
static Value *remapGetGlobalID(IRBuilderBase &Builder,
                               const NDRange &SrcNDRange,
                               const NDRange &FusedNDRange, uint32_t Index) {
  auto *GlobalLinearID = getGlobalLinearID(Builder, FusedNDRange);
  const auto getGS = [&Indices = SrcNDRange.getGlobalSize(),
                      Dimensions = SrcNDRange.getDimensions()](auto I) {
    return Indices[mirror(Dimensions, I)];
  };
  switch (Index) {
  case 0:
    return Builder.CreateUDiv(GlobalLinearID,
                              Builder.getInt64(getGS(1) * getGS(2)));
  case 1:
    return Builder.CreateURem(
        Builder.CreateUDiv(GlobalLinearID, Builder.getInt64(getGS(2))),
        Builder.getInt64(getGS(1)));
  case 2:
    return Builder.CreateURem(GlobalLinearID, Builder.getInt64(getGS(2)));
  default:
    llvm_unreachable("Invalid index");
  }
}

/// global_id(0) = global_linear_id(x) / (global_size(1) * global_size(2))
/// global_id(1) = (global_linear_id(x) / global_size(2)) % global_size(1)
/// global_id(2) = global_linear_id(x) % global_size(2)
static Value *generateGetGlobalIDCase(IRBuilderBase &Builder,
                                      const NDRange &SrcNDRange,
                                      const NDRange &FusedNDRange,
                                      uint32_t Index) {
  return remapGetGlobalID(Builder, SrcNDRange, FusedNDRange, Index);
}

/// local_id(x) = global_id(x) % local_size(x)
static Value *generateGetLocalIDCase(IRBuilderBase &Builder,
                                     const NDRange &SrcNDRange,
                                     const NDRange &FusedNDRange,
                                     uint32_t Index) {
  auto *GlobalID = remapGetGlobalID(Builder, SrcNDRange, FusedNDRange, Index);
  return Builder.CreateURem(GlobalID,
                            Builder.getInt64(SrcNDRange.getLocalSize()[mirror(
                                SrcNDRange.getDimensions(), Index)]));
}

/// group_id(x) = global_id(x) / local_size(x)
static Value *generateGetGroupIDCase(IRBuilderBase &Builder,
                                     const NDRange &SrcNDRange,
                                     const NDRange &FusedNDRange,
                                     uint32_t Index) {
  auto *GlobalID = remapGetGlobalID(Builder, SrcNDRange, FusedNDRange, Index);
  return Builder.CreateUDiv(GlobalID,
                            Builder.getInt64(SrcNDRange.getLocalSize()[mirror(
                                SrcNDRange.getDimensions(), Index)]));
}

static Value *generateCase(BuiltinKind K, IRBuilderBase &Builder,
                           const NDRange &SrcNDRange,
                           const NDRange &FusedNDRange, uint32_t Index) {
  switch (K) {
  case BuiltinKind::GlobalSizeRemapper:
    return generateGetGlobalSizeCase(Builder, SrcNDRange, FusedNDRange, Index);
  case BuiltinKind::LocalSizeRemapper:
    return generateGetLocalSizeCase(Builder, SrcNDRange, FusedNDRange, Index);
  case BuiltinKind::NumWorkGroupsRemapper:
    return generateNumWorkGroupsCase(Builder, SrcNDRange, FusedNDRange, Index);
  case BuiltinKind::GlobalIDRemapper:
    return generateGetGlobalIDCase(Builder, SrcNDRange, FusedNDRange, Index);
  case BuiltinKind::LocalIDRemapper:
    return generateGetLocalIDCase(Builder, SrcNDRange, FusedNDRange, Index);
  case BuiltinKind::GroupIDRemapper:
    return generateGetGroupIDCase(Builder, SrcNDRange, FusedNDRange, Index);
  case BuiltinKind::GlobalOffsetRemapper:
    return generateGetGlobalOffsetCase(Builder, SrcNDRange, FusedNDRange,
                                       Index);
  }
  llvm_unreachable("Unhandled kind");
}

static llvm::AttributeList getAttributes(StringRef FunctionName,
                                         LLVMContext &Ctx) {
  constexpr auto GetIndexSpaceAttrs = [](LLVMContext &Ctx) {
    return llvm::AttributeList::get(
        Ctx,
        AttributeSet::get(Ctx,
                          {Attribute::get(Ctx, Attribute::AttrKind::WillReturn),
                           Attribute::get(Ctx, Attribute::AttrKind::NoUnwind)}),
        {}, {});
  };
  constexpr auto GetBarrierAttrs = [](LLVMContext &Ctx) {
    return llvm::AttributeList::get(
        Ctx,
        AttributeSet::get(Ctx,
                          {Attribute::get(Ctx, Attribute::AttrKind::Convergent),
                           Attribute::get(Ctx, Attribute::AttrKind::NoUnwind)}),
        {}, {});
  };
  constexpr auto RemapperAttrs = [](LLVMContext &Ctx) {
    return llvm::AttributeList::get(
        Ctx,
        AttributeSet::get(
            Ctx, {Attribute::get(Ctx, Attribute::AttrKind::NoUnwind),
                  Attribute::get(Ctx, Attribute::AttrKind::AlwaysInline)}),
        {}, {});
  };
  constexpr auto OffloaderAttrs = RemapperAttrs;

  // This array is sorted by key value
  const std::array<
      std::pair<StringRef, function_ref<llvm::AttributeList(LLVMContext &)>>,
      NumBuiltins + 1>
      AttrMap{{{BarrierName, GetBarrierAttrs},
               {GetGlobalSizeName, GetIndexSpaceAttrs},
               {GetGroupIDName, GetIndexSpaceAttrs},
               {GetGlobalOffsetName, GetIndexSpaceAttrs},
               {GetNumWorkGroupsName, GetIndexSpaceAttrs},
               {GetLocalSizeName, GetIndexSpaceAttrs},
               {GetGlobalLinearIDName, GetIndexSpaceAttrs},
               {GetLocalIDName, GetIndexSpaceAttrs},
               {GetGlobalIDName, GetIndexSpaceAttrs},
               {RemapperCommonName, RemapperAttrs},
               {OffloadStartWrapperName, OffloaderAttrs},
               {OffloadFinishWrapperName, OffloaderAttrs}}};
  return mapArrayLookupValue(AttrMap.begin(), AttrMap.end(), FunctionName)(Ctx);
}

static void setFunctionMetadata(Function *F, StringRef FunctionName,
                                LLVMContext &Ctx) {
  F->setAttributes(getAttributes(FunctionName, Ctx));
  F->setCallingConv(CallingConv::SPIR_FUNC);
}

static FunctionType *getFunctionType(IRBuilderBase &Builder,
                                     StringRef FunctionName) {
  constexpr auto GetIndexSpaceGetTy = [](IRBuilderBase &Builder) {
    return FunctionType::get(Builder.getInt64Ty(), {Builder.getInt32Ty()},
                             false /*IsVarArg*/);
  };
  constexpr auto GetGlobalLinearIDTy = [](IRBuilderBase &Builder) {
    return FunctionType::get(Builder.getInt64Ty(), {}, false /*IsVarArg*/);
  };
  constexpr auto GetBarrierTy = [](IRBuilderBase &Builder) {
    return FunctionType::get(
        Builder.getVoidTy(),
        {Builder.getInt32Ty(), Builder.getInt32Ty(), Builder.getInt32Ty()},
        false /*IsVarArg*/);
  };
  constexpr auto OffloadWrapperTy = [](IRBuilderBase &Builder) {
    return FunctionType::get(Builder.getVoidTy(), {}, false /*IsVarArg*/);
  };

  // This array is sorted by key value
  const std::array<
      std::pair<StringRef, function_ref<FunctionType *(IRBuilderBase &)>>,
      NumBuiltins + 1>
      TypeMap{{{BarrierName, GetBarrierTy},
               {GetGlobalSizeName, GetIndexSpaceGetTy},
               {GetGroupIDName, GetIndexSpaceGetTy},
               {GetGlobalOffsetName, GetIndexSpaceGetTy},
               {GetNumWorkGroupsName, GetIndexSpaceGetTy},
               {GetLocalSizeName, GetIndexSpaceGetTy},
               {GetGlobalLinearIDName, GetGlobalLinearIDTy},
               {GetLocalIDName, GetIndexSpaceGetTy},
               {GetGlobalIDName, GetIndexSpaceGetTy},
               {RemapperCommonName, GetIndexSpaceGetTy},
               {OffloadFinishWrapperName, OffloadWrapperTy},
               {OffloadStartWrapperName, OffloadWrapperTy}}};

  return mapArrayLookupValue(TypeMap.begin(), TypeMap.end(),
                             FunctionName)(Builder);
}

static Function *initFunction(Function *OldF, const NDRange &SrcNDRange,
                              const NDRange &FusedNDRange) {
  const auto K = getBuiltinKind(OldF);
  if (!shouldRemap(K, SrcNDRange, FusedNDRange)) {
    // If the builtin should not be remapped, return the original function.
    return OldF;
  }
  const auto Name = getFunctionName(K, SrcNDRange, FusedNDRange);
  auto *M = OldF->getParent();
  auto *F = M->getFunction(Name);
  assert(!F && "Function name should be unique");

  auto &Ctx = M->getContext();
  IRBuilder<> Builder{Ctx};

  F = Function::Create(getFunctionType(Builder, RemapperCommonName),
                       Function::LinkageTypes::InternalLinkage, Name, *M);

  auto *EntryBlock = BasicBlock::Create(Ctx, "entry", F);
  Builder.SetInsertPoint(EntryBlock);

  constexpr unsigned SYCLDimensions{3};
  // Vector holding all the possible values
  auto *Vec = cast<Value>(
      ConstantVector::getSplat(ElementCount::getFixed(SYCLDimensions),
                               Builder.getInt64(getDefaultValue(K))));

  const auto NumDimensions = static_cast<uint32_t>(SrcNDRange.getDimensions());
  for (uint32_t I = 0; I < NumDimensions; ++I) {
    // Initialize vector
    Vec = Builder.CreateInsertElement(
        Vec, generateCase(K, Builder, SrcNDRange, FusedNDRange, I),
        Builder.getInt32(I));
  }
  // Get queried value
  Builder.CreateRet(Builder.CreateExtractElement(Vec, F->getArg(0)));

  setFunctionMetadata(F, RemapperCommonName, Ctx);

  return F;
}

Value *jit_compiler::getGlobalLinearID(IRBuilderBase &Builder,
                                       const NDRange &FusedNDRange) {
  return createSPIRVCall(Builder, GetGlobalLinearIDName, {});
}

static bool isIndexSpaceGetterBuiltin(Function *F) {
  constexpr std::array<StringLiteral, NumBuiltinsToRemap> BuiltinNames{
      GetGlobalSizeName,    GetGroupIDName,   GetGlobalOffsetName,
      GetNumWorkGroupsName, GetLocalSizeName, GetLocalIDName,
      GetGlobalIDName};
  const auto Name = F->getName();
  const auto *Iter = llvm::lower_bound(BuiltinNames, Name);
  return Iter != BuiltinNames.end() && *Iter == Name;
}

static bool isSafeToNotRemapBuiltin(Function *F) {
  constexpr std::size_t NumUnsafeBuiltins{8};
  // SPIRV builtins with kernel capabilities in alphabetical order.
  //
  // These builtins might need remapping, but are not supported by the remapper,
  // so we should abort kernel fusion if we find them during remapping.
  constexpr std::array<StringLiteral, NumUnsafeBuiltins> UnsafeBuiltIns{
      "EnqueuedWorkgroupSize",
      "NumEnqueuedSubgroups",
      "NumSubgroups",
      "SubgroupId",
      "SubgroupLocalInvocationId",
      "SubgroupMaxSize",
      "SubgroupSize",
      "WorkDim"};
  constexpr StringLiteral SPIRVBuiltinNamespace{"spirv"};
  constexpr StringLiteral SPIRVBuiltinPrefix{"BuiltIn"};

  auto Name = F->getName();
  if (!(Name.contains(SPIRVBuiltinNamespace) &&
        Name.contains(SPIRVBuiltinPrefix))) {
    return true;
  }
  // Drop "spirv" namespace name and "BuiltIn" prefix.
  Name = Name.drop_front(Name.find(SPIRVBuiltinPrefix) +
                         SPIRVBuiltinPrefix.size());
  // Check that Name does not start with any name in UnsafeBuiltIns
  const auto *Iter =
      std::upper_bound(UnsafeBuiltIns.begin(), UnsafeBuiltIns.end(), Name);
  return Iter == UnsafeBuiltIns.begin() || !Name.starts_with(*(Iter - 1));
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
    if (isIndexSpaceGetterBuiltin(F)) {
      // Remap given builtin.
      return Cached = initFunction(F, SrcNDRange, FusedNDRange);
    }
    if (isSafeToNotRemapBuiltin(F)) {
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
  for (auto &I : instructions(Clone)) {
    if (auto *Call = dyn_cast<CallBase>(&I)) {
      // Recursive call
      auto *OldF = Call->getCalledFunction();
      auto ErrOrNewF = remapBuiltins(OldF, SrcNDRange, FusedNDRange);
      if (auto Err = ErrOrNewF.takeError()) {
        return Err;
      }
      // Override called function.
      auto *NewF = *ErrOrNewF;
      Call->setCalledFunction(NewF);
      Call->setCallingConv(NewF->getCallingConv());
      Call->setAttributes(NewF->getAttributes());
    }
  }
  return Clone;
}

void jit_compiler::barrierCall(IRBuilderBase &Builder, int Flags) {
  assert((Flags == 1 || Flags == 2 || Flags == 3) && "Invalid barrier flags");

  // See
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-
  createSPIRVCall(Builder, BarrierName,
                  {Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
                   Builder.getInt32(/*Exec Scope : Workgroup = */ 2),
                   Builder.getInt32(0x10 | (Flags % 2 == 1 ? 0x100 : 0x0) |
                                    ((Flags >> 1 == 1 ? 0x200 : 0x0)))});
}

Value *jit_compiler::createSPIRVCall(IRBuilderBase &Builder,
                                     StringRef FunctionName,
                                     ArrayRef<Value *> Args) {
  auto *M = Builder.GetInsertBlock()->getParent()->getParent();
  auto *F = M->getFunction(FunctionName);
  if (!F) {
    constexpr auto Linkage = GlobalValue::LinkageTypes::ExternalLinkage;

    F = Function::Create(getFunctionType(Builder, FunctionName), Linkage,
                         FunctionName, M);

    setFunctionMetadata(F, FunctionName, Builder.getContext());
  }

  auto *Call = Builder.CreateCall(F, Args);

  Call->setAttributes(F->getAttributes());
  Call->setCallingConv(F->getCallingConv());

  return Call;
}
