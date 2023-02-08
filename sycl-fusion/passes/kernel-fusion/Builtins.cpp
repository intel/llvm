#include "Builtins.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace jit_compiler;

namespace {
/// This class implements all of the logic to remap builtins.
///
/// When constructed, the "remapping function" to be called instead of the
/// builtin is initialized (if needed) and the remapping then simply involves
/// replacing each relevant call by such function.
class Remapper {
public:
  enum class Kind : uint8_t {
    GlobalSizeRemapper,
    LocalSizeRemapper,
    NumWorkGroupsRemapper,
    GlobalOffsetRemapper,
    GlobalIDRemapper,
    LocalIDRemapper,
    GroupIDRemapper,
  } K;

  Remapper(Kind K, Module &M, const NDRange &SrcNDRange,
           const NDRange &FusedNDRange)
      : K{K}, F{initFunction(K, M, SrcNDRange, FusedNDRange)} {}

  void operator()(CallBase *C) const;

  static constexpr StringLiteral RemapperCommonName{"__remapper"};

private:
  static uint64_t getDefaultValue(Kind K);
  static std::string getFunctionName(Kind K, const NDRange &SrcNDRange,
                                     const NDRange &FusedNDRange);
  static bool shouldRemap(Kind K, const NDRange &SrcNDRange,
                          const NDRange &FusedNDRange);
  static Value *generateCase(Kind K, IRBuilderBase &Builder,
                             const NDRange &SrcNDRange,
                             const NDRange &FusedNDRange, uint32_t Index);
  static Function *initFunction(Kind K, Module &M, const NDRange &SrcNDRange,
                                const NDRange &FusedNDRange);

  Function *F;
};
} // namespace

void Remapper::operator()(CallBase *C) const {
  if (F) {
    C->setCalledFunction(F);
    C->setAttributes(F->getAttributes());
  }
}

/// 0 for IDs/offset and 1 for sizes.
uint64_t Remapper::getDefaultValue(Kind K) {
  switch (K) {
  case Kind::GlobalSizeRemapper:
  case Kind::LocalSizeRemapper:
  case Kind::NumWorkGroupsRemapper:
    return 1;
  case Kind::GlobalIDRemapper:
  case Kind::LocalIDRemapper:
  case Kind::GroupIDRemapper:
  case Kind::GlobalOffsetRemapper:
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
std::string Remapper::getFunctionName(Kind K, const NDRange &SrcNDRange,
                                      const NDRange &FusedNDRange) {
  std::string Res;
  raw_string_ostream S{Res};
  S << "__" <<
      [K]() {
        switch (K) {
        case Kind::GlobalSizeRemapper:
          return "global_size";
        case Kind::LocalSizeRemapper:
          return "local_size";
        case Kind::NumWorkGroupsRemapper:
          return "num_work_groups";
        case Kind::GlobalIDRemapper:
          return "global_id";
        case Kind::LocalIDRemapper:
          return "local_id";
        case Kind::GroupIDRemapper:
          return "group_id";
        case Kind::GlobalOffsetRemapper:
          return "global_offset";
        }
        llvm_unreachable("Unhandled kind");
      }()
    << "_remapper_" << SrcNDRange << "_" << FusedNDRange;
  return S.str();
}

bool Remapper::shouldRemap(Kind K, const NDRange &SrcNDRange,
                           const NDRange &FusedNDRange) {
  switch (K) {
  case Kind::GlobalSizeRemapper:
  case Kind::GlobalOffsetRemapper:
    return true;
  case Kind::NumWorkGroupsRemapper:
  case Kind::LocalSizeRemapper:
    // Do not remap when the local size is not specified.
    return SrcNDRange.hasSpecificLocalSize();
  case Kind::GlobalIDRemapper:
  case Kind::LocalIDRemapper:
  case Kind::GroupIDRemapper: {
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

Value *Remapper::generateCase(Kind K, IRBuilderBase &Builder,
                              const NDRange &SrcNDRange,
                              const NDRange &FusedNDRange, uint32_t Index) {
  switch (K) {
  case Kind::GlobalSizeRemapper:
    return generateGetGlobalSizeCase(Builder, SrcNDRange, FusedNDRange, Index);
  case Kind::LocalSizeRemapper:
    return generateGetLocalSizeCase(Builder, SrcNDRange, FusedNDRange, Index);
  case Kind::NumWorkGroupsRemapper:
    return generateNumWorkGroupsCase(Builder, SrcNDRange, FusedNDRange, Index);
  case Kind::GlobalIDRemapper:
    return generateGetGlobalIDCase(Builder, SrcNDRange, FusedNDRange, Index);
  case Kind::LocalIDRemapper:
    return generateGetLocalIDCase(Builder, SrcNDRange, FusedNDRange, Index);
  case Kind::GroupIDRemapper:
    return generateGetGroupIDCase(Builder, SrcNDRange, FusedNDRange, Index);
  case Kind::GlobalOffsetRemapper:
    return generateGetGlobalOffsetCase(Builder, SrcNDRange, FusedNDRange,
                                       Index);
  }
  llvm_unreachable("Unhandled kind");
}

static constexpr size_t NumBuiltins{8};
static constexpr size_t NumBuiltinsToRemap{7};

template <typename ForwardIt>
static ForwardIt mapArrayLookup(ForwardIt Begin, ForwardIt End,
                                const decltype(Begin->first) &Key) {
  return std::lower_bound(
      Begin, End, Key,
      [](const auto &Entry, const auto &Key) { return Entry.first < Key; });
}

template <typename ForwardIt>
static auto mapArrayLookupValue(ForwardIt Begin, ForwardIt End,
                                const decltype(Begin->first) &Key)
    -> decltype(Begin->second) {
  const auto Iter = mapArrayLookup(Begin, End, Key);
  assert(Iter != End && Iter->first == Key && "Invalid key");
  return Iter->second;
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

  // This array is sorted by key value
  const std::array<
      std::pair<StringRef, function_ref<llvm::AttributeList(LLVMContext &)>>,
      NumBuiltins + 1>
      AttrMap{{{BarrierName, GetBarrierAttrs},
               {GetGlobalSizeName, GetIndexSpaceAttrs},
               {GetGroupIDName, GetIndexSpaceAttrs},
               {GetNumWorkGroupsName, GetIndexSpaceAttrs},
               {GetLocalSizeName, GetIndexSpaceAttrs},
               {GetGlobalLinearIDName, GetIndexSpaceAttrs},
               {GetLocalIDName, GetIndexSpaceAttrs},
               {GetGlobalIDName, GetIndexSpaceAttrs},
               {Remapper::RemapperCommonName, RemapperAttrs}}};
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

  // This array is sorted by key value
  const std::array<
      std::pair<StringRef, function_ref<FunctionType *(IRBuilderBase &)>>,
      NumBuiltins + 1>
      TypeMap{{{BarrierName, GetBarrierTy},
               {GetGlobalSizeName, GetIndexSpaceGetTy},
               {GetGroupIDName, GetIndexSpaceGetTy},
               {GetNumWorkGroupsName, GetIndexSpaceGetTy},
               {GetLocalSizeName, GetIndexSpaceGetTy},
               {GetGlobalLinearIDName, GetGlobalLinearIDTy},
               {GetLocalIDName, GetIndexSpaceGetTy},
               {GetGlobalIDName, GetIndexSpaceGetTy},
               {Remapper::RemapperCommonName, GetIndexSpaceGetTy}}};

  return mapArrayLookupValue(TypeMap.begin(), TypeMap.end(),
                             FunctionName)(Builder);
}

Function *Remapper::initFunction(Kind K, Module &M, const NDRange &SrcNDRange,
                                 const NDRange &FusedNDRange) {
  if (!shouldRemap(K, SrcNDRange, FusedNDRange)) {
    // If the builtin should not be remapped, the function won't be initialized.
    return nullptr;
  }
  const auto Name = getFunctionName(K, SrcNDRange, FusedNDRange);
  auto *F = M.getFunction(Name);
  if (F) {
    // Remapping function already generated
    return F;
  }

  auto &Ctx = M.getContext();
  IRBuilder<> Builder{Ctx};

  F = Function::Create(getFunctionType(Builder, RemapperCommonName),
                       Function::LinkageTypes::InternalLinkage, Name, M);

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

Function *jit_compiler::remapBuiltins(Function *F, const NDRange &SrcNDRange,
                                      const NDRange &FusedNDRange) {
  {
    ValueToValueMapTy Map;
    F = CloneFunction(F, Map);
  }
  auto &M = *F->getParent();
  // This is a sorted array which should have the same order as the one in the
  // remap function.
  // Values are lazy initialized.
  std::array<std::pair<StringRef,
                       std::pair<Remapper::Kind, std::unique_ptr<Remapper>>>,
             NumBuiltinsToRemap>
      Remappers{{
          {GetGlobalSizeName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::GlobalSizeRemapper,
               std::unique_ptr<Remapper>{}}},
          {GetGroupIDName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::GroupIDRemapper, std::unique_ptr<Remapper>{}}},
          {GetGlobalOffsetName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::GlobalOffsetRemapper,
               std::unique_ptr<Remapper>{}}},
          {GetNumWorkGroupsName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::NumWorkGroupsRemapper,
               std::unique_ptr<Remapper>{}}},
          {GetLocalSizeName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::LocalSizeRemapper, std::unique_ptr<Remapper>{}}},
          {GetLocalIDName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::LocalIDRemapper, std::unique_ptr<Remapper>{}}},
          {GetGlobalIDName,
           std::pair<Remapper::Kind, std::unique_ptr<Remapper>>{
               Remapper::Kind::GlobalIDRemapper, std::unique_ptr<Remapper>{}}},
      }};
  for (auto &I : instructions(F)) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      const auto Name = Call->getCalledFunction()->getName();
      auto *Iter = mapArrayLookup(Remappers.begin(), Remappers.end(), Name);
      if (Iter != Remappers.end() && Iter->first == Name) {
        if (!Iter->second.second) {
          Iter->second.second = std::make_unique<Remapper>(
              Iter->second.first, M, SrcNDRange, FusedNDRange);
        }
        (*Iter->second.second)(Call);
      }
    }
  }
  return F;
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
