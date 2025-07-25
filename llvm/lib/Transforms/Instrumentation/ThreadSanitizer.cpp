//===-- ThreadSanitizer.cpp - race detector -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer, a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Type.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/SPIRVSanitizerCommonUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tsan"

static cl::opt<bool> ClInstrumentMemoryAccesses(
    "tsan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncEntryExit("tsan-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool> ClHandleCxxExceptions(
    "tsan-handle-cxx-exceptions", cl::init(true),
    cl::desc("Handle C++ exceptions (insert cleanup blocks for unwinding)"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("tsan-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "tsan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClDistinguishVolatile(
    "tsan-distinguish-volatile", cl::init(false),
    cl::desc("Emit special instrumentation for accesses to volatiles"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentReadBeforeWrite(
    "tsan-instrument-read-before-write", cl::init(false),
    cl::desc("Do not eliminate read instrumentation for read-before-writes"),
    cl::Hidden);
static cl::opt<bool> ClCompoundReadBeforeWrite(
    "tsan-compound-read-before-write", cl::init(false),
    cl::desc("Emit special compound instrumentation for reads-before-writes"),
    cl::Hidden);

static cl::opt<bool> ClSpirOffloadLocals("tsan-spir-locals",
                                         cl::desc("instrument local pointer"),
                                         cl::Hidden, cl::init(true));

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumInstrumentedVtableWrites, "Number of vtable ptr writes");
STATISTIC(NumInstrumentedVtableReads, "Number of vtable ptr reads");
STATISTIC(NumOmittedReadsFromConstantGlobals,
          "Number of reads from constant globals");
STATISTIC(NumOmittedReadsFromVtable, "Number of vtable reads");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");

const char kTsanModuleCtorName[] = "tsan.module_ctor";
const char kTsanInitName[] = "__tsan_init";

namespace {

struct ThreadSanitizer;

// SPIR-V specific instrumentation
struct ThreadSanitizerOnSpirv {
  ThreadSanitizerOnSpirv(Module &M)
      : M(M), C(M.getContext()), DL(M.getDataLayout()) {
    IntptrTy = DL.getIntPtrType(C);
  }

  void initialize();

  void instrumentModule();

  bool instrumentAllocInst(Function *F,
                           SmallVectorImpl<Instruction *> &AllocaInsts);

  void instrumentDynamicLocalMemory(Function &F);

  bool instrumentControlBarrier(CallInst *CI);

  void appendDebugInfoToArgs(Instruction *I, SmallVectorImpl<Value *> &Args);

  bool isUnsupportedSPIRAccess(Value *Addr, Instruction *Inst);

private:
  void instrumentGlobalVariables();

  void instrumentStaticLocalMemory();

  void instrumentKernelsMetadata();

  void initializeKernelCallerMap(Function *F);

  bool isSupportedSPIRKernel(Function &F);

  bool isUnsupportedDeviceGlobal(const GlobalVariable &G);

  GlobalVariable *GetOrCreateGlobalString(StringRef Name, StringRef Value,
                                          unsigned AddressSpace);

private:
  Module &M;
  LLVMContext &C;
  const DataLayout &DL;
  Type *IntptrTy;

  StringMap<GlobalVariable *> GlobalStringMap;

  DenseMap<Function *, DenseSet<Function *>> FuncToKernelCallerMap;

  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  static const size_t kNumberOfAddressSpace = 5;
  FunctionCallee TsanCleanupPrivate;
  FunctionCallee TsanCleanupStaticLocal;
  FunctionCallee TsanCleanupDynamicLocal;
  FunctionCallee TsanDeviceBarrier;
  FunctionCallee TsanGroupBarrier;
  FunctionCallee TsanRead[kNumberOfAccessSizes][kNumberOfAddressSpace];
  FunctionCallee TsanWrite[kNumberOfAccessSizes][kNumberOfAddressSpace];
  FunctionCallee TsanUnalignedRead[kNumberOfAccessSizes][kNumberOfAddressSpace];
  FunctionCallee TsanUnalignedWrite[kNumberOfAccessSizes]
                                   [kNumberOfAddressSpace];

  friend struct ThreadSanitizer;
};

/// ThreadSanitizer: instrument the code in module to find races.
///
/// Instantiating ThreadSanitizer inserts the tsan runtime library API function
/// declarations into the module if they don't exist already. Instantiating
/// ensures the __tsan_init function is in the list of global constructors for
/// the module.
struct ThreadSanitizer {
  ThreadSanitizer() {
    // Check options and warn user.
    if (ClInstrumentReadBeforeWrite && ClCompoundReadBeforeWrite) {
      errs()
          << "warning: Option -tsan-compound-read-before-write has no effect "
             "when -tsan-instrument-read-before-write is set.\n";
    }
  }

  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);

private:
  // Internal Instruction wrapper that contains more information about the
  // Instruction from prior analysis.
  struct InstructionInfo {
    // Instrumentation emitted for this instruction is for a compounded set of
    // read and write operations in the same basic block.
    static constexpr unsigned kCompoundRW = (1U << 0);

    explicit InstructionInfo(Instruction *Inst) : Inst(Inst) {}

    Instruction *Inst;
    unsigned Flags = 0;
  };

  void initialize(Module &M, const TargetLibraryInfo &TLI);
  bool instrumentLoadOrStore(const InstructionInfo &II, const DataLayout &DL);
  bool instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  void chooseInstructionsToInstrument(SmallVectorImpl<Instruction *> &Local,
                                      SmallVectorImpl<InstructionInfo> &All,
                                      const DataLayout &DL);
  bool addrPointsToConstantData(Value *Addr);
  int getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr, const DataLayout &DL);
  void InsertRuntimeIgnores(Function &F);

  std::optional<ThreadSanitizerOnSpirv> Spirv;

  Type *IntptrTy;
  FunctionCallee TsanFuncEntry;
  FunctionCallee TsanFuncExit;
  FunctionCallee TsanIgnoreBegin;
  FunctionCallee TsanIgnoreEnd;
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  FunctionCallee TsanRead[kNumberOfAccessSizes];
  FunctionCallee TsanWrite[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedRead[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedWrite[kNumberOfAccessSizes];
  FunctionCallee TsanVolatileRead[kNumberOfAccessSizes];
  FunctionCallee TsanVolatileWrite[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedVolatileRead[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedVolatileWrite[kNumberOfAccessSizes];
  FunctionCallee TsanCompoundRW[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedCompoundRW[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicLoad[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicStore[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicRMW[AtomicRMWInst::LAST_BINOP + 1]
                              [kNumberOfAccessSizes];
  FunctionCallee TsanAtomicCAS[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicThreadFence;
  FunctionCallee TsanAtomicSignalFence;
  FunctionCallee TsanVptrUpdate;
  FunctionCallee TsanVptrLoad;
  FunctionCallee MemmoveFn, MemcpyFn, MemsetFn;
};

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kTsanModuleCtorName, kTsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}
} // namespace

PreservedAnalyses ThreadSanitizerPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  ThreadSanitizer TSan;
  if (TSan.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleThreadSanitizerPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  // Return early if nosanitize_thread module flag is present for the module.
  if (checkIfAlreadyInstrumented(M, "nosanitize_thread"))
    return PreservedAnalyses::all();
  if (Triple(M.getTargetTriple()).isSPIROrSPIRV()) {
    ThreadSanitizerOnSpirv Spirv(M);
    Spirv.initialize();
    Spirv.instrumentModule();
  } else
    insertModuleCtor(M);
  return PreservedAnalyses::none();
}

void ThreadSanitizerOnSpirv::initialize() {
  IRBuilder<> IRB(C);
  AttributeList Attr;
  Attr = Attr.addFnAttribute(C, Attribute::NoUnwind);
  Type *Int8PtrTy = IRB.getInt8PtrTy(kSpirOffloadConstantAS);

  // __tsan_cleanup_private(
  //   uptr ptr,
  //   size_t size
  // )
  TsanCleanupPrivate = M.getOrInsertFunction(
      "__tsan_cleanup_private", Attr, IRB.getVoidTy(), IntptrTy, IntptrTy);

  // __tsan_cleanup_static_local(
  //   uptr ptr,
  //   size_t size
  // )
  TsanCleanupStaticLocal = M.getOrInsertFunction(
      "__tsan_cleanup_static_local", Attr, IRB.getVoidTy(), IntptrTy, IntptrTy);

  // __tsan_cleanup_dynamic_local(
  //   uptr ptr,
  //   size_t size
  // )
  TsanCleanupDynamicLocal =
      M.getOrInsertFunction("__tsan_cleanup_dynamic_local", Attr,
                            IRB.getVoidTy(), IntptrTy, IRB.getInt32Ty());

  TsanDeviceBarrier = M.getOrInsertFunction(
      "__tsan_device_barrier", Attr.addFnAttribute(C, Attribute::Convergent),
      IRB.getVoidTy());

  TsanGroupBarrier = M.getOrInsertFunction(
      "__tsan_group_barrier", Attr.addFnAttribute(C, Attribute::Convergent),
      IRB.getVoidTy());

  for (size_t AddressSpaceIndex = 0; AddressSpaceIndex < kNumberOfAddressSpace;
       AddressSpaceIndex++) {
    for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
      const unsigned ByteSize = 1U << i;
      std::string ByteSizeStr = utostr(ByteSize);
      std::string Suffix = "_p" + itostr(AddressSpaceIndex);
      // __tsan_readX/__tsan_writeX(
      //   ...
      //   char* file,
      //   unsigned int line,
      //   char* func
      // )
      SmallString<32> ReadName("__tsan_read" + ByteSizeStr + Suffix);
      TsanRead[i][AddressSpaceIndex] =
          M.getOrInsertFunction(ReadName, Attr, IRB.getVoidTy(), IntptrTy,
                                Int8PtrTy, IRB.getInt32Ty(), Int8PtrTy);

      SmallString<32> WriteName("__tsan_write" + ByteSizeStr + Suffix);
      TsanWrite[i][AddressSpaceIndex] =
          M.getOrInsertFunction(WriteName, Attr, IRB.getVoidTy(), IntptrTy,
                                Int8PtrTy, IRB.getInt32Ty(), Int8PtrTy);

      SmallString<32> UnalignedReadName("__tsan_unaligned_read" + ByteSizeStr +
                                        Suffix);
      TsanUnalignedRead[i][AddressSpaceIndex] = M.getOrInsertFunction(
          UnalignedReadName, Attr, IRB.getVoidTy(), IntptrTy, Int8PtrTy,
          IRB.getInt32Ty(), Int8PtrTy);

      SmallString<32> UnalignedWriteName("__tsan_unaligned_write" +
                                         ByteSizeStr + Suffix);
      TsanUnalignedWrite[i][AddressSpaceIndex] = M.getOrInsertFunction(
          UnalignedWriteName, Attr, IRB.getVoidTy(), IntptrTy, Int8PtrTy,
          IRB.getInt32Ty(), Int8PtrTy);
    }
  }
}

bool ThreadSanitizerOnSpirv::instrumentAllocInst(
    Function *F, SmallVectorImpl<Instruction *> &AllocaInsts) {
  bool Changed = false;

  EscapeEnumerator EE(*F, "tsan_cleanup", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, *F);
    for (auto *Inst : AllocaInsts) {
      AllocaInst *AI = cast<AllocaInst>(Inst);
      // For dynamic allocas, sometime it will not dominate exit BB, we need to
      // skip them.
      if (!AI->isStaticAlloca())
        continue;

      if (auto AllocSize = AI->getAllocationSize(DL)) {
        AtExit->CreateCall(TsanCleanupPrivate,
                           {AtExit->CreatePtrToInt(AI, IntptrTy),
                            ConstantInt::get(IntptrTy, *AllocSize)});
        Changed |= true;
      }
    }
  }

  return Changed;
}

bool ThreadSanitizerOnSpirv::instrumentControlBarrier(CallInst *CI) {
  assert(isa<ConstantInt>(CI->getArgOperand(0)));
  uint64_t Scope = cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
  // is not device scope or work group scope
  if (Scope != 1 && Scope != 2)
    return false;

  InstrumentationIRBuilder IRB(CI);
  CallInst *NewCI =
      IRB.CreateCall(Scope == 1 ? TsanDeviceBarrier : TsanGroupBarrier, {});
  NewCI->setAttributes(NewCI->getCalledFunction()->getAttributes());
  CI->eraseFromParent();
  return true;
}

void ThreadSanitizerOnSpirv::appendDebugInfoToArgs(
    Instruction *I, SmallVectorImpl<Value *> &Args) {
  auto &Loc = I->getDebugLoc();

  // SPIR constant address space
  PointerType *ConstASPtrTy = PointerType::get(C, kSpirOffloadConstantAS);

  // File & Line
  if (Loc) {
    llvm::SmallString<128> Source = Loc->getDirectory();
    sys::path::append(Source, Loc->getFilename());
    auto *FileNameGV =
        GetOrCreateGlobalString("__tsan_file", Source, kSpirOffloadConstantAS);
    Args.push_back(ConstantExpr::getPointerCast(FileNameGV, ConstASPtrTy));
    Args.push_back(ConstantInt::get(Type::getInt32Ty(C), Loc.getLine()));
  } else {
    Args.push_back(ConstantPointerNull::get(ConstASPtrTy));
    Args.push_back(ConstantInt::get(Type::getInt32Ty(C), 0));
  }

  // Function
  auto FuncName = I->getFunction()->getName();
  auto *FuncNameGV = GetOrCreateGlobalString("__tsan_func", demangle(FuncName),
                                             kSpirOffloadConstantAS);
  Args.push_back(ConstantExpr::getPointerCast(FuncNameGV, ConstASPtrTy));
}

bool ThreadSanitizerOnSpirv::isUnsupportedSPIRAccess(Value *Addr,
                                                     Instruction *Inst) {
  auto *OrigValue = getUnderlyingObject(Addr);
  if (OrigValue->getName().starts_with("__spirv_BuiltIn"))
    return true;

  // Ignore load/store for target ext type since we can't know exactly what size
  // it is.
  if (auto *SI = dyn_cast<StoreInst>(Inst))
    if (getTargetExtType(SI->getValueOperand()->getType()) ||
        isJointMatrixAccess(SI->getPointerOperand()))
      return true;

  if (auto *LI = dyn_cast<LoadInst>(Inst))
    if (getTargetExtType(Inst->getType()) ||
        isJointMatrixAccess(LI->getPointerOperand()))
      return true;

  auto AddrAS = cast<PointerType>(Addr->getType()->getScalarType())
                    ->getPointerAddressSpace();
  switch (AddrAS) {
  case kSpirOffloadPrivateAS:
  case kSpirOffloadConstantAS:
    return true;
  case kSpirOffloadLocalAS:
    return !ClSpirOffloadLocals;
  case kSpirOffloadGlobalAS:
  case kSpirOffloadGenericAS:
    return false;
  }
  return false;
}

bool ThreadSanitizerOnSpirv::isSupportedSPIRKernel(Function &F) {

  if (!F.hasFnAttribute(Attribute::SanitizeThread) ||
      F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  if (F.getName().contains("__sycl_service_kernel__")) {
    F.addFnAttr(Attribute::DisableSanitizerInstrumentation);
    return false;
  }

  // Skip referenced-indirectly function as we insert access to shared
  // local memory (SLM) __TsanLaunchInfo and access to SLM in
  // referenced-indirectly function isn't supported yet in
  // intel-graphics-compiler.
  if (F.hasFnAttribute("referenced-indirectly")) {
    F.addFnAttr(Attribute::DisableSanitizerInstrumentation);
    return false;
  }

  return true;
}

bool ThreadSanitizerOnSpirv::isUnsupportedDeviceGlobal(
    const GlobalVariable &G) {
  if (G.user_empty())
    return true;
  // Skip instrumenting on "__TsanKernelMetadata" etc.
  if (G.getName().starts_with("__Tsan"))
    return true;
  if (G.getName().starts_with("__tsan_"))
    return true;
  if (G.getName().starts_with("__spirv_BuiltIn"))
    return true;
  if (G.getName().starts_with("__usid_str"))
    return true;
  // Global variables have constant address space will not trigger race
  // condition.
  if (G.getAddressSpace() == kSpirOffloadConstantAS)
    return true;
  return false;
}

void ThreadSanitizerOnSpirv::instrumentModule() {
  instrumentGlobalVariables();
  instrumentStaticLocalMemory();
  instrumentKernelsMetadata();
}

void ThreadSanitizerOnSpirv::instrumentGlobalVariables() {
  SmallVector<Constant *, 8> DeviceGlobalMetadata;

  // Device global metadata is described by a structure
  //  size_t device_global_size
  //  size_t beginning address of the device global
  StructType *StructTy = StructType::get(IntptrTy, IntptrTy);

  for (auto &G : M.globals()) {
    // DeviceSanitizers cannot handle nameless globals, therefore we set a name
    // for them so that we can handle them like regular globals.
    if (G.getName().empty() && G.hasInternalLinkage())
      G.setName("nameless_global");

    if (isUnsupportedDeviceGlobal(G)) {
      for (auto *User : G.users())
        if (auto *Inst = dyn_cast<Instruction>(User))
          Inst->setNoSanitizeMetadata();
      continue;
    }

    // This case is handled by instrumentStaticLocalMemory
    if (G.getAddressSpace() == kSpirOffloadLocalAS)
      continue;

    DeviceGlobalMetadata.push_back(ConstantStruct::get(
        StructTy,
        ConstantInt::get(IntptrTy, DL.getTypeAllocSize(G.getValueType())),
        ConstantExpr::getPointerCast(&G, IntptrTy)));
  }

  if (DeviceGlobalMetadata.empty())
    return;

  // Create meta data global to record device globals' information
  ArrayType *ArrayTy = ArrayType::get(StructTy, DeviceGlobalMetadata.size());
  Constant *MetadataInitializer =
      ConstantArray::get(ArrayTy, DeviceGlobalMetadata);
  GlobalVariable *MsanDeviceGlobalMetadata = new GlobalVariable(
      M, MetadataInitializer->getType(), false, GlobalValue::AppendingLinkage,
      MetadataInitializer, "__TsanDeviceGlobalMetadata", nullptr,
      GlobalValue::NotThreadLocal, 1);
  MsanDeviceGlobalMetadata->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
}

void ThreadSanitizerOnSpirv::instrumentStaticLocalMemory() {
  if (!ClSpirOffloadLocals)
    return;

  auto Instrument = [this](GlobalVariable *G, Function *F) {
    const uint64_t SizeInBytes = DL.getTypeAllocSize(G->getValueType());

    if (!F->hasMetadata("tsan_instrumented_local")) {
      IRBuilder<> Builder(&F->getEntryBlock().front());
      Builder.CreateCall(TsanGroupBarrier);
    }

    // Poison shadow of static local memory
    {
      IRBuilder<> Builder(&F->getEntryBlock().front());
      Builder.CreateCall(TsanCleanupStaticLocal,
                         {Builder.CreatePointerCast(G, IntptrTy),
                          ConstantInt::get(IntptrTy, SizeInBytes)});
    }

    // Unpoison shadow of static local memory, required by CPU device
    EscapeEnumerator EE(*F, "tsan_cleanup_static_local", false);
    while (IRBuilder<> *AtExit = EE.Next()) {
      if (!F->hasMetadata("tsan_instrumented_local"))
        AtExit->CreateCall(TsanGroupBarrier);
      AtExit->CreateCall(TsanCleanupStaticLocal,
                         {AtExit->CreatePointerCast(G, IntptrTy),
                          ConstantInt::get(IntptrTy, SizeInBytes)});
    }

    if (!F->hasMetadata("tsan_instrumented_local")) {
      Constant *One = ConstantInt::get(Type::getInt32Ty(C), 1);
      MDNode *NewNode = MDNode::get(C, ConstantAsMetadata::get(One));
      F->addMetadata("tsan_instrumented_local", *NewNode);
    }
  };

  // We only instrument on spir_kernel, because local variables are
  // kind of global variable, which must be initialized only once.
  for (auto &G : M.globals()) {
    if (G.getAddressSpace() == kSpirOffloadLocalAS) {
      SmallVector<Function *> WorkList;
      DenseSet<Function *> InstrumentedKernel;
      for (auto *User : G.users())
        getFunctionsOfUser(User, WorkList);
      while (!WorkList.empty()) {
        Function *F = WorkList.pop_back_val();
        if (F->getCallingConv() == CallingConv::SPIR_KERNEL) {
          if (!InstrumentedKernel.contains(F)) {
            Instrument(&G, F);
            InstrumentedKernel.insert(F);
          }
          continue;
        }
        // Get root spir_kernel of spir_func
        initializeKernelCallerMap(F);
        for (auto *F : FuncToKernelCallerMap[F])
          WorkList.push_back(F);
      }
    }
  }
}

void ThreadSanitizerOnSpirv::instrumentDynamicLocalMemory(Function &F) {
  if (!ClSpirOffloadLocals)
    return;

  // Poison shadow of local memory in kernel argument, required by CPU device
  SmallVector<Argument *> LocalArgs;
  for (auto &Arg : F.args()) {
    Type *PtrTy = dyn_cast<PointerType>(Arg.getType()->getScalarType());
    if (PtrTy && PtrTy->getPointerAddressSpace() == kSpirOffloadLocalAS)
      LocalArgs.push_back(&Arg);
  }

  if (LocalArgs.empty())
    return;

  if (!F.hasMetadata("tsan_instrumented_local")) {
    IRBuilder<> Builder(&F.getEntryBlock().front());
    Builder.CreateCall(TsanGroupBarrier);
  }

  IRBuilder<> IRB(&F.getEntryBlock().front());

  AllocaInst *ArgsArray = IRB.CreateAlloca(
      IntptrTy, ConstantInt::get(IRB.getInt32Ty(), LocalArgs.size()),
      "local_args");
  for (size_t i = 0; i < LocalArgs.size(); i++) {
    auto *StoreDest = IRB.CreateGEP(IntptrTy, ArgsArray,
                                    ConstantInt::get(IRB.getInt32Ty(), i));
    IRB.CreateStore(IRB.CreatePointerCast(LocalArgs[i], IntptrTy), StoreDest);
  }

  auto *ArgsArrayAddr = IRB.CreatePointerCast(ArgsArray, IntptrTy);
  IRB.CreateCall(
      TsanCleanupDynamicLocal,
      {ArgsArrayAddr, ConstantInt::get(IRB.getInt32Ty(), LocalArgs.size())});

  // Unpoison shadow of dynamic local memory, required by CPU device
  EscapeEnumerator EE(F, "tsan_cleanup_dynamic_local", false);
  while (IRBuilder<> *AtExit = EE.Next()) {
    if (!F.hasMetadata("tsan_instrumented_local"))
      AtExit->CreateCall(TsanGroupBarrier);
    AtExit->CreateCall(TsanCleanupDynamicLocal,
                       {ArgsArrayAddr, ConstantInt::get(AtExit->getInt32Ty(),
                                                        LocalArgs.size())});
  }

  if (!F.hasMetadata("tsan_instrumented_local")) {
    Constant *One = ConstantInt::get(Type::getInt32Ty(C), 1);
    MDNode *NewNode = MDNode::get(C, ConstantAsMetadata::get(One));
    F.addMetadata("tsan_instrumented_local", *NewNode);
  }
}

void ThreadSanitizerOnSpirv::initializeKernelCallerMap(Function *F) {
  if (FuncToKernelCallerMap.find(F) != FuncToKernelCallerMap.end())
    return;

  for (auto *U : F->users()) {
    if (Instruction *Inst = dyn_cast<Instruction>(U)) {
      Function *Caller = Inst->getFunction();
      if (Caller->getCallingConv() == CallingConv::SPIR_KERNEL) {
        FuncToKernelCallerMap[F].insert(Caller);
        continue;
      }
      initializeKernelCallerMap(Caller);
      FuncToKernelCallerMap[F].insert(FuncToKernelCallerMap[Caller].begin(),
                                      FuncToKernelCallerMap[Caller].end());
    }
  }
}

void ThreadSanitizerOnSpirv::instrumentKernelsMetadata() {
  SmallVector<Constant *, 8> SpirKernelsMetadata;
  SmallVector<uint8_t, 256> KernelNamesBytes;

  // SpirKernelsMetadata only saves fixed kernels, and is described by
  // following structure:
  //  uptr unmangled_kernel_name
  //  uptr unmangled_kernel_name_size
  StructType *StructTy = StructType::get(IntptrTy, IntptrTy);

  for (Function &F : M) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    if (isSupportedSPIRKernel(F)) {
      auto KernelName = F.getName();
      KernelNamesBytes.append(KernelName.begin(), KernelName.end());
      auto *KernelNameGV = GetOrCreateGlobalString("__tsan_kernel", KernelName,
                                                   kSpirOffloadConstantAS);
      SpirKernelsMetadata.emplace_back(ConstantStruct::get(
          StructTy, ConstantExpr::getPointerCast(KernelNameGV, IntptrTy),
          ConstantInt::get(IntptrTy, KernelName.size())));
    }
  }

  // Create global variable to record spirv kernels' information
  ArrayType *ArrayTy = ArrayType::get(StructTy, SpirKernelsMetadata.size());
  Constant *MetadataInitializer =
      ConstantArray::get(ArrayTy, SpirKernelsMetadata);
  GlobalVariable *TsanSpirKernelMetadata = new GlobalVariable(
      M, MetadataInitializer->getType(), false, GlobalValue::AppendingLinkage,
      MetadataInitializer, "__TsanKernelMetadata", nullptr,
      GlobalValue::NotThreadLocal, 1);
  TsanSpirKernelMetadata->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
  // Add device global attributes
  TsanSpirKernelMetadata->addAttribute(
      "sycl-device-global-size", std::to_string(DL.getTypeAllocSize(ArrayTy)));
  TsanSpirKernelMetadata->addAttribute("sycl-device-image-scope");
  TsanSpirKernelMetadata->addAttribute("sycl-host-access", "0"); // read only
  TsanSpirKernelMetadata->addAttribute(
      "sycl-unique-id",
      computeKernelMetadataUniqueId("__TsanKernelMetadata", KernelNamesBytes));
  TsanSpirKernelMetadata->setDSOLocal(true);
}

GlobalVariable *
ThreadSanitizerOnSpirv::GetOrCreateGlobalString(StringRef Name, StringRef Value,
                                                unsigned AddressSpace) {
  GlobalVariable *StringGV = nullptr;
  if (GlobalStringMap.find(Value.str()) != GlobalStringMap.end())
    return GlobalStringMap.at(Value.str());

  auto *Ty = ArrayType::get(Type::getInt8Ty(M.getContext()), Value.size() + 1);
  StringGV = new GlobalVariable(
      M, Ty, true, GlobalValue::InternalLinkage,
      ConstantDataArray::getString(M.getContext(), Value), Name, nullptr,
      GlobalValue::NotThreadLocal, AddressSpace);
  GlobalStringMap[Value.str()] = StringGV;

  return StringGV;
}

void ThreadSanitizer::initialize(Module &M, const TargetLibraryInfo &TLI) {
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  IntptrTy = DL.getIntPtrType(Ctx);
  if (Triple(M.getTargetTriple()).isSPIROrSPIRV()) {
    Spirv.emplace(M);
    Spirv->initialize();
  }

  IRBuilder<> IRB(Ctx);
  AttributeList Attr;
  Attr = Attr.addFnAttribute(Ctx, Attribute::NoUnwind);
  // Initialize the callbacks.
  TsanFuncEntry = M.getOrInsertFunction("__tsan_func_entry", Attr,
                                        IRB.getVoidTy(), IRB.getPtrTy());
  TsanFuncExit =
      M.getOrInsertFunction("__tsan_func_exit", Attr, IRB.getVoidTy());
  TsanIgnoreBegin = M.getOrInsertFunction("__tsan_ignore_thread_begin", Attr,
                                          IRB.getVoidTy());
  TsanIgnoreEnd =
      M.getOrInsertFunction("__tsan_ignore_thread_end", Attr, IRB.getVoidTy());
  IntegerType *OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const unsigned ByteSize = 1U << i;
    const unsigned BitSize = ByteSize * 8;
    std::string ByteSizeStr = utostr(ByteSize);
    std::string BitSizeStr = utostr(BitSize);
    SmallString<32> ReadName("__tsan_read" + ByteSizeStr);
    TsanRead[i] =
        M.getOrInsertFunction(ReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<32> WriteName("__tsan_write" + ByteSizeStr);
    TsanWrite[i] =
        M.getOrInsertFunction(WriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedReadName("__tsan_unaligned_read" + ByteSizeStr);
    TsanUnalignedRead[i] = M.getOrInsertFunction(
        UnalignedReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedWriteName("__tsan_unaligned_write" + ByteSizeStr);
    TsanUnalignedWrite[i] = M.getOrInsertFunction(
        UnalignedWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> VolatileReadName("__tsan_volatile_read" + ByteSizeStr);
    TsanVolatileRead[i] = M.getOrInsertFunction(
        VolatileReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> VolatileWriteName("__tsan_volatile_write" + ByteSizeStr);
    TsanVolatileWrite[i] = M.getOrInsertFunction(
        VolatileWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedVolatileReadName("__tsan_unaligned_volatile_read" +
                                              ByteSizeStr);
    TsanUnalignedVolatileRead[i] = M.getOrInsertFunction(
        UnalignedVolatileReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedVolatileWriteName(
        "__tsan_unaligned_volatile_write" + ByteSizeStr);
    TsanUnalignedVolatileWrite[i] = M.getOrInsertFunction(
        UnalignedVolatileWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> CompoundRWName("__tsan_read_write" + ByteSizeStr);
    TsanCompoundRW[i] = M.getOrInsertFunction(CompoundRWName, Attr,
                                              IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedCompoundRWName("__tsan_unaligned_read_write" +
                                            ByteSizeStr);
    TsanUnalignedCompoundRW[i] = M.getOrInsertFunction(
        UnalignedCompoundRWName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    Type *Ty = Type::getIntNTy(Ctx, BitSize);
    Type *PtrTy = PointerType::get(Ctx, 0);
    SmallString<32> AtomicLoadName("__tsan_atomic" + BitSizeStr + "_load");
    TsanAtomicLoad[i] =
        M.getOrInsertFunction(AtomicLoadName,
                              TLI.getAttrList(&Ctx, {1}, /*Signed=*/true,
                                              /*Ret=*/BitSize <= 32, Attr),
                              Ty, PtrTy, OrdTy);

    // Args of type Ty need extension only when BitSize is 32 or less.
    using Idxs = std::vector<unsigned>;
    Idxs Idxs2Or12((BitSize <= 32) ? Idxs({1, 2}) : Idxs({2}));
    Idxs Idxs34Or1234((BitSize <= 32) ? Idxs({1, 2, 3, 4}) : Idxs({3, 4}));
    SmallString<32> AtomicStoreName("__tsan_atomic" + BitSizeStr + "_store");
    TsanAtomicStore[i] = M.getOrInsertFunction(
        AtomicStoreName,
        TLI.getAttrList(&Ctx, Idxs2Or12, /*Signed=*/true, /*Ret=*/false, Attr),
        IRB.getVoidTy(), PtrTy, Ty, OrdTy);

    for (unsigned Op = AtomicRMWInst::FIRST_BINOP;
         Op <= AtomicRMWInst::LAST_BINOP; ++Op) {
      TsanAtomicRMW[Op][i] = nullptr;
      const char *NamePart = nullptr;
      if (Op == AtomicRMWInst::Xchg)
        NamePart = "_exchange";
      else if (Op == AtomicRMWInst::Add)
        NamePart = "_fetch_add";
      else if (Op == AtomicRMWInst::Sub)
        NamePart = "_fetch_sub";
      else if (Op == AtomicRMWInst::And)
        NamePart = "_fetch_and";
      else if (Op == AtomicRMWInst::Or)
        NamePart = "_fetch_or";
      else if (Op == AtomicRMWInst::Xor)
        NamePart = "_fetch_xor";
      else if (Op == AtomicRMWInst::Nand)
        NamePart = "_fetch_nand";
      else
        continue;
      SmallString<32> RMWName("__tsan_atomic" + itostr(BitSize) + NamePart);
      TsanAtomicRMW[Op][i] = M.getOrInsertFunction(
          RMWName,
          TLI.getAttrList(&Ctx, Idxs2Or12, /*Signed=*/true,
                          /*Ret=*/BitSize <= 32, Attr),
          Ty, PtrTy, Ty, OrdTy);
    }

    SmallString<32> AtomicCASName("__tsan_atomic" + BitSizeStr +
                                  "_compare_exchange_val");
    TsanAtomicCAS[i] = M.getOrInsertFunction(
        AtomicCASName,
        TLI.getAttrList(&Ctx, Idxs34Or1234, /*Signed=*/true,
                        /*Ret=*/BitSize <= 32, Attr),
        Ty, PtrTy, Ty, Ty, OrdTy, OrdTy);
  }
  TsanVptrUpdate =
      M.getOrInsertFunction("__tsan_vptr_update", Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), IRB.getPtrTy());
  TsanVptrLoad = M.getOrInsertFunction("__tsan_vptr_read", Attr,
                                       IRB.getVoidTy(), IRB.getPtrTy());
  TsanAtomicThreadFence = M.getOrInsertFunction(
      "__tsan_atomic_thread_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getVoidTy(), OrdTy);

  TsanAtomicSignalFence = M.getOrInsertFunction(
      "__tsan_atomic_signal_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getVoidTy(), OrdTy);

  MemmoveFn = M.getOrInsertFunction("__tsan_memmove", Attr, IRB.getPtrTy(),
                                    IRB.getPtrTy(), IRB.getPtrTy(), IntptrTy);
  MemcpyFn = M.getOrInsertFunction("__tsan_memcpy", Attr, IRB.getPtrTy(),
                                   IRB.getPtrTy(), IRB.getPtrTy(), IntptrTy);
  MemsetFn = M.getOrInsertFunction(
      "__tsan_memset",
      TLI.getAttrList(&Ctx, {1}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getPtrTy(), IRB.getPtrTy(), IRB.getInt32Ty(), IntptrTy);
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

// Do not instrument known races/"benign races" that come from compiler
// instrumentatin. The user has no way of suppressing them.
static bool shouldInstrumentReadWriteFromAddress(const Module *M, Value *Addr) {
  // Peel off GEPs and BitCasts.
  Addr = Addr->stripInBoundsOffsets();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      // Check if the global is in the PGO counters section.
      auto OF = M->getTargetTriple().getObjectFormat();
      if (SectionName.ends_with(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }
  }

  // Do not instrument accesses from different address spaces; we cannot deal
  // with them.
  if (Addr) {
    Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0)
      return false;
  }

  return true;
}

bool ThreadSanitizer::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      // Reads from constant globals can not race with any writes.
      NumOmittedReadsFromConstantGlobals++;
      return true;
    }
  } else if (LoadInst *L = dyn_cast<LoadInst>(Addr)) {
    if (isVtableAccess(L)) {
      // Reads from a vtable pointer can not race with any writes.
      NumOmittedReadsFromVtable++;
      return true;
    }
  }
  return false;
}

// Instrumenting some of the accesses may be proven redundant.
// Currently handled:
//  - read-before-write (within same BB, no calls between)
//  - not captured variables
//
// We do not handle some of the patterns that should not survive
// after the classic compiler optimizations.
// E.g. two reads from the same temp should be eliminated by CSE,
// two writes should be eliminated by DSE, etc.
//
// 'Local' is a vector of insns within the same BB (no calls between).
// 'All' is a vector of insns that will be instrumented.
void ThreadSanitizer::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local,
    SmallVectorImpl<InstructionInfo> &All, const DataLayout &DL) {
  DenseMap<Value *, size_t> WriteTargets; // Map of addresses to index in All
  // Iterate from the end.
  for (Instruction *I : reverse(Local)) {
    const bool IsWrite = isa<StoreInst>(*I);
    Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                          : cast<LoadInst>(I)->getPointerOperand();

    if (Spirv) {
      if (Spirv->isUnsupportedSPIRAccess(Addr, I))
        continue;
    } else if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
      continue;

    if (!IsWrite) {
      const auto WriteEntry = WriteTargets.find(Addr);
      if (!ClInstrumentReadBeforeWrite && WriteEntry != WriteTargets.end()) {
        auto &WI = All[WriteEntry->second];
        // If we distinguish volatile accesses and if either the read or write
        // is volatile, do not omit any instrumentation.
        const bool AnyVolatile =
            ClDistinguishVolatile && (cast<LoadInst>(I)->isVolatile() ||
                                      cast<StoreInst>(WI.Inst)->isVolatile());
        if (!AnyVolatile) {
          // We will write to this temp, so no reason to analyze the read.
          // Mark the write instruction as compound.
          WI.Flags |= InstructionInfo::kCompoundRW;
          NumOmittedReadsBeforeWrite++;
          continue;
        }
      }

      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        continue;
      }
    }

    const AllocaInst *AI = findAllocaForValue(Addr);
    // Instead of Addr, we should check whether its base pointer is captured.
    if (AI && !PointerMayBeCaptured(AI, /*ReturnCaptures=*/true)) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different thread and participate in a data race
      // (see llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }

    // Instrument this instruction.
    All.emplace_back(I);
    if (IsWrite) {
      // For read-before-write and compound instrumentation we only need one
      // write target, and we can override any previous entry if it exists.
      WriteTargets[Addr] = All.size() - 1;
    }
  }
  Local.clear();
}

static bool isTsanAtomic(const Instruction *I) {
  // TODO: Ask TTI whether synchronization scope is between threads.
  auto SSID = getAtomicSyncScopeID(I);
  if (!SSID)
    return false;
  if (isa<LoadInst>(I) || isa<StoreInst>(I))
    return *SSID != SyncScope::SingleThread;
  return true;
}

void ThreadSanitizer::InsertRuntimeIgnores(Function &F) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  IRB.CreateCall(TsanIgnoreBegin);
  EscapeEnumerator EE(F, "tsan_ignore_cleanup", ClHandleCxxExceptions);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(TsanIgnoreEnd);
  }
}

bool ThreadSanitizer::sanitizeFunction(Function &F,
                                       const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __tsan_init from within
  // the module constructor.
  if (F.getName() == kTsanModuleCtorName)
    return false;
  // Naked functions can not have prologue/epilogue
  // (__tsan_func_entry/__tsan_func_exit) generated, so don't instrument them at
  // all.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  // __attribute__(disable_sanitizer_instrumentation) prevents all kinds of
  // instrumentation.
  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  initialize(*F.getParent(), TLI);
  SmallVector<InstructionInfo, 8> AllLoadsAndStores;
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  SmallVector<Instruction *, 8> Allocas;
  SmallVector<CallInst *, 8> SpirControlBarrierCalls;
  bool Res = false;
  bool HasCalls = false;
  bool SanitizeFunction = F.hasFnAttribute(Attribute::SanitizeThread);
  const DataLayout &DL = F.getDataLayout();

  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      // Skip instructions inserted by another instrumentation.
      if (Inst.hasMetadata(LLVMContext::MD_nosanitize))
        continue;
      if (isTsanAtomic(&Inst))
        AtomicAccesses.push_back(&Inst);
      else if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (Spirv && isa<AllocaInst>(Inst) &&
               cast<AllocaInst>(Inst).getAllocatedType()->isSized() &&
               !getTargetExtType(cast<AllocaInst>(Inst).getAllocatedType()))
        Allocas.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);
          if (Spirv) {
            Function *CalledFn = CI->getCalledFunction();
            if (CalledFn &&
                CalledFn->getName() == "_Z22__spirv_ControlBarrieriii") {
              SpirControlBarrierCalls.push_back(CI);
            }
          }
        }
        if (isa<MemIntrinsic>(Inst))
          MemIntrinCalls.push_back(&Inst);
        HasCalls = true;
        chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores,
                                       DL);
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, DL);
  }

  // We have collected all loads and stores.
  // FIXME: many of these accesses do not need to be checked for races
  // (e.g. variables that do not escape, etc).

  // Instrument memory accesses only if we want to report bugs in the function.
  if (ClInstrumentMemoryAccesses && SanitizeFunction)
    for (const auto &II : AllLoadsAndStores) {
      Res |= instrumentLoadOrStore(II, DL);
    }

  // Instrument atomic memory accesses in any case (they can be used to
  // implement synchronization).
  // TODO: Disable atomics check for spirv target temporarily, will support it
  // later.
  if (!Spirv && ClInstrumentAtomics)
    for (auto *Inst : AtomicAccesses) {
      Res |= instrumentAtomic(Inst, DL);
    }

  if (ClInstrumentMemIntrinsics && SanitizeFunction)
    for (auto *Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(Inst);
    }

  if (F.hasFnAttribute("sanitize_thread_no_checking_at_run_time")) {
    assert(!F.hasFnAttribute(Attribute::SanitizeThread));
    if (HasCalls)
      InsertRuntimeIgnores(F);
  }

  if (Spirv)
    for (auto *CI : SpirControlBarrierCalls) {
      Res |= Spirv->instrumentControlBarrier(CI);
    }

  // FIXME: We need to skip the check for private memory, otherwise OpenCL CPU
  // device may generate false positive reports due to stack re-use in different
  // threads. However, SPIR-V builts 'ToPrivate' doesn't work as expected on
  // OpenCL CPU device. So we need to manually cleanup private shadow before
  // each function exit point.
  if (Spirv && !Allocas.empty())
    Res |= Spirv->instrumentAllocInst(&F, Allocas);

  // Instrument function entry/exit points if there were instrumented accesses.
  if ((Res || HasCalls) && ClInstrumentFuncEntryExit) {
    InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                                 F.getEntryBlock().getFirstNonPHIIt());
    Value *ReturnAddress =
        IRB.CreateIntrinsic(Intrinsic::returnaddress, IRB.getInt32(0));
    IRB.CreateCall(TsanFuncEntry, ReturnAddress);

    EscapeEnumerator EE(F, "tsan_cleanup", ClHandleCxxExceptions);
    while (IRBuilder<> *AtExit = EE.Next()) {
      InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
      AtExit->CreateCall(TsanFuncExit, {});
    }
    Res = true;
  }

  if (Spirv && F.getCallingConv() == CallingConv::SPIR_KERNEL)
    Spirv->instrumentDynamicLocalMemory(F);
  return Res;
}

bool ThreadSanitizer::instrumentLoadOrStore(const InstructionInfo &II,
                                            const DataLayout &DL) {
  InstrumentationIRBuilder IRB(II.Inst);
  const bool IsWrite = isa<StoreInst>(*II.Inst);
  Value *Addr = IsWrite ? cast<StoreInst>(II.Inst)->getPointerOperand()
                        : cast<LoadInst>(II.Inst)->getPointerOperand();
  Type *OrigTy = getLoadStoreType(II.Inst);

  // swifterror memory addresses are mem2reg promoted by instruction selection.
  // As such they cannot have regular uses like an instrumentation function and
  // it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;

  int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
  if (Idx < 0)
    return false;
  // There is no race-free access scenario to vtable for spirv target.
  if (!Spirv && IsWrite && isVtableAccess(II.Inst)) {
    LLVM_DEBUG(dbgs() << "  VPTR : " << *II.Inst << "\n");
    Value *StoredValue = cast<StoreInst>(II.Inst)->getValueOperand();
    // StoredValue may be a vector type if we are storing several vptrs at once.
    // In this case, just take the first element of the vector since this is
    // enough to find vptr races.
    if (isa<VectorType>(StoredValue->getType()))
      StoredValue = IRB.CreateExtractElement(
          StoredValue, ConstantInt::get(IRB.getInt32Ty(), 0));
    if (StoredValue->getType()->isIntegerTy())
      StoredValue = IRB.CreateIntToPtr(StoredValue, IRB.getPtrTy());
    // Call TsanVptrUpdate.
    IRB.CreateCall(TsanVptrUpdate, {Addr, StoredValue});
    NumInstrumentedVtableWrites++;
    return true;
  }
  if (!Spirv && !IsWrite && isVtableAccess(II.Inst)) {
    IRB.CreateCall(TsanVptrLoad, Addr);
    NumInstrumentedVtableReads++;
    return true;
  }

  const Align Alignment = IsWrite ? cast<StoreInst>(II.Inst)->getAlign()
                                  : cast<LoadInst>(II.Inst)->getAlign();
  const bool IsCompoundRW =
      ClCompoundReadBeforeWrite && (II.Flags & InstructionInfo::kCompoundRW);
  const bool IsVolatile = ClDistinguishVolatile &&
                          (IsWrite ? cast<StoreInst>(II.Inst)->isVolatile()
                                   : cast<LoadInst>(II.Inst)->isVolatile());
  assert((!IsVolatile || !IsCompoundRW) && "Compound volatile invalid!");

  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  const unsigned int AS = cast<PointerType>(Addr->getType()->getScalarType())
                              ->getPointerAddressSpace();
  FunctionCallee OnAccessFunc = nullptr;
  if (Alignment >= Align(8) || (Alignment.value() % (TypeSize / 8)) == 0) {
    if (IsCompoundRW)
      OnAccessFunc = TsanCompoundRW[Idx];
    else if (IsVolatile)
      OnAccessFunc = IsWrite ? TsanVolatileWrite[Idx] : TsanVolatileRead[Idx];
    else if (Spirv)
      OnAccessFunc =
          IsWrite ? Spirv->TsanWrite[Idx][AS] : Spirv->TsanRead[Idx][AS];
    else
      OnAccessFunc = IsWrite ? TsanWrite[Idx] : TsanRead[Idx];
  } else {
    if (IsCompoundRW)
      OnAccessFunc = TsanUnalignedCompoundRW[Idx];
    else if (IsVolatile)
      OnAccessFunc = IsWrite ? TsanUnalignedVolatileWrite[Idx]
                             : TsanUnalignedVolatileRead[Idx];
    else if (Spirv)
      OnAccessFunc = IsWrite ? Spirv->TsanUnalignedWrite[Idx][AS]
                             : Spirv->TsanUnalignedRead[Idx][AS];
    else
      OnAccessFunc = IsWrite ? TsanUnalignedWrite[Idx] : TsanUnalignedRead[Idx];
  }
  if (Spirv) {
    SmallVector<Value *, 5> Args;
    Args.push_back(IRB.CreatePointerCast(Addr, IntptrTy));
    Spirv->appendDebugInfoToArgs(II.Inst, Args);
    IRB.CreateCall(OnAccessFunc, Args);
  } else
    IRB.CreateCall(OnAccessFunc, Addr);
  if (IsCompoundRW || IsWrite)
    NumInstrumentedWrites++;
  if (IsCompoundRW || !IsWrite)
    NumInstrumentedReads++;
  return true;
}

static ConstantInt *createOrdering(IRBuilder<> *IRB, AtomicOrdering ord) {
  uint32_t v = 0;
  switch (ord) {
  case AtomicOrdering::NotAtomic:
    llvm_unreachable("unexpected atomic ordering!");
  case AtomicOrdering::Unordered:
    [[fallthrough]];
  case AtomicOrdering::Monotonic:
    v = 0;
    break;
  // Not specified yet:
  // case AtomicOrdering::Consume:                v = 1; break;
  case AtomicOrdering::Acquire:
    v = 2;
    break;
  case AtomicOrdering::Release:
    v = 3;
    break;
  case AtomicOrdering::AcquireRelease:
    v = 4;
    break;
  case AtomicOrdering::SequentiallyConsistent:
    v = 5;
    break;
  }
  return IRB->getInt32(v);
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since tsan is running after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __tsan_memset to avoid the intrinsics.
bool ThreadSanitizer::instrumentMemIntrinsic(Instruction *I) {
  InstrumentationIRBuilder IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    Value *Cast1 =
        IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false);
    Value *Cast2 = IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false);
    IRB.CreateCall(MemsetFn, {M->getArgOperand(0), Cast1, Cast2});
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall(isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
                   {M->getArgOperand(0), M->getArgOperand(1),
                    IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  }
  return false;
}

// Both llvm and ThreadSanitizer atomic operations are based on C++11/C1x
// standards.  For background see C++11 standard.  A slightly older, publicly
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

bool ThreadSanitizer::instrumentAtomic(Instruction *I, const DataLayout &DL) {
  InstrumentationIRBuilder IRB(I);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Addr = LI->getPointerOperand();
    Type *OrigTy = LI->getType();
    int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
    if (Idx < 0)
      return false;
    Value *Args[] = {Addr, createOrdering(&IRB, LI->getOrdering())};
    Value *C = IRB.CreateCall(TsanAtomicLoad[Idx], Args);
    Value *Cast = IRB.CreateBitOrPointerCast(C, OrigTy);
    I->replaceAllUsesWith(Cast);
    I->eraseFromParent();
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Addr = SI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(SI->getValueOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *Args[] = {Addr,
                     IRB.CreateBitOrPointerCast(SI->getValueOperand(), Ty),
                     createOrdering(&IRB, SI->getOrdering())};
    IRB.CreateCall(TsanAtomicStore[Idx], Args);
    SI->eraseFromParent();
  } else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Value *Addr = RMWI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(RMWI->getValOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    FunctionCallee F = TsanAtomicRMW[RMWI->getOperation()][Idx];
    if (!F)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *Val = RMWI->getValOperand();
    Value *Args[] = {Addr, IRB.CreateBitOrPointerCast(Val, Ty),
                     createOrdering(&IRB, RMWI->getOrdering())};
    Value *C = IRB.CreateCall(F, Args);
    I->replaceAllUsesWith(IRB.CreateBitOrPointerCast(C, Val->getType()));
    I->eraseFromParent();
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Addr = CASI->getPointerOperand();
    Type *OrigOldValTy = CASI->getNewValOperand()->getType();
    int Idx = getMemoryAccessFuncIndex(OrigOldValTy, Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *CmpOperand =
        IRB.CreateBitOrPointerCast(CASI->getCompareOperand(), Ty);
    Value *NewOperand =
        IRB.CreateBitOrPointerCast(CASI->getNewValOperand(), Ty);
    Value *Args[] = {Addr, CmpOperand, NewOperand,
                     createOrdering(&IRB, CASI->getSuccessOrdering()),
                     createOrdering(&IRB, CASI->getFailureOrdering())};
    CallInst *C = IRB.CreateCall(TsanAtomicCAS[Idx], Args);
    Value *Success = IRB.CreateICmpEQ(C, CmpOperand);
    Value *OldVal = C;
    if (Ty != OrigOldValTy) {
      // The value is a pointer, so we need to cast the return value.
      OldVal = IRB.CreateIntToPtr(C, OrigOldValTy);
    }

    Value *Res =
        IRB.CreateInsertValue(PoisonValue::get(CASI->getType()), OldVal, 0);
    Res = IRB.CreateInsertValue(Res, Success, 1);

    I->replaceAllUsesWith(Res);
    I->eraseFromParent();
  } else if (FenceInst *FI = dyn_cast<FenceInst>(I)) {
    Value *Args[] = {createOrdering(&IRB, FI->getOrdering())};
    FunctionCallee F = FI->getSyncScopeID() == SyncScope::SingleThread
                           ? TsanAtomicSignalFence
                           : TsanAtomicThreadFence;
    IRB.CreateCall(F, Args);
    FI->eraseFromParent();
  }
  return true;
}

int ThreadSanitizer::getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr,
                                              const DataLayout &DL) {
  assert(OrigTy->isSized());
  if (OrigTy->isScalableTy()) {
    // FIXME: support vscale.
    return -1;
  }
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8 && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 &&
      TypeSize != 128) {
    NumAccessesWithBadSize++;
    // Ignore all unusual sizes.
    return -1;
  }
  size_t Idx = llvm::countr_zero(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  return Idx;
}
