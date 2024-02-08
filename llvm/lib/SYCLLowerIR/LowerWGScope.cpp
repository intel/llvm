//===-- LowerWGScope.cpp - lower work group scope code and locals ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Simple work group (WG) code and data lowering pass. SYCL specification
// requires that
// - the code in the parallel_for_work_group (PFWG) but outside the
//    parallel_for_work_item (PFWI) (called "work group scope" further here) is
//    executed once per WG
// - data declared at the work group scope is shared among work items (WIs) in
//    a WG.
// - private_memory<T> data declared at the work group scope remains private per
//   physical work item and lives across all parallel_for_work_item invocations
//
// To enforce this semantics, this pass
// - inserts "if (get_local_id(0)) == 0" guards ("is leader" guard) to disable
//   WG-scope code execution in "worker" WIs
// - transforms allocas in the PFWG lambda function; the function is identified
//   by the "work_group_scope" string metadata added by the Front End
//
// There are 3 kinds of local variables in the PFWG lambda function which are
// handled differently by the compiler:
// 1) Local variables of type private_memory<T> declared by the user. FE marks
//    allocas created for them with "work_item_scope" string metadata.
// 2) Other local variables declared by the user (shared). Front end turns them
//    into globals in the local address space - work group shared locals. There
//    are no allocas for them in the PFWG lambda.
// 3) Compiler-generated locals:
//    - the PFWI lambda object (1 per PFWI) which captures variables passed into
//      the PFWI lambda
//    - a local copy of the PFWG lambda object parameter passed by value into
//      the PFWG lambda
//
// ** Kind 2: no further transformations are needed for kind 2.
// ** Kind 3:
// For a kind 3 variable (alloca w/o metadata) this pass creates a WG-shared
// local "shadow" variable. Before each PFWI invocation leader WI stores its
// private copy of the variable into the shadow (under "is leader" guard), then
// all WIs (outside of "is leader" guard) load the shadow value into their
// private copies ("materialize" the private copy). This works because these
// variables are uniform - i.e. have the same value in all WIs and are not
// changed within PFWI. The only exceptions are captures of private_memory
// instances - see next.
// ** Kind 1:
// Even though WG-scope locals are supposed to be uniform, there is one
// exception - capture of local of kind 1. It is always captured by non-const
// reference because as there no
// 'const T &operator()(const h_item<Dimensions> &id);' which means the result
// of kind 1 variable's alloca is stored within the PFWI lambda.
// Materialization of the lambda object value writes result of alloca of the
// leader WI's private variable into the private copy of the lambda object,
// which is wrong. So for these variables this pass adds a write of the private
// variable's address into the private copy of the lambda object right after its
// materialization:
//     if (is_leader())
//       *PFWI_lambda_obj_shadow_addr = *PFWI_lambda_obj_alloca;
//     barrier();
// (1) *PFWI_lambda_obj_alloca = *PFWI_lambda_obj_shadow_addr;
// (2) PFWI_lambda_obj_alloca->priv_var_addr = priv_var_alloca;
//     parallel_for_work_item(..., PFWI_lambda_obj_alloca);
//
// (1) - materialization of a PFWI object
// (2) - "fixup" of the private variable address.
//
// TODO: add support for the case when there are other functions between
// parallel_for_work_group and parallel_for_work_item in the call stack.
// For example:
//
// void foo(sycl::group<1> group, ...) {
//   group.parallel_for_work_item(range<1>(), [&](h_item<1> i) { ... });
// }
// ...
//   cgh.parallel_for_work_group<class kernel>(
//     range<1>(...), range<1>(...), [=](group<1> g) {
//       foo(g, ...);
//     });
//
// TODO The approach employed by this pass generates lots of barriers and data
// copying between private and local memory, which might not be efficient. There
// are optimization opportunities listed below. Also other approaches can be
// considered like
// "Efficient Fork-Join on GPUs through Warp Specialization" by Arpith C. Jacob
// et. al.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/LowerWGScope.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#ifndef NDEBUG
#include "llvm/IR/CFG.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#endif

using namespace llvm;

#define DEBUG_TYPE "lowerwgcode"

STATISTIC(LocalMemUsed, "amount of additional local memory used for sharing");

static constexpr char WG_SCOPE_MD[] = "work_group_scope";
static constexpr char WI_SCOPE_MD[] = "work_item_scope";
static constexpr char PFWI_MD[] = "parallel_for_work_item";

static cl::opt<int> Debug("sycl-lower-wg-debug", llvm::cl::Optional,
                          llvm::cl::Hidden,
                          llvm::cl::desc("Debug SYCL work group code lowering"),
                          llvm::cl::init(1));

namespace {
class SYCLLowerWGScopeLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  SYCLLowerWGScopeLegacyPass() : FunctionPass(ID) {
    initializeSYCLLowerWGScopeLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // run the LowerWGScope pass on the specified module
  bool runOnFunction(Function &F) override {
    FunctionAnalysisManager FAM;
    auto PA = Impl.run(F, FAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLLowerWGScopePass Impl;
};
} // namespace

char SYCLLowerWGScopeLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLLowerWGScopeLegacyPass, "LowerWGScope",
                "Lower Work Group Scope Code", false, false)

// Public interface to the SYCLLowerWGScopePass.
FunctionPass *llvm::createSYCLLowerWGScopePass() {
  return new SYCLLowerWGScopeLegacyPass();
}

template <typename T> static unsigned asUInt(T val) {
  return static_cast<unsigned>(val);
}

static IntegerType *getSizeTTy(Module &M) {
  LLVMContext &Ctx = M.getContext();
  auto PtrSize = M.getDataLayout().getPointerTypeSize(PointerType::getUnqual(Ctx));
  return PtrSize == 8 ? Type::getInt64Ty(Ctx) : Type::getInt32Ty(Ctx);
}

// Encapsulates SPIR-V-dependent code generation.
// TODO this should be factored out into a separate project in clang
namespace spirv {
// There is no TargetMachine for SPIR-V, so define those inline here
enum class AddrSpace : unsigned {
  Private = 0,
  Global = 1,
  Constant = 2,
  Local = 3,
  Generic = 4,
  Input = 5,
  Output = 6
};

enum class Scope : unsigned {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};

enum class MemorySemantics : unsigned {
  None = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10,
  UniformMemory = 0x40,
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200,
  AtomicCounterMemory = 0x400,
  ImageMemory = 0x800,
};

Instruction *genWGBarrier(Instruction &Before, const Triple &TT);
Value *genPseudoLocalID(Instruction &Before, const Triple &TT);
GlobalVariable *createWGLocalVariable(Module &M, Type *T, const Twine &Name);
} // namespace spirv

static bool isCallToAFuncMarkedWithMD(const Instruction *I, const char *MD) {
  const CallInst *Call = dyn_cast<CallInst>(I);
  const Function *F =
      dyn_cast_or_null<Function>(Call ? Call->getCalledFunction() : nullptr);
  return F && F->getMetadata(MD);
}

// Checks is this is a call to parallel_for_work_item.
static bool isPFWICall(const Instruction *I) {
  return isCallToAFuncMarkedWithMD(I, PFWI_MD);
}

// Checks if given instruction must be executed by all work items.
static bool isWIScopeInst(const Instruction *I) {
  if (I->isTerminator())
    return true;

  switch (I->getOpcode()) {
  case Instruction::Alloca: {
    llvm_unreachable("allocas must have been skipped");
    return true;
  }
  case Instruction::PHI:
    llvm_unreachable("PHIs must have been skipped");
    return true;
  case Instruction::Call:
    return isCallToAFuncMarkedWithMD(I, WI_SCOPE_MD);
  default:
    return false;
  }
}

// Checks if given instruction may have side effects visible outside current
// work item.
static bool mayHaveSideEffects(const Instruction *I) {
  if (I->isTerminator())
    return false;

  switch (I->getOpcode()) {
  case Instruction::Alloca:
    llvm_unreachable("allocas must have been handled");
    return false;
  case Instruction::PHI:
    llvm_unreachable("PHIs must have been skipped");
    return false;
  case Instruction::Call:
    assert(!isPFWICall(I) && "pfwi must have been handled separately");
    return true;
  case Instruction::AddrSpaceCast:
    return false;
  default:
    return true;
  }
}

// Generates control flow which disables execution of TrueBB in worker WIs:
//   IfBB:
//     ...
//     %a = load i64, i64 addrspace(1)* @__spirv_BuiltInLocalInvocationIndex
//     %b = icmp eq i64 %a, 0
//     br i1 %b, label %TrueBB, label %MergeBB
//
//   TrueBB:
//     ...
//     br label %MergeBB
//
//   MergeBB:
//     ...
// IfBB's terminator instruction is replaced with the branch.
//
static void guardBlockWithIsLeaderCheck(BasicBlock *IfBB, BasicBlock *TrueBB,
                                        BasicBlock *MergeBB,
                                        const DebugLoc &DbgLoc,
                                        const Triple &TT) {
  Value *LinearLocalID = spirv::genPseudoLocalID(*IfBB->getTerminator(), TT);
  auto *Ty = LinearLocalID->getType();
  Value *Zero = Constant::getNullValue(Ty);
  IRBuilder<> Builder(IfBB->getContext());
  spirv::genWGBarrier(*(IfBB->getTerminator()), TT);
  Builder.SetInsertPoint(IfBB->getTerminator());
  Value *Cmp = Builder.CreateICmpEQ(LinearLocalID, Zero, "cmpz");
  Builder.SetCurrentDebugLocation(DbgLoc);
  Builder.CreateCondBr(Cmp, TrueBB, MergeBB);
  IfBB->getTerminator()->eraseFromParent();
  assert(TrueBB->getSingleSuccessor() == MergeBB && "CFG tform error");
}

static void
shareOutputViaLocalMem(Instruction &I, BasicBlock &BBa, BasicBlock &BBb,
                       SmallPtrSetImpl<Instruction *> &LeaderScope) {

  SmallPtrSet<Instruction *, 4> Users;

  for (auto User : I.users()) {
    Instruction *UI = dyn_cast<Instruction>(User);
    if (!UI || LeaderScope.find(UI) != LeaderScope.end())
      // not interested in the user if it is within the scope
      continue;
    Users.insert(UI);
  }
  // Skip instruction w/o uses or if all its uses lie within the scope
  if (Users.size() == 0)
    return;
  LLVMContext &Ctx = I.getContext();
  Type *T = I.getType();
  // 1) Create WG local variable
  Value *WGLocal = spirv::createWGLocalVariable(*I.getModule(), T,
                                                I.getFunction()->getName() +
                                                    "WG_" + Twine(I.getName()));
  // 2) Generate a store of the produced value into the WG local var
  IRBuilder<> Bld(Ctx);
  Bld.SetInsertPoint(I.getNextNode());
  Bld.CreateStore(&I, WGLocal);
  // 3) Generate a load in the "worker" BB of the value stored by the leader
  Bld.SetInsertPoint(&BBb.front());
  auto *WGVal = Bld.CreateLoad(T, WGLocal, "wg_val_" + Twine(I.getName()));
  // 4) Finally, replace usages of I outside the scope
  for (auto *U : Users)
    U->replaceUsesOfWith(&I, WGVal);
}

using InstrRange = std::pair<Instruction *, Instruction *>;

// Input IR, where I1..IN is the range. I1 has uses outside the range:
//   A
//   %I1 = ...;
//   ... USE1(%I1) ...
//   %IN = ...;
//   B
//   ... USE2(%I1) ...
//
// Resulting basic blocks:
// BBa:
//   A
//   %linear_id = call get_linear_local_id()
//   %is_leader = cmp %linear_id, 0
//   branch %is_leader LeaderBB, BB
//
// LeaderBB:
//   %I1 = ...;
//   store %I1, @WG_I1
//   ... USE1(%I1) ...
//   %IN = ...;
//   store %IN, @WG_I1
//   branch BBb
//
// BBb:
//   call WG_control_barrier()
//   %I1_new = load @WG_I1
//   ...
//   B
//   ... USE2(%I1_new) ...
static void tformRange(const InstrRange &R, const Triple &TT) {
  // Instructions seen between the first and the last
  SmallPtrSet<Instruction *, 16> Seen;
  Instruction *FirstSE = R.first;
  Instruction *LastSE = R.second;
  LLVM_DEBUG(llvm::dbgs() << "Tform range {\n  " << *FirstSE << "\n  "
                          << *LastSE << "\n}\n");
  assert(FirstSE->getParent() == LastSE->getParent() && "invalid range");

  for (auto *I = FirstSE; I != LastSE; I = I->getNextNode())
    Seen.insert(I);
  Seen.insert(LastSE);

  BasicBlock *BBa = FirstSE->getParent();
  BasicBlock *LeaderBB = BBa->splitBasicBlock(FirstSE, "wg_leader");
  BasicBlock *BBb = LeaderBB->splitBasicBlock(LastSE->getNextNode(), "wg_cf");

  // 1) insert the first "is work group leader" test (at the first split) for
  //     the worker WIs to detour the side effects instructions
  guardBlockWithIsLeaderCheck(BBa, LeaderBB, BBb, FirstSE->getDebugLoc(), TT);

  // 2) "Share" the output values of the instructions in the range
  for (auto *I : Seen)
    shareOutputViaLocalMem(*I, *BBa, *BBb, Seen);

  // 3) Insert work group barrier so that workers further read valid data
  //    (before the materialization reads inserted at step 2)
  spirv::genWGBarrier(BBb->front(), TT);
}

namespace {
using LocalsSet = SmallPtrSet<AllocaInst *, 4>;
}

static void copyBetweenPrivateAndShadow(Value *L, GlobalVariable *Shadow,
                                        IRBuilder<> &Builder, bool Loc2Shadow) {
  assert(isa<PointerType>(L->getType()));
  Type *T = nullptr;
  MaybeAlign LocAlign(0);

  if (const auto *AI = dyn_cast<AllocaInst>(L)) {
    T = AI->getAllocatedType();
    LocAlign = AI->getAlign();
  } else {
    auto Arg = cast<Argument>(L);
    T = Arg->getParamByValType();
    LocAlign = Arg->getParamAlign();
  }

  assert(T && "Unexpected type");

  if (T->isAggregateType()) {
    // TODO: we should use methods which directly return MaybeAlign once such
    // are added to LLVM for AllocaInst and GlobalVariable
    auto ShdAlign = MaybeAlign(Shadow->getAlignment());
    Module &M = *Shadow->getParent();
    auto SizeVal = M.getDataLayout().getTypeStoreSize(T);
    auto Size = ConstantInt::get(getSizeTTy(M), SizeVal);
    if (Loc2Shadow)
      Builder.CreateMemCpy(Shadow, ShdAlign, L, LocAlign, Size);
    else
      Builder.CreateMemCpy(L, LocAlign, Shadow, ShdAlign, Size);
  } else {
    Value *Src = L;
    Value *Dst = Shadow;

    if (!Loc2Shadow)
      std::swap(Src, Dst);
    Value *LocalVal = Builder.CreateLoad(T, Src, "mat_ld");
    Builder.CreateStore(LocalVal, Dst);
  }
}

// Performs the following transformation for each basic block in the input map:
//
// BB:
//   some_instructions
// =>
// TestBB:
//   %linear_id = call get_linear_local_id()
//   %is_leader = cmp %linear_id, 0
//   branch %is_leader LeaderBB, OriginalBB
//
// LeaderBB:
//   *@Shadow_local1 = *local1
//   ...
//   *@Shadow_localN = *localN
//   branch BB
//
// BB:
//   call WG_control_barrier()
//   *local1 = *@Shadow_local1
//   ...
//   *localN = *@Shadow_localN
//   some_instructions
//
// Where:
// - local<i> is the set of to-be-materialized locals for OriginalBB taken from
//   the first input map
// - @Shadow_local<i> is the shadow workgroup-shared global variable for
// local<i>,
//   taken from the second input map
//
static void materializeLocalsInWIScopeBlocksImpl(
    const DenseMap<BasicBlock *, std::unique_ptr<LocalsSet>> &BB2MatLocals,
    const DenseMap<AllocaInst *, GlobalVariable *> &Local2Shadow,
    const Triple &TT) {
  for (auto &P : BB2MatLocals) {
    // generate LeaderBB and private<->shadow copies in proper BBs
    BasicBlock *LeaderBB = P.first;
    BasicBlock *BB = LeaderBB->splitBasicBlock(&LeaderBB->front(), "LeaderMat");
    // Add a barrier to the original block:
    Instruction *At =
        spirv::genWGBarrier(*BB->getFirstNonPHI(), TT)->getNextNode();

    for (AllocaInst *L : *P.second.get()) {
      auto MapEntry = Local2Shadow.find(L);
      assert(MapEntry != Local2Shadow.end() && "local must have a shadow");
      auto *Shadow = MapEntry->second;
      LLVMContext &Ctx = L->getContext();
      IRBuilder<> Builder(Ctx);
      // fill the leader BB:
      // fetch data from leader's private copy (which is always up to date) into
      // the corresponding shadow variable
      Builder.SetInsertPoint(&LeaderBB->front());
      copyBetweenPrivateAndShadow(L, Shadow, Builder, true /*private->shadow*/);
      // store data to the local variable - effectively "refresh" the value of
      // the local in each work item in the work group
      Builder.SetInsertPoint(At);
      copyBetweenPrivateAndShadow(L, Shadow, Builder,
                                  false /*shadow->private*/);
    }
    // now generate the TestBB and the leader WI guard
    BasicBlock *TestBB =
        LeaderBB->splitBasicBlock(&LeaderBB->front(), "TestMat");
    std::swap(TestBB, LeaderBB);
    guardBlockWithIsLeaderCheck(TestBB, LeaderBB, BB, At->getDebugLoc(), TT);
  }
}

// Checks if there is a need to materialize value of given local in given work
// item-scope basic block.
static bool localMustBeMaterialized(const AllocaInst *L, const BasicBlock &BB) {
  // TODO this is overly conservative - see speculations below.
  return true;
}

// This function handles locals of kind 3 (see comments at the top of file).
//
// For each alloca the following transformation is done for each WI scope basic
// block basic_block10 where the alloca is used:
//
//   T *p = alloca(T);
//   if (is_leader) { use1(p); } // WG scope
//   ...
// basic_block10: // WI scope basic block (executed by all WIs)
//   use2(p);
// =>
//   T *p = alloca(T);
//   if (is_leader) { use1(p); }
//   ...
// // p materialization code; note that all locals in the WG scope are uniform
//   if (is_leader) { *@Shadow_p = *p; } // store actual value of the local
//   barrier(); // make sure workers wait till the value write above is complete
//   branch basic_block10;
// basic_block10: // WI scope basic block
//   *p = *@Shadow_p; // materialize the value in the local variable before use;
//                 // maybe skipped for the leader, but does not seem worth it
//                 // as the leader WI is just a vector lane, so there should be
//                 // one load per thread (subgroup) anyway.
//   use2(p);
//
// NOTE:
// Simply redirecting all the p uses (dereferences) in WI scope blocks is not
// enough in the general case. E.g. consider this example:
//
//   T *p = alloca(T);
//   T *p1 = p+10;
//   if (is_leader) { use1(p); } // WG scope
//   ...
// basic_block10: // WI scope
//   use2(p1);
//
// TODO. This implementation is quite ineffective. Currently it materializes
// all locals in all WI scope basic blocks.
// Will be improved incrementally:
// - For each alloca: determine all derived (via GEPs) pointers and make sure
//   they don't escape. Then check if there are reads in current WI scope BB
//   through either of those. If none of them escape and there are no reads then
//   materialization of this alloca in this BB is not needed.
// - Materialization is not needed if there is dominating BB with materialized
//   value, and there are no WG scope writes to this alloca on any path from
//   that BB to current.
// - Avoid unnecessary '*p = *@Shadow_p' reloads and redirect p uses them to the
//   @Shadow_p in case it can be proved it is safe (see note above). Might not
//   have any noticeable effect, though, as reading from Shadow always goes to a
//   register file anyway.
//
void materializeLocalsInWIScopeBlocks(SmallPtrSetImpl<AllocaInst *> &Locals,
                                      SmallPtrSetImpl<BasicBlock *> &WIScopeBBs,
                                      const Triple &TT) {
  // maps local variable to its "shadow" workgroup-shared global:
  DenseMap<AllocaInst *, GlobalVariable *> Local2Shadow;
  // records which locals must be materialized at the beginning of a block:
  DenseMap<BasicBlock *, std::unique_ptr<LocalsSet>> BB2MatLocals;

  // TODO: iterating over BBs first then over locals would require less
  // book-keeping with current implementation, but later improvements will need
  // global info like mapping BBs to locals sets to optimize.

  // Fill the local-to-shadow and basic block-to-locals maps:
  for (auto L : Locals) {
    for (auto *BB : WIScopeBBs) {
      if (!localMustBeMaterialized(L, *BB))
        continue;
      if (Local2Shadow.find(L) == Local2Shadow.end()) {
        // lazily create a "shadow" for current local:
        GlobalVariable *Shadow = spirv::createWGLocalVariable(
            *BB->getModule(), L->getAllocatedType(), "WGCopy");
        Local2Shadow.insert(std::make_pair(L, Shadow));
      }
      auto &MatLocals = BB2MatLocals[BB];

      if (!MatLocals.get()) {
        // lazily create a locals set for current BB:
        MatLocals.reset(new LocalsSet());
      }
      MatLocals->insert(L);
    }
  }
  // perform the materialization
  materializeLocalsInWIScopeBlocksImpl(BB2MatLocals, Local2Shadow, TT);
}

#ifndef NDEBUG
static void dumpDot(const Function &F, const Twine &Suff) {
  std::error_code EC;
  auto FName =
      ("PFWG_Kernel_" + Suff + "_" + Twine(F.getValueID()) + ".dot").str();
  raw_fd_ostream File(FName, EC, sys::fs::OF_Text);

  if (!EC)
    WriteGraph(File, (const Function *)&F, false);
  else
    errs() << "  error opening file for writing: << " << FName << "\n";
}

static void dumpIR(const Function &F, const Twine &Suff) {
  std::error_code EC;
  auto FName =
      ("PFWG_Kernel_" + Suff + "_" + Twine(F.getValueID()) + ".ll").str();
  raw_fd_ostream File(FName, EC, sys::fs::OF_Text);

  if (!EC)
    F.print(File, 0, 1, 1);
  else
    errs() << "  error opening file for writing: << " << FName << "\n";
}
#endif // NDEBUG

using CaptureDesc = std::pair<AllocaInst *, GetElementPtrInst *>;

// This function handles locals of kind 1 (see comments at the top of file) -
// captures of private_memory<T> variables. It basically adds (*) instruction in
// the pattern below.
//     if (is_leader())
//       *PFWI_lambda_obj_shadow_addr = *PFWI_lambda_obj_alloca;
//     barrier();
//     *PFWI_lambda_obj_alloca = *PFWI_lambda_obj_shadow_addr;
// (*) PFWI_lambda_obj_alloca->priv_var_addr = priv_var_alloca;
//     parallel_for_work_item(..., PFWI_lambda_obj_alloca);
//
static void fixupPrivateMemoryPFWILambdaCaptures(CallInst *PFWICall) {
  // Lambda object is always the last argument to the PFWI lambda function:
  auto NArgs = PFWICall->arg_size();
  if (PFWICall->arg_size() == 1)
    return;

  Value *LambdaObj =
      PFWICall->getArgOperand(NArgs - 1 /*lambda object parameter*/);
  // First go through all stores through the LambdaObj pointer - those are
  // initialization of captures, and for each stored value find its origin -
  // whether it is an alloca with "work_item_scope"
  SmallVector<CaptureDesc, 4> PrivMemCaptures;

  // Look through cast
  if (auto *Cast = dyn_cast<AddrSpaceCastInst>(LambdaObj))
    LambdaObj = Cast->getOperand(0);

  for (auto *U : LambdaObj->users()) {
    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U);

    if (!GEP)
      continue;
    assert(GEP->hasOneUse());
    StoreInst *CaptureInit = dyn_cast<StoreInst>(*(GEP->users().begin()));

    if (!CaptureInit)
      // this can't be private_memory<T> capture, which is always captured by
      // address via the StoreInst instruction
      continue;
    Value *StoredVal = CaptureInit->getValueOperand();
    // '[=]' capture of a private_memory<T> instance is not permitted, there is
    // no 'const T &operator()(const h_item<Dimensions> &id);', so compiler
    // would generate an error; this means captured value is always a pointer -
    // whether it is private_memory instance or some other type
    if (!StoredVal->getType()->isPointerTy())
      continue;

    while (StoredVal && !isa<AllocaInst>(StoredVal)) {
      if (auto *BC = dyn_cast<BitCastInst>(StoredVal)) {
        StoredVal = BC->getOperand(0);
        continue;
      }
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(StoredVal)) {
        StoredVal = ASC->getOperand(0);
        continue;
      }
      StoredVal = nullptr; // something else is captured
      break;
    }
    auto *AI = dyn_cast_or_null<AllocaInst>(StoredVal);

    // only private_memory allocations (allocas marked with "work_item_scope"
    // are of interest here:
    if (!AI || !AI->getMetadata(WI_SCOPE_MD))
      continue;
    PrivMemCaptures.push_back(CaptureDesc{AI, GEP});
  }
  // now rewrite the captured address of a private_memory variables within the
  // PFWI lambda object:
  for (auto &C : PrivMemCaptures) {
    GetElementPtrInst *NewGEP = cast<GetElementPtrInst>(C.second->clone());
    NewGEP->insertBefore(PFWICall);
    IRBuilder<> Bld(PFWICall->getContext());
    Bld.SetInsertPoint(PFWICall);
    Value *Val = C.first;
    auto ValAS = cast<PointerType>(Val->getType())->getAddressSpace();
    auto PtrAS =
        cast<PointerType>(NewGEP->getResultElementType())->getAddressSpace();

    if (ValAS != PtrAS)
      Val = Bld.CreateAddrSpaceCast(Val, NewGEP->getResultElementType());
    Bld.CreateStore(Val, NewGEP);
  }
}

// Go through "byval" parameters which are passed as AS(0) pointers
// and: (1) create local shadows for them (2) and initialize them from the
// leader's copy and (3) materialize the value in the local variable before use
static void shareByValParams(Function &F, const Triple &TT) {
  // Skip alloca instructions and split. Alloca instructions must be in the
  // beginning of the function otherwise they are considered as dynamic which
  // can cause the problems with inlining.
  BasicBlock *EntryBB = &F.getEntryBlock();
  Instruction *SplitPoint = &*EntryBB->begin();
  for (; SplitPoint->getOpcode() == Instruction::Alloca;
       SplitPoint = SplitPoint->getNextNode())
    ;
  BasicBlock *LeaderBB = EntryBB->splitBasicBlock(SplitPoint, "leader");
  BasicBlock *MergeBB = LeaderBB->splitBasicBlock(&LeaderBB->front(), "merge");

  // Rewire the above basic blocks so that LeaderBB is executed only for the
  // leader workitem
  guardBlockWithIsLeaderCheck(EntryBB, LeaderBB, MergeBB,
                              EntryBB->back().getDebugLoc(), TT);
  Instruction &At = LeaderBB->back();

  for (auto &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;

    assert(Arg.getType()->getPointerAddressSpace() ==
           asUInt(spirv::AddrSpace::Private));

    // Create the shared copy - "shadow" - for current arg
    Type *T = Arg.getParamByValType();
    GlobalVariable *Shadow =
        spirv::createWGLocalVariable(*F.getParent(), T, "ArgShadow");

    LLVMContext &Ctx = At.getContext();
    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(&LeaderBB->front());

    copyBetweenPrivateAndShadow(&Arg, Shadow, Builder,
                                true /*private->shadow*/);
    // Materialize the value in the local variable before use
    Builder.SetInsertPoint(&MergeBB->front());
    copyBetweenPrivateAndShadow(&Arg, Shadow, Builder,
                                false /*shadow->private*/);
  }
  // Insert barrier to make sure workers use up-to-date shared values written by
  // the leader
  spirv::genWGBarrier(MergeBB->front(), TT);
}

PreservedAnalyses SYCLLowerWGScopePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  if (!F.getMetadata(WG_SCOPE_MD))
    return PreservedAnalyses::all();
  LLVM_DEBUG(llvm::dbgs() << "Function name: " << F.getName() << "\n");
  const auto &TT = llvm::Triple(F.getParent()->getTargetTriple());
  // Ranges of "side effect" instructions
  SmallVector<InstrRange, 16> Ranges;
  SmallPtrSet<AllocaInst *, 16> Allocas;
  SmallPtrSet<Instruction *, 16> WIScopeInsts;
  SmallPtrSet<CallInst *, 4> PFWICalls;

  // Collect the ranges which need transformation
  for (auto &BB : F) {
    // first and last instructions with side effects, which must be executed
    // only once per work group:
    Instruction *First = nullptr;
    Instruction *Last = nullptr;

    // Skip PHIs, allocas and addrspacecasts associated with allocas, as they
    // don't have side effects and must never be guarded with the WG leader
    // test. Note that there should be no allocas in local address space at this
    // point - they must have been converted to globals.
    Instruction *I = BB.getFirstNonPHI();

    for (; I->getOpcode() == Instruction::Alloca ||
           I->getOpcode() == Instruction::AddrSpaceCast ||
           I->isDebugOrPseudoInst();
         I = I->getNextNode()) {
      auto *AllocaI = dyn_cast<AllocaInst>(I);
      // Allocas marked with "work_item_scope" are those originating from
      // sycl::private_memory<T> variables, which must be in private memory.
      // No shadows/materialization is needed for them because they can be
      // updated only within PFWIs
      if (AllocaI && !AllocaI->getMetadata(WI_SCOPE_MD))
        Allocas.insert(AllocaI);
    }
    for (; I && (I != BB.getTerminator()); I = I->getNextNode()) {
      if (isWIScopeInst(I)) {
        if (isPFWICall(I))
          PFWICalls.insert(dyn_cast<CallInst>(I));
        WIScopeInsts.insert(I);
        LLVM_DEBUG(llvm::dbgs() << "+++ Exec by all: " << *I << "\n");
        // need to split the range here, because the instruction must be
        // executed by all work items - force range addition
        if (First) {
          assert(Last && "range must have been closed 1");
          Ranges.push_back(InstrRange{First, Last});
          First = nullptr;
          Last = nullptr;
        }
        continue;
      }
      if (!mayHaveSideEffects(I))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "+++ Side effects: " << *I << "\n");
      if (!First)
        First = I;
      Last = I;
    }
    if (First) {
      assert(Last && "range must have been closed 2");
      Ranges.push_back(InstrRange{First, Last});
    }
  }

  int NByval = 0;
  for (const auto &Arg : F.args()) {
    if (Arg.hasByValAttr())
      NByval++;
  }

  bool HaveChanges = (Ranges.size() > 0) || (Allocas.size() > 0) || NByval > 0;

#ifndef NDEBUG
  if (HaveChanges && Debug > 1) {
    dumpIR(F, "before");
    dumpDot(F, "before");
  }
#endif // NDEBUG

  // Perform the transformation
  for (auto &R : Ranges)
    tformRange(R, TT);

  // There can be allocas not corresponding to any variable declared in user
  // code but generated by the compiler - e.g. for non-trivially typed
  // parameters passed by value. There can be WG scope stores into such
  // allocas, which need to be made visible to all WIs. This is done via
  // creating a "shadow" workgroup-shared variable and using it to propagate
  // the value of the alloca'ed variable to worker WIs from the leader.

  // First collect WIScope BBs where locals will be materialized:
  SmallPtrSet<BasicBlock *, 16> WIScopeBBs;

  for (auto *I : WIScopeInsts)
    WIScopeBBs.insert(I->getParent());

  // Now materialize the locals:
  materializeLocalsInWIScopeBlocks(Allocas, WIScopeBBs, TT);

  // Fixup captured addresses of private_memory instances in current WI
  for (auto *PFWICall : PFWICalls)
    fixupPrivateMemoryPFWILambdaCaptures(PFWICall);

  // Finally, create shadows for and replace usages of byval pointer params.
  shareByValParams(F, TT);

#ifndef NDEBUG
  if (HaveChanges && Debug > 0)
    verifyModule(*F.getParent(), &llvm::errs());
  if (HaveChanges && Debug > 1) {
    dumpIR(F, "after");
    dumpDot(F, "after");
  }
#endif // NDEBUG
  return HaveChanges ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

GlobalVariable *spirv::createWGLocalVariable(Module &M, Type *T,
                                             const Twine &Name) {
  GlobalVariable *G =
      new GlobalVariable(M,                              // module
                         T,                              // type
                         false,                          // isConstant
                         GlobalValue::InternalLinkage,   // Linkage
                         UndefValue::get(T),             // Initializer
                         Name,                           // Name
                         nullptr,                        // InsertBefore
                         GlobalVariable::NotThreadLocal, // ThreadLocalMode
                         asUInt(spirv::AddrSpace::Local) // AddressSpace
      );
  G->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  const DataLayout &DL = M.getDataLayout();
  G->setAlignment(MaybeAlign(DL.getPreferredAlign(G)));
  LocalMemUsed += DL.getTypeStoreSize(G->getValueType());
  LLVM_DEBUG(llvm::dbgs() << "Local AS Var created: " << G->getName() << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Local mem used: " << LocalMemUsed << "B\n");
  return G;
}

// Functions below expose SPIR-V translator-specific intrinsics to the use
// in LLVM IR. Those calls and global references will be translated to
// corresponding SPIR-V operations and builtin variables.
//
// TODO generalize to support all SPIR-V intrinsic operations and builtin
//      variables

// Return a value equals to 0 if and only if the local linear id is 0.
Value *spirv::genPseudoLocalID(Instruction &Before, const Triple &TT) {
  Module &M = *Before.getModule();
  if (TT.isNVPTX() || TT.isAMDGCN()) {
    LLVMContext &Ctx = Before.getContext();
    Type *RetTy = getSizeTTy(M);

    IRBuilder<> Bld(Ctx);
    Bld.SetInsertPoint(&Before);

#define CREATE_CALLEE(NAME, FN_NAME)                                           \
  FunctionCallee FnCallee##NAME = M.getOrInsertFunction(FN_NAME, RetTy);       \
  assert(FnCallee##NAME && "spirv intrinsic creation failed");                 \
  auto NAME = Bld.CreateCall(FnCallee##NAME, {});

    CREATE_CALLEE(LocalInvocationId_X, "_Z27__spirv_LocalInvocationId_xv");
    CREATE_CALLEE(LocalInvocationId_Y, "_Z27__spirv_LocalInvocationId_yv");
    CREATE_CALLEE(LocalInvocationId_Z, "_Z27__spirv_LocalInvocationId_zv");

#undef CREATE_CALLEE

    // 1: returns
    //   __spirv_LocalInvocationId_x() |
    //   __spirv_LocalInvocationId_y() |
    //   __spirv_LocalInvocationId_z()
    //
    return Bld.CreateOr(LocalInvocationId_X,
                        Bld.CreateOr(LocalInvocationId_Y, LocalInvocationId_Z));
  } else {
    // extern "C" const __constant size_t __spirv_BuiltInLocalInvocationIndex;
    // Must correspond to the code in
    // llvm-spirv/lib/SPIRV/OCL20ToSPIRV.cpp
    // OCL20ToSPIRV::transWorkItemBuiltinsToVariables()
    StringRef Name = "__spirv_BuiltInLocalInvocationIndex";
    GlobalVariable *G = M.getGlobalVariable(Name);

    if (!G) {
      Type *T = getSizeTTy(M);
      G = new GlobalVariable(M,                              // module
                             T,                              // type
                             true,                           // isConstant
                             GlobalValue::ExternalLinkage,   // Linkage
                             nullptr,                        // Initializer
                             Name,                           // Name
                             nullptr,                        // InsertBefore
                             GlobalVariable::NotThreadLocal, // ThreadLocalMode
                             // TODO 'Input' crashes CPU Back-End
                             // asUInt(spirv::AddrSpace::Input) // AddressSpace
                             asUInt(spirv::AddrSpace::Global) // AddressSpace
      );
      Align Alignment = M.getDataLayout().getPreferredAlign(G);
      G->setAlignment(MaybeAlign(Alignment));
    }
    Value *Res = new LoadInst(G->getValueType(), G, "", &Before);
    return Res;
  }
}

// extern void __spirv_ControlBarrier(Scope Execution, Scope Memory,
//  uint32_t Semantics) noexcept;
Instruction *spirv::genWGBarrier(Instruction &Before, const Triple &TT) {
  Module &M = *Before.getModule();
  StringRef Name = "_Z22__spirv_ControlBarrierjjj";
  LLVMContext &Ctx = Before.getContext();
  Type *ScopeTy = Type::getInt32Ty(Ctx);
  Type *SemanticsTy = Type::getInt32Ty(Ctx);
  Type *RetTy = Type::getVoidTy(Ctx);

  AttributeList Attr;
  Attr = Attr.addFnAttribute(Ctx, Attribute::Convergent);
  FunctionCallee FC =
      M.getOrInsertFunction(Name, Attr, RetTy, ScopeTy, ScopeTy, SemanticsTy);
  assert(FC.getCallee() && "spirv intrinsic creation failed");

  IRBuilder<> Bld(Ctx);
  Bld.SetInsertPoint(&Before);
  auto ArgExec = ConstantInt::get(ScopeTy, asUInt(spirv::Scope::Workgroup));
  auto ArgMem = ConstantInt::get(ScopeTy, asUInt(spirv::Scope::Workgroup));
  auto ArgSema = ConstantInt::get(
      ScopeTy, asUInt(spirv::MemorySemantics::SequentiallyConsistent) |
                   asUInt(spirv::MemorySemantics::WorkgroupMemory));
  auto BarrierCall = Bld.CreateCall(FC, {ArgExec, ArgMem, ArgSema});
  BarrierCall->addFnAttr(llvm::Attribute::Convergent);
  return BarrierCall;
}
