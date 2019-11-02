//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This pass implements a data flow analysis that propagates debug location
/// information by inserting additional DBG_VALUE insts into the machine
/// instruction stream. Before running, each DBG_VALUE inst corresponds to a
/// source assignment of a variable. Afterwards, a DBG_VALUE inst specifies a
/// variable location for the current basic block (see SourceLevelDebugging.rst).
///
/// This is a separate pass from DbgValueHistoryCalculator to facilitate
/// testing and improve modularity.
///
/// Each variable location is represented by a VarLoc object that identifies the
/// source variable, its current machine-location, and the DBG_VALUE inst that
/// specifies the location. Each VarLoc is indexed in the (function-scope)
/// VarLocMap, giving each VarLoc a unique index. Rather than operate directly
/// on machine locations, the dataflow analysis in this pass identifies
/// locations by their index in the VarLocMap, meaning all the variable
/// locations in a block can be described by a sparse vector of VarLocMap
/// indexes.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");
STATISTIC(NumRemoved, "Number of DBG_VALUE instructions removed");

// If @MI is a DBG_VALUE with debug value described by a defined
// register, returns the number of this register. In the other case, returns 0.
static Register isDbgValueDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue() && "expected a DBG_VALUE");
  assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
  // If location of variable is described using a register (directly
  // or indirectly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : Register();
}

namespace {

class LiveDebugValues : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;

  enum struct TransferKind { TransferCopy, TransferSpill, TransferRestore };

  /// Keeps track of lexical scopes associated with a user value's source
  /// location.
  class UserValueScopes {
    DebugLoc DL;
    LexicalScopes &LS;
    SmallPtrSet<const MachineBasicBlock *, 4> LBlocks;

  public:
    UserValueScopes(DebugLoc D, LexicalScopes &L) : DL(std::move(D)), LS(L) {}

    /// Return true if current scope dominates at least one machine
    /// instruction in a given machine basic block.
    bool dominates(MachineBasicBlock *MBB) {
      if (LBlocks.empty())
        LS.getMachineBasicBlocks(DL, LBlocks);
      return LBlocks.count(MBB) != 0 || LS.dominates(DL, MBB);
    }
  };

  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  /// Storage for identifying a potentially inlined instance of a variable,
  /// or a fragment thereof.
  class DebugVariable {
    const DILocalVariable *Variable;
    OptFragmentInfo Fragment;
    const DILocation *InlinedAt;

    /// Fragment that will overlap all other fragments. Used as default when
    /// caller demands a fragment.
    static const FragmentInfo DefaultFragment;

  public:
    DebugVariable(const DILocalVariable *Var, OptFragmentInfo &&FragmentInfo,
                  const DILocation *InlinedAt)
        : Variable(Var), Fragment(FragmentInfo), InlinedAt(InlinedAt) {}

    DebugVariable(const DILocalVariable *Var, OptFragmentInfo &FragmentInfo,
                  const DILocation *InlinedAt)
        : Variable(Var), Fragment(FragmentInfo), InlinedAt(InlinedAt) {}

    DebugVariable(const DILocalVariable *Var, const DIExpression *DIExpr,
                  const DILocation *InlinedAt)
        : DebugVariable(Var, DIExpr->getFragmentInfo(), InlinedAt) {}

    DebugVariable(const MachineInstr &MI)
        : DebugVariable(MI.getDebugVariable(),
                        MI.getDebugExpression()->getFragmentInfo(),
                        MI.getDebugLoc()->getInlinedAt()) {}

    const DILocalVariable *getVar() const { return Variable; }
    const OptFragmentInfo &getFragment() const { return Fragment; }
    const DILocation *getInlinedAt() const { return InlinedAt; }

    const FragmentInfo getFragmentDefault() const {
      return Fragment.getValueOr(DefaultFragment);
    }

    static bool isFragmentDefault(FragmentInfo &F) {
      return F == DefaultFragment;
    }

    bool operator==(const DebugVariable &Other) const {
      return std::tie(Variable, Fragment, InlinedAt) ==
             std::tie(Other.Variable, Other.Fragment, Other.InlinedAt);
    }

    bool operator<(const DebugVariable &Other) const {
      return std::tie(Variable, Fragment, InlinedAt) <
             std::tie(Other.Variable, Other.Fragment, Other.InlinedAt);
    }
  };

  friend struct llvm::DenseMapInfo<DebugVariable>;

  /// A pair of debug variable and value location.
  struct VarLoc {
    // The location at which a spilled variable resides. It consists of a
    // register and an offset.
    struct SpillLoc {
      unsigned SpillBase;
      int SpillOffset;
      bool operator==(const SpillLoc &Other) const {
        return SpillBase == Other.SpillBase && SpillOffset == Other.SpillOffset;
      }
    };

    /// Identity of the variable at this location.
    const DebugVariable Var;

    /// The expression applied to this location.
    const DIExpression *Expr;

    /// DBG_VALUE to clone var/expr information from if this location
    /// is moved.
    const MachineInstr &MI;

    mutable UserValueScopes UVS;
    enum VarLocKind {
      InvalidKind = 0,
      RegisterKind,
      SpillLocKind,
      ImmediateKind,
      EntryValueKind
    } Kind = InvalidKind;

    /// The value location. Stored separately to avoid repeatedly
    /// extracting it from MI.
    union {
      uint64_t RegNo;
      SpillLoc SpillLocation;
      uint64_t Hash;
      int64_t Immediate;
      const ConstantFP *FPImm;
      const ConstantInt *CImm;
    } Loc;

    VarLoc(const MachineInstr &MI, LexicalScopes &LS)
        : Var(MI), Expr(MI.getDebugExpression()), MI(MI),
          UVS(MI.getDebugLoc(), LS) {
      static_assert((sizeof(Loc) == sizeof(uint64_t)),
                    "hash does not cover all members of Loc");
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      if (int RegNo = isDbgValueDescribedByReg(MI)) {
        Kind = MI.isDebugEntryValue() ? EntryValueKind : RegisterKind;
        Loc.RegNo = RegNo;
      } else if (MI.getOperand(0).isImm()) {
        Kind = ImmediateKind;
        Loc.Immediate = MI.getOperand(0).getImm();
      } else if (MI.getOperand(0).isFPImm()) {
        Kind = ImmediateKind;
        Loc.FPImm = MI.getOperand(0).getFPImm();
      } else if (MI.getOperand(0).isCImm()) {
        Kind = ImmediateKind;
        Loc.CImm = MI.getOperand(0).getCImm();
      }
      assert((Kind != ImmediateKind || !MI.isDebugEntryValue()) &&
             "entry values must be register locations");
    }

    /// Take the variable and machine-location in DBG_VALUE MI, and build an
    /// entry location using the given expression.
    static VarLoc CreateEntryLoc(const MachineInstr &MI, LexicalScopes &LS,
                                 const DIExpression *EntryExpr) {
      VarLoc VL(MI, LS);
      VL.Kind = EntryValueKind;
      VL.Expr = EntryExpr;
      return VL;
    }

    /// Copy the register location in DBG_VALUE MI, updating the register to
    /// be NewReg.
    static VarLoc CreateCopyLoc(const MachineInstr &MI, LexicalScopes &LS,
                                unsigned NewReg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Loc.RegNo = NewReg;
      return VL;
    }

    /// Take the variable described by DBG_VALUE MI, and create a VarLoc
    /// locating it in the specified spill location.
    static VarLoc CreateSpillLoc(const MachineInstr &MI, unsigned SpillBase,
                                 int SpillOffset, LexicalScopes &LS) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = SpillLocKind;
      VL.Loc.SpillLocation = {SpillBase, SpillOffset};
      return VL;
    }

    /// Create a DBG_VALUE representing this VarLoc in the given function.
    /// Copies variable-specific information such as DILocalVariable and
    /// inlining information from the original DBG_VALUE instruction, which may
    /// have been several transfers ago.
    MachineInstr *BuildDbgValue(MachineFunction &MF) const {
      const DebugLoc &DbgLoc = MI.getDebugLoc();
      bool Indirect = MI.isIndirectDebugValue();
      const auto &IID = MI.getDesc();
      const DILocalVariable *Var = MI.getDebugVariable();
      const DIExpression *DIExpr = MI.getDebugExpression();

      switch (Kind) {
      case EntryValueKind:
        // An entry value is a register location -- but with an updated
        // expression.
        return BuildMI(MF, DbgLoc, IID, Indirect, Loc.RegNo, Var, Expr);
      case RegisterKind:
        // Register locations are like the source DBG_VALUE, but with the
        // register number from this VarLoc.
        return BuildMI(MF, DbgLoc, IID, Indirect, Loc.RegNo, Var, DIExpr);
      case SpillLocKind: {
        // Spills are indirect DBG_VALUEs, with a base register and offset.
        // Use the original DBG_VALUEs expression to build the spilt location
        // on top of. FIXME: spill locations created before this pass runs
        // are not recognized, and not handled here.
        auto *SpillExpr = DIExpression::prepend(
            DIExpr, DIExpression::ApplyOffset, Loc.SpillLocation.SpillOffset);
        unsigned Base = Loc.SpillLocation.SpillBase;
        return BuildMI(MF, DbgLoc, IID, true, Base, Var, SpillExpr);
      }
      case ImmediateKind: {
        MachineOperand MO = MI.getOperand(0);
        return BuildMI(MF, DbgLoc, IID, Indirect, MO, Var, DIExpr);
      }
      case InvalidKind:
        llvm_unreachable("Tried to produce DBG_VALUE for invalid VarLoc");
      }
      llvm_unreachable("Unrecognized LiveDebugValues.VarLoc.Kind enum");
    }

    /// Is the Loc field a constant or constant object?
    bool isConstant() const { return Kind == ImmediateKind; }

    /// If this variable is described by a register, return it,
    /// otherwise return 0.
    unsigned isDescribedByReg() const {
      if (Kind == RegisterKind)
        return Loc.RegNo;
      return 0;
    }

    /// Determine whether the lexical scope of this value's debug location
    /// dominates MBB.
    bool dominates(MachineBasicBlock &MBB) const { return UVS.dominates(&MBB); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    // TRI can be null.
    void dump(const TargetRegisterInfo *TRI, raw_ostream &Out = dbgs()) const {
      dbgs() << "VarLoc(";
      switch (Kind) {
      case RegisterKind:
      case EntryValueKind:
        dbgs() << printReg(Loc.RegNo, TRI);
        break;
      case SpillLocKind:
        dbgs() << printReg(Loc.SpillLocation.SpillBase, TRI);
        dbgs() << "[" << Loc.SpillLocation.SpillOffset << "]";
        break;
      case ImmediateKind:
        dbgs() << Loc.Immediate;
        break;
      case InvalidKind:
        llvm_unreachable("Invalid VarLoc in dump method");
      }

      dbgs() << ", \"" << Var.getVar()->getName() << "\", " << *Expr << ", ";
      if (Var.getInlinedAt())
        dbgs() << "!" << Var.getInlinedAt()->getMetadataID() << ")\n";
      else
        dbgs() << "(null))\n";
    }
#endif

    bool operator==(const VarLoc &Other) const {
      return Kind == Other.Kind && Var == Other.Var &&
             Loc.Hash == Other.Loc.Hash && Expr == Other.Expr;
    }

    /// This operator guarantees that VarLocs are sorted by Variable first.
    bool operator<(const VarLoc &Other) const {
      return std::tie(Var, Kind, Loc.Hash, Expr) <
             std::tie(Other.Var, Other.Kind, Other.Loc.Hash, Other.Expr);
    }
  };

  using DebugParamMap = SmallDenseMap<const DILocalVariable *, MachineInstr *>;
  using VarLocMap = UniqueVector<VarLoc>;
  using VarLocSet = SparseBitVector<>;
  using VarLocInMBB = SmallDenseMap<const MachineBasicBlock *, VarLocSet>;
  struct TransferDebugPair {
    MachineInstr *TransferInst; /// Instruction where this transfer occurs.
    unsigned LocationID;        /// Location number for the transfer dest.
  };
  using TransferMap = SmallVector<TransferDebugPair, 4>;

  // Types for recording sets of variable fragments that overlap. For a given
  // local variable, we record all other fragments of that variable that could
  // overlap it, to reduce search time.
  using FragmentOfVar =
      std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
  using OverlapMap =
      DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// This holds the working set of currently open ranges. For fast
  /// access, this is done both as a set of VarLocIDs, and a map of
  /// DebugVariable to recent VarLocID. Note that a DBG_VALUE ends all
  /// previous open ranges for the same variable.
  class OpenRangesSet {
    VarLocSet VarLocs;
    SmallDenseMap<DebugVariable, unsigned, 8> Vars;
    OverlapMap &OverlappingFragments;

  public:
    OpenRangesSet(OverlapMap &_OLapMap) : OverlappingFragments(_OLapMap) {}

    const VarLocSet &getVarLocs() const { return VarLocs; }

    /// Terminate all open ranges for Var by removing it from the set.
    void erase(DebugVariable Var);

    /// Terminate all open ranges listed in \c KillSet by removing
    /// them from the set.
    void erase(const VarLocSet &KillSet, const VarLocMap &VarLocIDs) {
      VarLocs.intersectWithComplement(KillSet);
      for (unsigned ID : KillSet)
        Vars.erase(VarLocIDs[ID].Var);
    }

    /// Insert a new range into the set.
    void insert(unsigned VarLocID, DebugVariable Var) {
      VarLocs.set(VarLocID);
      Vars.insert({Var, VarLocID});
    }

    /// Insert a set of ranges.
    void insertFromLocSet(const VarLocSet &ToLoad, const VarLocMap &Map) {
      for (unsigned Id : ToLoad) {
        const VarLoc &Var = Map[Id];
        insert(Id, Var.Var);
      }
    }

    /// Empty the set.
    void clear() {
      VarLocs.clear();
      Vars.clear();
    }

    /// Return whether the set is empty or not.
    bool empty() const {
      assert(Vars.empty() == VarLocs.empty() && "open ranges are inconsistent");
      return VarLocs.empty();
    }
  };

  /// Tests whether this instruction is a spill to a stack location.
  bool isSpillInstruction(const MachineInstr &MI, MachineFunction *MF);

  /// Decide if @MI is a spill instruction and return true if it is. We use 2
  /// criteria to make this decision:
  /// - Is this instruction a store to a spill slot?
  /// - Is there a register operand that is both used and killed?
  /// TODO: Store optimization can fold spills into other stores (including
  /// other spills). We do not handle this yet (more than one memory operand).
  bool isLocationSpill(const MachineInstr &MI, MachineFunction *MF,
                       unsigned &Reg);

  /// If a given instruction is identified as a spill, return the spill location
  /// and set \p Reg to the spilled register.
  Optional<VarLoc::SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                                  MachineFunction *MF,
                                                  unsigned &Reg);
  /// Given a spill instruction, extract the register and offset used to
  /// address the spill location in a target independent way.
  VarLoc::SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);
  void insertTransferDebugPair(MachineInstr &MI, OpenRangesSet &OpenRanges,
                               TransferMap &Transfers, VarLocMap &VarLocIDs,
                               unsigned OldVarID, TransferKind Kind,
                               unsigned NewReg = 0);

  void transferDebugValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                          VarLocMap &VarLocIDs);
  void transferSpillOrRestoreInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                                  VarLocMap &VarLocIDs, TransferMap &Transfers);
  void emitEntryValues(MachineInstr &MI, OpenRangesSet &OpenRanges,
                       VarLocMap &VarLocIDs, TransferMap &Transfers,
                       DebugParamMap &DebugEntryVals,
                       SparseBitVector<> &KillSet);
  void transferRegisterCopy(MachineInstr &MI, OpenRangesSet &OpenRanges,
                            VarLocMap &VarLocIDs, TransferMap &Transfers);
  void transferRegisterDef(MachineInstr &MI, OpenRangesSet &OpenRanges,
                           VarLocMap &VarLocIDs, TransferMap &Transfers,
                           DebugParamMap &DebugEntryVals);
  bool transferTerminator(MachineBasicBlock *MBB, OpenRangesSet &OpenRanges,
                          VarLocInMBB &OutLocs, const VarLocMap &VarLocIDs);

  void process(MachineInstr &MI, OpenRangesSet &OpenRanges,
               VarLocMap &VarLocIDs, TransferMap &Transfers,
               DebugParamMap &DebugEntryVals);

  void accumulateFragmentMap(MachineInstr &MI, VarToFragments &SeenFragments,
                             OverlapMap &OLapMap);

  bool join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
            const VarLocMap &VarLocIDs,
            SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
            SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks,
            VarLocInMBB &PendingInLocs);

  /// Create DBG_VALUE insts for inlocs that have been propagated but
  /// had their instruction creation deferred.
  void flushPendingLocs(VarLocInMBB &PendingInLocs, VarLocMap &VarLocIDs);

  bool ExtendRanges(MachineFunction &MF);

public:
  static char ID;

  /// Default construct and initialize the pass.
  LiveDebugValues();

  /// Tell the pass manager which passes we depend on and what
  /// information we preserve.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  /// Print to ostream with a message.
  void printVarLocInMBB(const MachineFunction &MF, const VarLocInMBB &V,
                        const VarLocMap &VarLocIDs, const char *msg,
                        raw_ostream &Out) const;

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

namespace llvm {

template <> struct DenseMapInfo<LiveDebugValues::DebugVariable> {
  using DV = LiveDebugValues::DebugVariable;
  using OptFragmentInfo = LiveDebugValues::OptFragmentInfo;
  using FragmentInfo = LiveDebugValues::FragmentInfo;

  // Empty key: no key should be generated that has no DILocalVariable.
  static inline DV getEmptyKey() {
    return DV(nullptr, OptFragmentInfo(), nullptr);
  }

  // Difference in tombstone is that the Optional is meaningful
  static inline DV getTombstoneKey() {
    return DV(nullptr, OptFragmentInfo({0, 0}), nullptr);
  }

  static unsigned getHashValue(const DV &D) {
    unsigned HV = 0;
    const OptFragmentInfo &Fragment = D.getFragment();
    if (Fragment)
      HV = DenseMapInfo<FragmentInfo>::getHashValue(*Fragment);

    return hash_combine(D.getVar(), HV, D.getInlinedAt());
  }

  static bool isEqual(const DV &A, const DV &B) { return A == B; }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

const DIExpression::FragmentInfo
    LiveDebugValues::DebugVariable::DefaultFragment = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::min()};

char LiveDebugValues::ID = 0;

char &llvm::LiveDebugValuesID = LiveDebugValues::ID;

INITIALIZE_PASS(LiveDebugValues, DEBUG_TYPE, "Live DEBUG_VALUE analysis",
                false, false)

/// Default construct and initialize the pass.
LiveDebugValues::LiveDebugValues() : MachineFunctionPass(ID) {
  initializeLiveDebugValuesPass(*PassRegistry::getPassRegistry());
}

/// Tell the pass manager which passes we depend on and what information we
/// preserve.
void LiveDebugValues::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// Erase a variable from the set of open ranges, and additionally erase any
/// fragments that may overlap it.
void LiveDebugValues::OpenRangesSet::erase(DebugVariable Var) {
  // Erasure helper.
  auto DoErase = [this](DebugVariable VarToErase) {
    auto It = Vars.find(VarToErase);
    if (It != Vars.end()) {
      unsigned ID = It->second;
      VarLocs.reset(ID);
      Vars.erase(It);
    }
  };

  // Erase the variable/fragment that ends here.
  DoErase(Var);

  // Extract the fragment. Interpret an empty fragment as one that covers all
  // possible bits.
  FragmentInfo ThisFragment = Var.getFragmentDefault();

  // There may be fragments that overlap the designated fragment. Look them up
  // in the pre-computed overlap map, and erase them too.
  auto MapIt = OverlappingFragments.find({Var.getVar(), ThisFragment});
  if (MapIt != OverlappingFragments.end()) {
    for (auto Fragment : MapIt->second) {
      LiveDebugValues::OptFragmentInfo FragmentHolder;
      if (!DebugVariable::isFragmentDefault(Fragment))
        FragmentHolder = LiveDebugValues::OptFragmentInfo(Fragment);
      DoErase({Var.getVar(), FragmentHolder, Var.getInlinedAt()});
    }
  }
}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
void LiveDebugValues::printVarLocInMBB(const MachineFunction &MF,
                                       const VarLocInMBB &V,
                                       const VarLocMap &VarLocIDs,
                                       const char *msg,
                                       raw_ostream &Out) const {
  Out << '\n' << msg << '\n';
  for (const MachineBasicBlock &BB : MF) {
    const VarLocSet &L = V.lookup(&BB);
    if (L.empty())
      continue;
    Out << "MBB: " << BB.getNumber() << ":\n";
    for (unsigned VLL : L) {
      const VarLoc &VL = VarLocIDs[VLL];
      Out << " Var: " << VL.Var.getVar()->getName();
      Out << " MI: ";
      VL.dump(TRI, Out);
    }
  }
  Out << "\n";
}
#endif

LiveDebugValues::VarLoc::SpillLoc
LiveDebugValues::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
  assert(MI.hasOneMemOperand() &&
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  unsigned Reg;
  int Offset = TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
  return {Reg, Offset};
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
void LiveDebugValues::transferDebugValue(const MachineInstr &MI,
                                         OpenRangesSet &OpenRanges,
                                         VarLocMap &VarLocIDs) {
  if (!MI.isDebugValue())
    return;
  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  // End all previous ranges of Var.
  DebugVariable V(Var, Expr, InlinedAt);
  OpenRanges.erase(V);

  // Add the VarLoc to OpenRanges from this DBG_VALUE.
  unsigned ID;
  if (isDbgValueDescribedByReg(MI) || MI.getOperand(0).isImm() ||
      MI.getOperand(0).isFPImm() || MI.getOperand(0).isCImm()) {
    // Use normal VarLoc constructor for registers and immediates.
    VarLoc VL(MI, LS);
    ID = VarLocIDs.insert(VL);
    OpenRanges.insert(ID, VL.Var);
  } else if (MI.hasOneMemOperand()) {
    llvm_unreachable("DBG_VALUE with mem operand encountered after regalloc?");
  } else {
    // This must be an undefined location. We should leave OpenRanges closed.
    assert(MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == 0 &&
           "Unexpected non-undef DBG_VALUE encountered");
  }
}

void LiveDebugValues::emitEntryValues(MachineInstr &MI,
                                      OpenRangesSet &OpenRanges,
                                      VarLocMap &VarLocIDs,
                                      TransferMap &Transfers,
                                      DebugParamMap &DebugEntryVals,
                                      SparseBitVector<> &KillSet) {
  for (unsigned ID : KillSet) {
    if (!VarLocIDs[ID].Var.getVar()->isParameter())
      continue;

    const MachineInstr *CurrDebugInstr = &VarLocIDs[ID].MI;

    // If parameter's DBG_VALUE is not in the map that means we can't
    // generate parameter's entry value.
    if (!DebugEntryVals.count(CurrDebugInstr->getDebugVariable()))
      continue;

    auto ParamDebugInstr = DebugEntryVals[CurrDebugInstr->getDebugVariable()];
    DIExpression *NewExpr = DIExpression::prepend(
        ParamDebugInstr->getDebugExpression(), DIExpression::EntryValue);

    VarLoc EntryLoc = VarLoc::CreateEntryLoc(*ParamDebugInstr, LS, NewExpr);

    unsigned EntryValLocID = VarLocIDs.insert(EntryLoc);
    Transfers.push_back({&MI, EntryValLocID});
    OpenRanges.insert(EntryValLocID, EntryLoc.Var);
  }
}

/// Create new TransferDebugPair and insert it in \p Transfers. The VarLoc
/// with \p OldVarID should be deleted form \p OpenRanges and replaced with
/// new VarLoc. If \p NewReg is different than default zero value then the
/// new location will be register location created by the copy like instruction,
/// otherwise it is variable's location on the stack.
void LiveDebugValues::insertTransferDebugPair(
    MachineInstr &MI, OpenRangesSet &OpenRanges, TransferMap &Transfers,
    VarLocMap &VarLocIDs, unsigned OldVarID, TransferKind Kind,
    unsigned NewReg) {
  const MachineInstr *DebugInstr = &VarLocIDs[OldVarID].MI;

  auto ProcessVarLoc = [&MI, &OpenRanges, &Transfers, &DebugInstr,
                        &VarLocIDs](VarLoc &VL) {
    unsigned LocId = VarLocIDs.insert(VL);

    // Close this variable's previous location range.
    DebugVariable V(*DebugInstr);
    OpenRanges.erase(V);

    // Record the new location as an open range, and a postponed transfer
    // inserting a DBG_VALUE for this location.
    OpenRanges.insert(LocId, VL.Var);
    TransferDebugPair MIP = {&MI, LocId};
    Transfers.push_back(MIP);
  };

  // End all previous ranges of Var.
  OpenRanges.erase(VarLocIDs[OldVarID].Var);
  switch (Kind) {
  case TransferKind::TransferCopy: {
    assert(NewReg &&
           "No register supplied when handling a copy of a debug value");
    // Create a DBG_VALUE instruction to describe the Var in its new
    // register location.
    VarLoc VL = VarLoc::CreateCopyLoc(*DebugInstr, LS, NewReg);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for register copy:";
      VL.dump(TRI);
    });
    return;
  }
  case TransferKind::TransferSpill: {
    // Create a DBG_VALUE instruction to describe the Var in its spilled
    // location.
    VarLoc::SpillLoc SpillLocation = extractSpillBaseRegAndOffset(MI);
    VarLoc VL = VarLoc::CreateSpillLoc(*DebugInstr, SpillLocation.SpillBase,
                                       SpillLocation.SpillOffset, LS);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for spill:";
      VL.dump(TRI);
    });
    return;
  }
  case TransferKind::TransferRestore: {
    assert(NewReg &&
           "No register supplied when handling a restore of a debug value");
    // DebugInstr refers to the pre-spill location, therefore we can reuse
    // its expression.
    VarLoc VL = VarLoc::CreateCopyLoc(*DebugInstr, LS, NewReg);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for restore:";
      VL.dump(TRI);
    });
    return;
  }
  }
  llvm_unreachable("Invalid transfer kind");
}

/// A definition of a register may mark the end of a range.
void LiveDebugValues::transferRegisterDef(
    MachineInstr &MI, OpenRangesSet &OpenRanges, VarLocMap &VarLocIDs,
    TransferMap &Transfers, DebugParamMap &DebugEntryVals) {
  MachineFunction *MF = MI.getMF();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  SparseBitVector<> KillSet;
  for (const MachineOperand &MO : MI.operands()) {
    // Determine whether the operand is a register def.  Assume that call
    // instructions never clobber SP, because some backends (e.g., AArch64)
    // never list SP in the regmask.
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        Register::isPhysicalRegister(MO.getReg()) &&
        !(MI.isCall() && MO.getReg() == SP)) {
      // Remove ranges of all aliased registers.
      for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
        for (unsigned ID : OpenRanges.getVarLocs())
          if (VarLocIDs[ID].isDescribedByReg() == *RAI)
            KillSet.set(ID);
    } else if (MO.isRegMask()) {
      // Remove ranges of all clobbered registers. Register masks don't usually
      // list SP as preserved.  While the debug info may be off for an
      // instruction or two around callee-cleanup calls, transferring the
      // DEBUG_VALUE across the call is still a better user experience.
      for (unsigned ID : OpenRanges.getVarLocs()) {
        unsigned Reg = VarLocIDs[ID].isDescribedByReg();
        if (Reg && Reg != SP && MO.clobbersPhysReg(Reg))
          KillSet.set(ID);
      }
    }
  }
  OpenRanges.erase(KillSet, VarLocIDs);

  if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>()) {
    auto &TM = TPC->getTM<TargetMachine>();
    if (TM.Options.EnableDebugEntryValues)
      emitEntryValues(MI, OpenRanges, VarLocIDs, Transfers, DebugEntryVals,
                      KillSet);
  }
}

bool LiveDebugValues::isSpillInstruction(const MachineInstr &MI,
                                         MachineFunction *MF) {
  // TODO: Handle multiple stores folded into one.
  if (!MI.hasOneMemOperand())
    return false;

  if (!MI.getSpillSize(TII) && !MI.getFoldedSpillSize(TII))
    return false; // This is not a spill instruction, since no valid size was
                  // returned from either function.

  return true;
}

bool LiveDebugValues::isLocationSpill(const MachineInstr &MI,
                                      MachineFunction *MF, unsigned &Reg) {
  if (!isSpillInstruction(MI, MF))
    return false;

  auto isKilledReg = [&](const MachineOperand MO, unsigned &Reg) {
    if (!MO.isReg() || !MO.isUse()) {
      Reg = 0;
      return false;
    }
    Reg = MO.getReg();
    return MO.isKill();
  };

  for (const MachineOperand &MO : MI.operands()) {
    // In a spill instruction generated by the InlineSpiller the spilled
    // register has its kill flag set.
    if (isKilledReg(MO, Reg))
      return true;
    if (Reg != 0) {
      // Check whether next instruction kills the spilled register.
      // FIXME: Current solution does not cover search for killed register in
      // bundles and instructions further down the chain.
      auto NextI = std::next(MI.getIterator());
      // Skip next instruction that points to basic block end iterator.
      if (MI.getParent()->end() == NextI)
        continue;
      unsigned RegNext;
      for (const MachineOperand &MONext : NextI->operands()) {
        // Return true if we came across the register from the
        // previous spill instruction that is killed in NextI.
        if (isKilledReg(MONext, RegNext) && RegNext == Reg)
          return true;
      }
    }
  }
  // Return false if we didn't find spilled register.
  return false;
}

Optional<LiveDebugValues::VarLoc::SpillLoc>
LiveDebugValues::isRestoreInstruction(const MachineInstr &MI,
                                      MachineFunction *MF, unsigned &Reg) {
  if (!MI.hasOneMemOperand())
    return None;

  // FIXME: Handle folded restore instructions with more than one memory
  // operand.
  if (MI.getRestoreSize(TII)) {
    Reg = MI.getOperand(0).getReg();
    return extractSpillBaseRegAndOffset(MI);
  }
  return None;
}

/// A spilled register may indicate that we have to end the current range of
/// a variable and create a new one for the spill location.
/// A restored register may indicate the reverse situation.
/// We don't want to insert any instructions in process(), so we just create
/// the DBG_VALUE without inserting it and keep track of it in \p Transfers.
/// It will be inserted into the BB when we're done iterating over the
/// instructions.
void LiveDebugValues::transferSpillOrRestoreInst(MachineInstr &MI,
                                                 OpenRangesSet &OpenRanges,
                                                 VarLocMap &VarLocIDs,
                                                 TransferMap &Transfers) {
  MachineFunction *MF = MI.getMF();
  TransferKind TKind;
  unsigned Reg;
  Optional<VarLoc::SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, then close the variable location. The value in memory
  // will have changed.
  VarLocSet KillSet;
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);
    for (unsigned ID : OpenRanges.getVarLocs()) {
      const VarLoc &VL = VarLocIDs[ID];
      if (VL.Kind == VarLoc::SpillLocKind && VL.Loc.SpillLocation == *Loc) {
        // This location is overwritten by the current instruction -- terminate
        // the open range, and insert an explicit DBG_VALUE $noreg.
        //
        // Doing this at a later stage would require re-interpreting all
        // DBG_VALUes and DIExpressions to identify whether they point at
        // memory, and then analysing all memory writes to see if they
        // overwrite that memory, which is expensive.
        //
        // At this stage, we already know which DBG_VALUEs are for spills and
        // where they are located; it's best to fix handle overwrites now.
        KillSet.set(ID);
        VarLoc UndefVL = VarLoc::CreateCopyLoc(VL.MI, LS, 0);
        unsigned UndefLocID = VarLocIDs.insert(UndefVL);
        Transfers.push_back({&MI, UndefLocID});
      }
    }
    OpenRanges.erase(KillSet, VarLocIDs);
  }

  // Try to recognise spill and restore instructions that may create a new
  // variable location.
  if (isLocationSpill(MI, MF, Reg)) {
    TKind = TransferKind::TransferSpill;
    LLVM_DEBUG(dbgs() << "Recognized as spill: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return;
    TKind = TransferKind::TransferRestore;
    LLVM_DEBUG(dbgs() << "Recognized as restore: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  }
  // Check if the register or spill location is the location of a debug value.
  for (unsigned ID : OpenRanges.getVarLocs()) {
    if (TKind == TransferKind::TransferSpill &&
        VarLocIDs[ID].isDescribedByReg() == Reg) {
      LLVM_DEBUG(dbgs() << "Spilling Register " << printReg(Reg, TRI) << '('
                        << VarLocIDs[ID].Var.getVar()->getName() << ")\n");
    } else if (TKind == TransferKind::TransferRestore &&
               VarLocIDs[ID].Kind == VarLoc::SpillLocKind &&
               VarLocIDs[ID].Loc.SpillLocation == *Loc) {
      LLVM_DEBUG(dbgs() << "Restoring Register " << printReg(Reg, TRI) << '('
                        << VarLocIDs[ID].Var.getVar()->getName() << ")\n");
    } else
      continue;
    insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, ID, TKind,
                            Reg);
    return;
  }
}

/// If \p MI is a register copy instruction, that copies a previously tracked
/// value from one register to another register that is callee saved, we
/// create new DBG_VALUE instruction  described with copy destination register.
void LiveDebugValues::transferRegisterCopy(MachineInstr &MI,
                                           OpenRangesSet &OpenRanges,
                                           VarLocMap &VarLocIDs,
                                           TransferMap &Transfers) {
  const MachineOperand *SrcRegOp, *DestRegOp;

  if (!TII->isCopyInstr(MI, SrcRegOp, DestRegOp) || !SrcRegOp->isKill() ||
      !DestRegOp->isDef())
    return;

  auto isCalleeSavedReg = [&](unsigned Reg) {
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  };

  Register SrcReg = SrcRegOp->getReg();
  Register DestReg = DestRegOp->getReg();

  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register location, which is callee
  // saved, is going to stay unclobbered longer, even if it is killed.
  if (!isCalleeSavedReg(DestReg))
    return;

  for (unsigned ID : OpenRanges.getVarLocs()) {
    if (VarLocIDs[ID].isDescribedByReg() == SrcReg) {
      insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, ID,
                              TransferKind::TransferCopy, DestReg);
      return;
    }
  }
}

/// Terminate all open ranges at the end of the current basic block.
bool LiveDebugValues::transferTerminator(MachineBasicBlock *CurMBB,
                                         OpenRangesSet &OpenRanges,
                                         VarLocInMBB &OutLocs,
                                         const VarLocMap &VarLocIDs) {
  bool Changed = false;

  LLVM_DEBUG(for (unsigned ID
                  : OpenRanges.getVarLocs()) {
    // Copy OpenRanges to OutLocs, if not already present.
    dbgs() << "Add to OutLocs in MBB #" << CurMBB->getNumber() << ":  ";
    VarLocIDs[ID].dump(TRI);
  });
  VarLocSet &VLS = OutLocs[CurMBB];
  Changed = VLS != OpenRanges.getVarLocs();
  // New OutLocs set may be different due to spill, restore or register
  // copy instruction processing.
  if (Changed)
    VLS = OpenRanges.getVarLocs();
  OpenRanges.clear();
  return Changed;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
/// \param SeenFragments Map from DILocalVariable to all fragments of that
///           Variable which are known to exist.
/// \param OverlappingFragments The overlap map being constructed, from one
///           Var/Fragment pair to a vector of fragments known to overlap.
void LiveDebugValues::accumulateFragmentMap(MachineInstr &MI,
                                            VarToFragments &SeenFragments,
                                            OverlapMap &OverlappingFragments) {
  DebugVariable MIVar(MI);
  FragmentInfo ThisFragment = MIVar.getFragmentDefault();

  // If this is the first sighting of this variable, then we are guaranteed
  // there are currently no overlapping fragments either. Initialize the set
  // of seen fragments, record no overlaps for the current one, and return.
  auto SeenIt = SeenFragments.find(MIVar.getVar());
  if (SeenIt == SeenFragments.end()) {
    SmallSet<FragmentInfo, 4> OneFragment;
    OneFragment.insert(ThisFragment);
    SeenFragments.insert({MIVar.getVar(), OneFragment});

    OverlappingFragments.insert({{MIVar.getVar(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlappingFragments.insert({{MIVar.getVar(), ThisFragment}, {}});
  if (!IsInOLapMap.second)
    return;

  auto &ThisFragmentsOverlaps = IsInOLapMap.first->second;
  auto &AllSeenFragments = SeenIt->second;

  // Otherwise, examine all other seen fragments for this variable, with "this"
  // fragment being a previously unseen fragment. Record any pair of
  // overlapping fragments.
  for (auto &ASeenFragment : AllSeenFragments) {
    // Does this previously seen fragment overlap?
    if (DIExpression::fragmentsOverlap(ThisFragment, ASeenFragment)) {
      // Yes: Mark the current fragment as being overlapped.
      ThisFragmentsOverlaps.push_back(ASeenFragment);
      // Mark the previously seen fragment as being overlapped by the current
      // one.
      auto ASeenFragmentsOverlaps =
          OverlappingFragments.find({MIVar.getVar(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlappingFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

/// This routine creates OpenRanges.
void LiveDebugValues::process(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocMap &VarLocIDs, TransferMap &Transfers,
                              DebugParamMap &DebugEntryVals) {
  transferDebugValue(MI, OpenRanges, VarLocIDs);
  transferRegisterDef(MI, OpenRanges, VarLocIDs, Transfers,
                      DebugEntryVals);
  transferRegisterCopy(MI, OpenRanges, VarLocIDs, Transfers);
  transferSpillOrRestoreInst(MI, OpenRanges, VarLocIDs, Transfers);
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool LiveDebugValues::join(
    MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
    const VarLocMap &VarLocIDs,
    SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks,
    VarLocInMBB &PendingInLocs) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  VarLocSet InLocsT; // Temporary incoming locations.

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  for (auto p : MBB.predecessors()) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had locations propagated into it, most locations
    // will not yet be valid, so treat them as all being uninitialized and
    // potentially valid. If a location guessed to be correct here is
    // invalidated later, we will remove it when we revisit this block.
    if (!Visited.count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }
    auto OL = OutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == OutLocs.end())
      return false;

    // Just copy over the Out locs to incoming locs for the first visited
    // predecessor, and for all other predecessors join the Out locs.
    if (!NumVisited)
      InLocsT = OL->second;
    else
      InLocsT &= OL->second;

    LLVM_DEBUG({
      if (!InLocsT.empty()) {
        for (auto ID : InLocsT)
          dbgs() << "  gathered candidate incoming var: "
                 << VarLocIDs[ID].Var.getVar()->getName() << "\n";
      }
    });

    NumVisited++;
  }

  // Filter out DBG_VALUES that are out of scope.
  VarLocSet KillSet;
  bool IsArtificial = ArtificialBlocks.count(&MBB);
  if (!IsArtificial) {
    for (auto ID : InLocsT) {
      if (!VarLocIDs[ID].dominates(MBB)) {
        KillSet.set(ID);
        LLVM_DEBUG({
          auto Name = VarLocIDs[ID].Var.getVar()->getName();
          dbgs() << "  killing " << Name << ", it doesn't dominate MBB\n";
        });
      }
    }
  }
  InLocsT.intersectWithComplement(KillSet);

  // As we are processing blocks in reverse post-order we
  // should have processed at least one predecessor, unless it
  // is the entry block which has no predecessor.
  assert((NumVisited || MBB.pred_empty()) &&
         "Should have processed at least one predecessor");

  VarLocSet &ILS = InLocs[&MBB];
  VarLocSet &Pending = PendingInLocs[&MBB];

  // New locations will have DBG_VALUE insts inserted at the start of the
  // block, after location propagation has finished. Record the insertions
  // that we need to perform in the Pending set.
  VarLocSet Diff = InLocsT;
  Diff.intersectWithComplement(ILS);
  for (auto ID : Diff) {
    Pending.set(ID);
    ILS.set(ID);
    ++NumInserted;
    Changed = true;
  }

  // We may have lost locations by learning about a predecessor that either
  // loses or moves a variable. Find any locations in ILS that are not in the
  // new in-locations, and delete those.
  VarLocSet Removed = ILS;
  Removed.intersectWithComplement(InLocsT);
  for (auto ID : Removed) {
    Pending.reset(ID);
    ILS.reset(ID);
    ++NumRemoved;
    Changed = true;
  }

  return Changed;
}

void LiveDebugValues::flushPendingLocs(VarLocInMBB &PendingInLocs,
                                       VarLocMap &VarLocIDs) {
  // PendingInLocs records all locations propagated into blocks, which have
  // not had DBG_VALUE insts created. Go through and create those insts now.
  for (auto &Iter : PendingInLocs) {
    // Map is keyed on a constant pointer, unwrap it so we can insert insts.
    auto &MBB = const_cast<MachineBasicBlock &>(*Iter.first);
    VarLocSet &Pending = Iter.second;

    for (unsigned ID : Pending) {
      // The ID location is live-in to MBB -- work out what kind of machine
      // location it is and create a DBG_VALUE.
      const VarLoc &DiffIt = VarLocIDs[ID];
      MachineInstr *MI = DiffIt.BuildDbgValue(*MBB.getParent());
      MBB.insert(MBB.instr_begin(), MI);

      (void)MI;
      LLVM_DEBUG(dbgs() << "Inserted: "; MI->dump(););
    }
  }
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  bool Changed = false;
  bool OLChanged = false;
  bool MBBJoined = false;

  VarLocMap VarLocIDs;         // Map VarLoc<>unique ID for use in bitvectors.
  OverlapMap OverlapFragments; // Map of overlapping variable fragments.
  OpenRangesSet OpenRanges(OverlapFragments);
                              // Ranges that are open until end of bb.
  VarLocInMBB OutLocs;        // Ranges that exist beyond bb.
  VarLocInMBB InLocs;         // Ranges that are incoming after joining.
  TransferMap Transfers;      // DBG_VALUEs associated with transfers (such as
                              // spills, copies and restores).
  VarLocInMBB PendingInLocs;  // Ranges that are incoming after joining, but
                              // that we have deferred creating DBG_VALUE insts
                              // for immediately.

  VarToFragments SeenFragments;

  // Blocks which are artificial, i.e. blocks which exclusively contain
  // instructions without locations, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;

  // Besides parameter's modification, check whether a DBG_VALUE is inlined
  // in order to deduce whether the variable that it tracks comes from
  // a different function. If that is the case we can't track its entry value.
  auto IsUnmodifiedFuncParam = [&](const MachineInstr &MI) {
    auto *DIVar = MI.getDebugVariable();
    return DIVar->isParameter() && DIVar->isNotModified() &&
           !MI.getDebugLoc()->getInlinedAt();
  };

  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  Register FP = TRI->getFrameRegister(MF);
  auto IsRegOtherThanSPAndFP = [&](const MachineOperand &Op) -> bool {
    return Op.isReg() && Op.getReg() != SP && Op.getReg() != FP;
  };

  // Working set of currently collected debug variables mapped to DBG_VALUEs
  // representing candidates for production of debug entry values.
  DebugParamMap DebugEntryVals;

  MachineBasicBlock &First_MBB = *(MF.begin());
  // Only in the case of entry MBB collect DBG_VALUEs representing
  // function parameters in order to generate debug entry values for them.
  // Currently, we generate debug entry values only for parameters that are
  // unmodified throughout the function and located in a register.
  // TODO: Add support for parameters that are described as fragments.
  // TODO: Add support for modified arguments that can be expressed
  // by using its entry value.
  // TODO: Add support for local variables that are expressed in terms of
  // parameters entry values.
  for (auto &MI : First_MBB)
    if (MI.isDebugValue() && IsUnmodifiedFuncParam(MI) &&
        !MI.isIndirectDebugValue() && IsRegOtherThanSPAndFP(MI.getOperand(0)) &&
        !DebugEntryVals.count(MI.getDebugVariable()) &&
        !MI.getDebugExpression()->isFragment())
      DebugEntryVals[MI.getDebugVariable()] = &MI;

  // Initialize per-block structures and scan for fragment overlaps.
  for (auto &MBB : MF) {
    PendingInLocs[&MBB] = VarLocSet();

    for (auto &MI : MBB) {
      if (MI.isDebugValue())
        accumulateFragmentMap(MI, SeenFragments, OverlapFragments);
    }
  }

  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                              "OutLocs after initialization", dbgs()));

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    Worklist.push(RPONumber);
    ++RPONumber;
  }
  // This is a standard "union of predecessor outs" dataflow problem.
  // To solve it, we perform join() and process() using the two worklist method
  // until the ranges converge.
  // Ranges have converged when both worklists are empty.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    LLVM_DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      Worklist.pop();
      MBBJoined = join(*MBB, OutLocs, InLocs, VarLocIDs, Visited,
                       ArtificialBlocks, PendingInLocs);
      MBBJoined |= Visited.insert(MBB).second;
      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;
        // Now that we have started to extend ranges across BBs we need to
        // examine spill, copy and restore instructions to see whether they
        // operate with registers that correspond to user variables.
        // First load any pending inlocs.
        OpenRanges.insertFromLocSet(PendingInLocs[MBB], VarLocIDs);
        for (auto &MI : *MBB)
          process(MI, OpenRanges, VarLocIDs, Transfers, DebugEntryVals);
        OLChanged |= transferTerminator(MBB, OpenRanges, OutLocs, VarLocIDs);

        LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                                    "OutLocs after propagating", dbgs()));
        LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs,
                                    "InLocs after propagating", dbgs()));

        if (OLChanged) {
          OLChanged = false;
          for (auto s : MBB->successors())
            if (OnPending.insert(s).second) {
              Pending.push(BBToOrder[s]);
            }
        }
      }
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  // Add any DBG_VALUE instructions created by location transfers.
  for (auto &TR : Transfers) {
    MachineBasicBlock *MBB = TR.TransferInst->getParent();
    const VarLoc &VL = VarLocIDs[TR.LocationID];
    MachineInstr *MI = VL.BuildDbgValue(MF);
    MBB->insertAfterBundle(TR.TransferInst->getIterator(), MI);
  }
  Transfers.clear();

  // Deferred inlocs will not have had any DBG_VALUE insts created; do
  // that now.
  flushPendingLocs(PendingInLocs, VarLocIDs);

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs, "Final OutLocs", dbgs()));
  LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs, "Final InLocs", dbgs()));
  return Changed;
}

bool LiveDebugValues::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().getSubprogram())
    // LiveDebugValues will already have removed all DBG_VALUEs.
    return false;

  // Skip functions from NoDebug compilation units.
  if (MF.getFunction().getSubprogram()->getUnit()->getEmissionKind() ==
      DICompileUnit::NoDebug)
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  TFI->determineCalleeSaves(MF, CalleeSavedRegs,
                            std::make_unique<RegScavenger>().get());
  LS.initialize(MF);

  bool Changed = ExtendRanges(MF);
  return Changed;
}
