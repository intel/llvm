//===-- LowerESIMD.cpp - lower Explicit SIMD (ESIMD) constructs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See intro comments in the header.
//
// Since the spir* targets use Itanium mangling for C/C++ symbols, the
// implementation uses the Itanium demangler to demangle device code's
// C++ intrinsics and access various information, such their C++ names and
// values of integer template parameters they were instantiated with.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

#include "../../IR/ConstantsContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
#include "llvm/GenXIntrinsics/GenXMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include <cctype>
#include <cstring>
#include <unordered_map>

using namespace llvm;
namespace id = itanium_demangle;
using namespace llvm::esimd;

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-esimd"

#define SLM_BTI 254

#define MAX_DIMS 3

cl::opt<bool> ForceStatelessMem(
    "lower-esimd-force-stateless-mem", llvm::cl::Optional, llvm::cl::Hidden,
    llvm::cl::desc("Use stateless API for accessor based API."),
    llvm::cl::init(false));

namespace {
SmallPtrSet<Type *, 4> collectGenXVolatileTypes(Module &);
void generateKernelMetadata(Module &);

class SYCLLowerESIMDLegacyPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  SYCLLowerESIMDLegacyPass() : ModulePass(ID) {
    initializeSYCLLowerESIMDLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // run the LowerESIMD pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLLowerESIMDPass Impl;
};
} // namespace

char SYCLLowerESIMDLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLLowerESIMDLegacyPass, "LowerESIMD",
                "Lower constructs specific to the 'explicit SIMD' extension",
                false, false)

// Public interface to the SYCLLowerESIMDPass.
ModulePass *llvm::createSYCLLowerESIMDPass() {
  return new SYCLLowerESIMDLegacyPass();
}

namespace {
enum class lsc_subopcode : uint8_t {
  load = 0x00,
  load_strided = 0x01,
  load_quad = 0x02,
  load_block2d = 0x03,
  store = 0x04,
  store_strided = 0x05,
  store_quad = 0x06,
  store_block2d = 0x07,
  //
  atomic_iinc = 0x08,
  atomic_idec = 0x09,
  atomic_load = 0x0a,
  atomic_store = 0x0b,
  atomic_iadd = 0x0c,
  atomic_isub = 0x0d,
  atomic_smin = 0x0e,
  atomic_smax = 0x0f,
  atomic_umin = 0x10,
  atomic_umax = 0x11,
  atomic_icas = 0x12,
  atomic_fadd = 0x13,
  atomic_fsub = 0x14,
  atomic_fmin = 0x15,
  atomic_fmax = 0x16,
  atomic_fcas = 0x17,
  atomic_and = 0x18,
  atomic_or = 0x19,
  atomic_xor = 0x1a,
  //
  load_status = 0x1b,
  store_uncompressed = 0x1c,
  ccs_update = 0x1d,
  read_state_info = 0x1e,
  fence = 0x1f,
};
// The regexp for ESIMD intrinsics:
// /^_Z(\d+)__esimd_\w+/
static constexpr char ESIMD_INTRIN_PREF0[] = "_Z";
static constexpr char ESIMD_INTRIN_PREF1[] = "__esimd_";
static constexpr char ESIMD_INSERTED_VSTORE_FUNC_NAME[] = "_Z14__esimd_vstorev";
static constexpr char SPIRV_INTRIN_PREF[] = "__spirv_BuiltIn";
struct ESIMDIntrinDesc {
  // Denotes argument translation rule kind.
  enum GenXArgRuleKind {
    SRC_CALL_ARG, // is a call argument
    SRC_CALL_ALL, // this and subsequent args are just copied from the src call
    SRC_TMPL_ARG, // is an integer template argument
    UNDEF,        // is an undef value
    CONST_INT8,   // is an i8 constant
    CONST_INT16,  // is an i16 constant
    CONST_INT32,  // is an i32 constant
    CONST_INT64,  // is an i64 constant
  };

  enum class GenXArgConversion : int16_t {
    NONE,   // no conversion
    TO_I1,  // convert vector of N-bit integer to 1-bit
    TO_I8,  // convert vector of N-bit integer to 8-bit
    TO_I16, // convert vector of N-bit integer to 16-bit
    TO_I32, // convert vector of N-bit integer to 32-bit
    TO_I64, // convert vector of N-bit integer to 64-bit
  };

  // Denotes GenX intrinsic name suffix creation rule kind.
  enum GenXSuffixRuleKind {
    NO_RULE,
    BIN_OP,  // ".<binary operation>" - e.g. "*.add"
    NUM_KIND // "<numeric kind>" - e.g. "*i" for integer, "*f" for float
  };

  // Represents a rule how a GenX intrinsic argument is created from the source
  // call instruction.
  struct ArgRule {
    GenXArgRuleKind Kind;
    union Info {
      struct {
        int16_t CallArgNo;      // SRC_CALL_ARG: source call arg num
                                // SRC_TMPL_ARG: source template arg num
                                // UNDEF: source call arg num to get type from
                                // -1 denotes return value
        GenXArgConversion Conv; // GenXArgConversion
      } Arg;
      int NRemArgs;          // SRC_CALL_ALL: number of remaining args
      unsigned int ArgConst; // CONST_I16 OR CONST_I32: constant value
    } I;
  };

  // Represents a rule how a GenX intrinsic name suffix is created from the
  // source call instruction.
  struct NameRule {
    GenXSuffixRuleKind Kind;
    union Info {
      int CallArgNo; // DATA_TYPE: source call arg num to get type from
      int TmplArgNo; // BINOP: source template arg num denoting the binary op
    } I;
  };

  std::string GenXSpelling;
  SmallVector<ArgRule, 16> ArgRules;
  NameRule SuffixRule = {NO_RULE, {0}};

  int getNumGenXArgs() const {
    auto NRules = ArgRules.size();

    if (NRules == 0)
      return 0;

    // SRC_CALL_ALL is a "shortcut" to save typing, must be the last rule
    if (ArgRules[NRules - 1].Kind == GenXArgRuleKind::SRC_CALL_ALL)
      return ArgRules[NRules - 1].I.NRemArgs + (NRules - 1);
    return NRules;
  }

  bool isValid() const { return !GenXSpelling.empty(); }
};

using IntrinTable = std::unordered_map<std::string, ESIMDIntrinDesc>;

class ESIMDIntrinDescTable {
private:
  IntrinTable Table;

#define DEF_ARG_RULE(Nm, Kind)                                                 \
  static constexpr ESIMDIntrinDesc::ArgRule Nm(int16_t N) {                    \
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::Kind, {{N, {}}}};         \
  }
  DEF_ARG_RULE(l, SRC_CALL_ALL)
  DEF_ARG_RULE(u, UNDEF)

  static constexpr ESIMDIntrinDesc::ArgRule t(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::NONE}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t1(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I1}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t8(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I8}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t16(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I16}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t32(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I32}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t64(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I64}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule a(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_CALL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::NONE}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule ai1(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_CALL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I1}}};
  }

  // Just an alias for a(int16_t N) to mark surface index arguments.
  static constexpr ESIMDIntrinDesc::ArgRule aSI(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_CALL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::NONE}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c8(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT8, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c8(lsc_subopcode OpCode) {
    return c8(static_cast<uint8_t>(OpCode));
  }

  static constexpr ESIMDIntrinDesc::ArgRule c16(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT16, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c32(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT32, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c64(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT64, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::NameRule bo(int16_t N) {
    return ESIMDIntrinDesc::NameRule{ESIMDIntrinDesc::BIN_OP, {N}};
  }

  static constexpr ESIMDIntrinDesc::NameRule nk(int16_t N) {
    return ESIMDIntrinDesc::NameRule{ESIMDIntrinDesc::NUM_KIND, {N}};
  }

public:
  // The table which describes rules how to generate @llvm.genx.* intrinsics
  // from templated __esimd* intrinsics. The general rule is that the order and
  // the semantics of intrinsic arguments is the same in both intrinsic forms.
  // But for some arguments, where @llvm.genx.* mandates that the argument must
  // be 'constant' (see Intrinsic_definitions.py from the vcintrinsics repo),
  // it is passed as template argument to the corrsponding __esimd* intrinsic,
  // hence leading to some "gaps" in __esimd* form's arguments compared to the
  // @llvm.genx.* form.
  // TODO - fix all __esimd* intrinsics and table entries according to the rule
  // above.
  ESIMDIntrinDescTable() {
    // clang-format off
    Table = {
        // An element of the table is std::pair of <key, value>; key is the
        // source
        // spelling of and intrinsic (what follows the "__esimd_" prefix), and
        // the
        // value is an instance of the ESIMDIntrinDesc class.
        // Example for the "rdregion" intrinsic encoding:
        // "rdregion" - the GenX spelling of the intrinsic ("llvm.genx." prefix
        //      and type suffixes maybe added to get full GenX name)
        // {a(0), t(3),...}
        //      defines a map from the resulting genx.* intrinsic call arguments
        //      to the source call's template or function call arguments, e.g.
        //      0th genx arg - maps to 0th source call arg
        //      1st genx arg - maps to 3rd template argument of the source call
        // nk(N) or bo(N)
        //      a rule applied to the base intrinsic name in order to
        //      construct a full name ("llvm.genx." prefix s also added); e.g.
        //      - nk(-1) denotes adding the return type name-based suffix - "i"
        //          for integer, "f" - for floating point
        {"rdregion",
         {"rdregion", {a(0), t(3), t(4), t(5), a(1), t(6)}, nk(-1)}},
        {"rdindirect",
         {"rdregion", {a(0), c32(0), c32(1), c32(0), a(1), t(3)}, nk(-1)}},
        {{"wrregion"},
         {{"wrregion"},
          {a(0), a(1), t(3), t(4), t(5), a(2), t(6), ai1(3)},
          nk(-1)}},
        {{"wrindirect"},
         {{"wrregion"},
          {a(0), a(1), c32(0), c32(1), c32(0), a(2), t(3), ai1(3)},
          nk(-1)}},
        {"vload", {"vload", {l(0)}}},
        {"vstore", {"vstore", {a(1), a(0)}}},
        {"svm_gather", {"svm.gather", {ai1(1), t(3), a(0), u(-1)}}},
        {"svm_gather4_scaled",
         {"svm.gather4.scaled", {ai1(1), t(2), c16(0), c64(0), a(0), u(-1)}}},
        {"svm_scatter", {"svm.scatter", {ai1(2), t(3), a(0), a(1)}}},
        {"svm_scatter4_scaled",
         {"svm.scatter4.scaled", {ai1(2), t(2), c16(0), c64(0), a(0), a(1)}}},

        {"svm_atomic0", {"svm.atomic", {ai1(1), a(0), u(-1)}, bo(0)}},
        {"svm_atomic1", {"svm.atomic", {ai1(2), a(0), a(1), u(-1)}, bo(0)}},
        {"svm_atomic2",
         {"svm.atomic", {ai1(3), a(0), a(1), a(2), u(-1)}, bo(0)}},
        {"dp4", {"dp4", {a(0), a(1)}}},

        {"fence", {"fence", {a(0)}}},
        {"barrier", {"barrier", {}}},
        {"sbarrier", {"sbarrier", {a(0)}}},

        // arg0: i32 modifiers, constant
        // arg1: i32 surface index
        // arg2: i32 plane, constant
        // arg3: i32 block width in bytes, constant
        // (block height inferred from return type size and block width)
        // arg4: i32 x byte offset
        // arg5: i32 y byte offset
        {"media_ld", {"media.ld", {t(3), aSI(0), t(5), t(6), a(1), a(2)}}},

        // arg0: i32 modifiers, constant
        // arg1: i32 surface index
        // arg2: i32 plane, constant
        // arg3: i32 block width in bytes, constant
        // (block height inferred from data type size and block width)
        // arg4: i32 x byte offset
        // arg5: i32 y byte offset
        // arg6: data to write (overloaded)
        {"media_st",
         {"media.st", {t(3), aSI(0), t(5), t(6), a(1), a(2), a(3)}}},

        // arg0 : i32 is_modified, CONSTANT
        // arg1 : i32 surface index
        // arg2 : i32 offset(in owords for.ld / in bytes for.ld.unaligned)
        {"oword_ld_unaligned", {"oword.ld.unaligned", {t(3), aSI(0), a(1)}}},
        {"oword_ld", {"oword.ld", {t(3), aSI(0), a(1)}}},

        // arg0: i32 surface index
        // arg1: i32 offset (in owords)
        // arg2: data to write (overloaded)
        {"oword_st", {"oword.st", {aSI(0), a(1), a(2)}}},

        // arg0: i32 log2 num blocks, CONSTANT (0/1/2 for num blocks 1/2/4)
        // arg1: i16 scale, CONSTANT
        // arg2: i32 surface index
        // arg3: i32 global offset in bytes
        // arg4: vXi32 element offset in bytes (overloaded)
        // arg5: vXi1 predicate (overloaded)
        {"gather_masked_scaled2",
         {"gather.masked.scaled2", {t(3), t(4), aSI(0), a(1), a(2), ai1(3)}}},

        // arg0: i32 channel mask, CONSTANT
        // arg1: i16 scale, CONSTANT
        // arg2: i32 surface index
        // arg3: i32 global offset in bytes
        // arg4: vXi32 element offset in bytes
        // arg5: vXi1 predicate (overloaded)
        {"gather4_masked_scaled2",
         {"gather4.masked.scaled2", {t(2), t(4), aSI(0), a(1), a(2), ai1(3)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 log2 num blocks, CONSTANT (0/1/2 for num blocks 1/2/4)
        // arg2: i16 scale, CONSTANT
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset (overloaded)
        // arg6: data to write (overloaded)
        {"scatter_scaled",
         {"scatter.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), a(4)}}},

        // arg0: vXi1 predicate (overloaded) (overloaded)
        // arg1: i32 channel mask, CONSTANT
        // arg2: i16 scale, CONSTANT
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset in bytes (overloaded)
        // arg6: old value of the data read
        {"gather4_scaled",
         {"gather4.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), u(-1)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 channel mask, constant
        // arg2: i16 scale, constant
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset in bytes (overloaded)
        // arg6: data to write (overloaded)
        {"scatter4_scaled",
         {"scatter4.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), a(4)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 surface index
        // arg2: vXi32 element offset in bytes
        // arg3: vXi32 original value of the register that the data is read into
        {"dword_atomic0",
         {"dword.atomic", {ai1(0), aSI(1), a(2), u(-1)}, bo(0)}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 surface index
        // arg2: vXi32 element offset in bytes (overloaded)
        // arg3: vXi32/vXfloat src
        // arg4: vXi32/vXfloat original value of the register that the data is
        // read into
        {"dword_atomic1",
         {"dword.atomic", {ai1(0), aSI(1), a(2), a(3), u(-1)}, bo(0)}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 surface index
        // arg2: vXi32 element offset in bytes
        // arg3: vXi32 src0
        // arg4: vXi32 src1
        // arg5: vXi32 original value of the register that the data is read into
        {"dword_atomic2",
         {"dword.atomic", {ai1(0), aSI(1), a(2), a(3), a(4), u(-1)}, bo(0)}},

        {"raw_sends2",
         {"raw.sends2",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10),
           a(11)}}},
        {"raw_send2",
         {"raw.send2",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9)}}},
        {"raw_sends2_noresult",
         {"raw.sends2.noresult",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9)}}},
        {"raw_send2_noresult",
         {"raw.send2.noresult",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7)}}},
        {"wait", {"dummy.mov", {a(0)}}},
        {"dpas2",
         {"dpas2", {a(0), a(1), a(2), t(0), t(1), t(2), t(3), t(11), t(12)}}},
        {"dpas_nosrc0", {"dpas.nosrc0", {a(0), a(1), t(0)}}},
        {"dpasw", {"dpasw", {a(0), a(1), a(2), t(0)}}},
        {"dpasw_nosrc0", {"dpasw.nosrc0", {a(0), a(1), t(0)}}},
        {"nbarrier", {"nbarrier", {a(0), a(1), a(2)}}},
        {"raw_send_nbarrier_signal",
         {"raw.send.noresult", {a(0), ai1(4), a(1), a(2), a(3)}}},
        {"lsc_load_slm",
         {"lsc.load.slm",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0)}}},
        {"lsc_load_merge_slm",
         {"lsc.load.merge.slm",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0), a(2)}}},
        {"lsc_load_bti",
         {"lsc.load.bti",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), aSI(2)}}},
        {"lsc_load_merge_bti",
         {"lsc.load.merge.bti",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), aSI(2), a(3)}}},
        {"lsc_load_stateless",
         {"lsc.load.stateless",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0)}}},
        {"lsc_load_merge_stateless",
         {"lsc.load.merge.stateless",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0), a(2)}}},
        {"lsc_prefetch_bti",
         {"lsc.prefetch.bti",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), aSI(2)}}},
        {"lsc_prefetch_stateless",
         {"lsc.prefetch.stateless",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0)}}},
        {"lsc_store_slm",
         {"lsc.store.slm",
          {ai1(0), c8(lsc_subopcode::store), t8(1), t8(2), t16(3), t32(4),
           t8(5), t8(6), t8(7), c8(0), a(1), a(2), c32(0)}}},
        {"lsc_store_bti",
         {"lsc.store.bti",
          {ai1(0), c8(lsc_subopcode::store), t8(1), t8(2), t16(3), t32(4),
           t8(5), t8(6), t8(7), c8(0), a(1), a(2), aSI(3)}}},
        {"lsc_store_stateless",
         {"lsc.store.stateless",
          {ai1(0), c8(lsc_subopcode::store), t8(1), t8(2), t16(3), t32(4),
           t8(5), t8(6), t8(7), c8(0), a(1), a(2), c32(0)}}},
        {"lsc_load2d_stateless",
         {"lsc.load2d.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t8(4), t8(5), t16(6), t16(7), t8(8),
           a(1), a(2), a(3), a(4), a(5), a(6)}}},
        {"lsc_prefetch2d_stateless",
         {"lsc.prefetch2d.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t8(4), t8(5), t16(6), t16(7), t8(8),
           a(1), a(2), a(3), a(4), a(5), a(6)}}},
        {"lsc_store2d_stateless",
         {"lsc.store2d.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t8(4), t8(5), t16(6), t16(7), t8(8),
           a(1), a(2), a(3), a(4), a(5), a(6), a(7)}}},
        {"lsc_xatomic_slm_0",
         {"lsc.xatomic.slm",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), u(-1), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_slm_1",
         {"lsc.xatomic.slm",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_slm_2",
         {"lsc.xatomic.slm",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), a(3), c32(0), u(-1)}}},
        {"lsc_xatomic_bti_0",
         {"lsc.xatomic.bti",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), u(-1), u(-1), aSI(2), u(-1)}}},
        {"lsc_xatomic_bti_1",
         {"lsc.xatomic.bti",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), u(-1), aSI(3), u(-1)}}},
        {"lsc_xatomic_bti_2",
         {"lsc.xatomic.bti",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), a(3), aSI(4), u(-1)}}},
        {"lsc_xatomic_stateless_0",
         {"lsc.xatomic.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), u(-1), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_stateless_1",
         {"lsc.xatomic.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_stateless_2",
         {"lsc.xatomic.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), a(3), c32(0), u(-1)}}},
        {"lsc_fence", {"lsc.fence", {ai1(0), t8(0), t8(1), t8(2)}}},
        {"sat", {"sat", {a(0)}}},
        {"fptoui_sat", {"fptoui.sat", {a(0)}}},
        {"fptosi_sat", {"fptosi.sat", {a(0)}}},
        {"uutrunc_sat", {"uutrunc.sat", {a(0)}}},
        {"ustrunc_sat", {"ustrunc.sat", {a(0)}}},
        {"sutrunc_sat", {"sutrunc.sat", {a(0)}}},
        {"sstrunc_sat", {"sstrunc.sat", {a(0)}}},
        {"abs", {"abs", {a(0)}, nk(-1)}},
        {"ssshl", {"ssshl", {a(0), a(1)}}},
        {"sushl", {"sushl", {a(0), a(1)}}},
        {"usshl", {"usshl", {a(0), a(1)}}},
        {"uushl", {"uushl", {a(0), a(1)}}},
        {"ssshl_sat", {"ssshl.sat", {a(0), a(1)}}},
        {"sushl_sat", {"sushl.sat", {a(0), a(1)}}},
        {"usshl_sat", {"usshl.sat", {a(0), a(1)}}},
        {"uushl_sat", {"uushl.sat", {a(0), a(1)}}},
        {"rol", {"rol", {a(0), a(1)}}},
        {"ror", {"ror", {a(0), a(1)}}},
        {"rndd", {"rndd", {a(0)}}},
        {"rnde", {"rnde", {a(0)}}},
        {"rndu", {"rndu", {a(0)}}},
        {"rndz", {"rndz", {a(0)}}},
        {"umulh", {"umulh", {a(0), a(1)}}},
        {"smulh", {"smulh", {a(0), a(1)}}},
        {"frc", {"frc", {a(0)}}},
        {"fmax", {"fmax", {a(0), a(1)}}},
        {"umax", {"umax", {a(0), a(1)}}},
        {"smax", {"smax", {a(0), a(1)}}},
        {"lzd", {"lzd", {a(0)}}},
        {"fmin", {"fmin", {a(0), a(1)}}},
        {"umin", {"umin", {a(0), a(1)}}},
        {"smin", {"smin", {a(0), a(1)}}},
        {"bfrev", {"bfrev", {a(0)}}},
        {"cbit", {"cbit", {a(0)}}},
        {"bfi", {"bfi", {a(0), a(1), a(2), a(3)}}},
        {"sbfe", {"sbfe", {a(0), a(1), a(2)}}},
        {"fbl", {"fbl", {a(0)}}},
        {"sfbh", {"sfbh", {a(0)}}},
        {"ufbh", {"ufbh", {a(0)}}},
        {"inv", {"inv", {a(0)}}},
        {"log", {"log", {a(0)}}},
        {"exp", {"exp", {a(0)}}},
        {"sqrt", {"sqrt", {a(0)}}},
        {"ieee_sqrt", {"ieee.sqrt", {a(0)}}},
        {"rsqrt", {"rsqrt", {a(0)}}},
        {"sin", {"sin", {a(0)}}},
        {"cos", {"cos", {a(0)}}},
        {"pow", {"pow", {a(0), a(1)}}},
        {"ieee_div", {"ieee.div", {a(0), a(1)}}},
        {"uudp4a", {"uudp4a", {a(0), a(1), a(2)}}},
        {"usdp4a", {"usdp4a", {a(0), a(1), a(2)}}},
        {"sudp4a", {"sudp4a", {a(0), a(1), a(2)}}},
        {"ssdp4a", {"ssdp4a", {a(0), a(1), a(2)}}},
        {"uudp4a_sat", {"uudp4a.sat", {a(0), a(1), a(2)}}},
        {"usdp4a_sat", {"usdp4a.sat", {a(0), a(1), a(2)}}},
        {"sudp4a_sat", {"sudp4a.sat", {a(0), a(1), a(2)}}},
        {"ssdp4a_sat", {"ssdp4a.sat", {a(0), a(1), a(2)}}},
        {"any", {"any", {ai1(0)}}},
        {"all", {"all", {ai1(0)}}},
        {"lane_id", {"lane.id", {}}},
        {"test_src_tmpl_arg",
         {"test.src.tmpl.arg", {t(0), t1(1), t8(2), t16(3), t32(4), c8(17)}}},
        {"slm_init", {"slm.init", {a(0)}}},
        {"bf_cvt", {"bf.cvt", {a(0)}}},
        {"tf32_cvt", {"tf32.cvt", {a(0)}}},
        {"__devicelib_ConvertFToBF16INTEL",
         {"__spirv_ConvertFToBF16INTEL", {a(0)}}},
        {"__devicelib_ConvertBF16ToFINTEL",
         {"__spirv_ConvertBF16ToFINTEL", {a(0)}}},
        {"addc", {"addc", {l(0)}}},
        {"subb", {"subb", {l(0)}}},
        {"bfn", {"bfn", {a(0), a(1), a(2), t(0)}}},
        {"srnd", {"srnd", {a(0), a(1)}}},
        {"timestamp",{"timestamp",{}}}};
  }
  // clang-format on

  const IntrinTable &getTable() { return Table; }
};

static bool isStructureReturningFunction(StringRef FunctionName) {
  return llvm::StringSwitch<bool>(FunctionName)
      .Case("addc", true)
      .Case("subb", true)
      .Default(false);
}

// The C++11 "magic static" idiom to lazily initialize the ESIMD intrinsic table
static const IntrinTable &getIntrinTable() {
  static ESIMDIntrinDescTable TheTable;
  return TheTable.getTable();
}

static const ESIMDIntrinDesc &getIntrinDesc(StringRef SrcSpelling) {
  static ESIMDIntrinDesc InvalidDesc{"", {}, {}};
  const auto &Table = getIntrinTable();
  auto It = Table.find(SrcSpelling.str());

  llvm::esimd::assert_and_diag(It != Table.end(),
                               "unknown ESIMD intrinsic: ", SrcSpelling);
  return It->second;
}

static bool isDevicelibFunction(StringRef FunctionName) {
  return llvm::StringSwitch<bool>(FunctionName)
      .Case("__devicelib_ConvertFToBF16INTEL", true)
      .Case("__devicelib_ConvertBF16ToFINTEL", true)
      .Default(false);
}

static std::string mangleFunction(StringRef FunctionName) {
  // Mangle deviceLib function to make it pass through the regular workflow
  // These functions are defined as extern "C" which Demangler that is used
  // fails to handle properly.
  if (isDevicelibFunction(FunctionName)) {
    if (FunctionName.startswith("__devicelib_ConvertFToBF16INTEL")) {
      return (Twine("_Z31") + FunctionName + "RKf").str();
    }
    if (FunctionName.startswith("__devicelib_ConvertBF16ToFINTEL")) {
      return (Twine("_Z31") + FunctionName + "RKt").str();
    }
  }
  // Every inserted vstore gets its own function with the same name,
  // so they are mangled with ".[0-9]+". Just use the
  // raw name to pass through the demangler.
  if (FunctionName.startswith(ESIMD_INSERTED_VSTORE_FUNC_NAME))
    return ESIMD_INSERTED_VSTORE_FUNC_NAME;
  return FunctionName.str();
}

Type *parsePrimitiveTypeString(StringRef TyStr, LLVMContext &Ctx) {
  return llvm::StringSwitch<Type *>(TyStr)
      .Case("bool", IntegerType::getInt1Ty(Ctx))
      .Case("char", IntegerType::getInt8Ty(Ctx))
      .Case("unsigned char", IntegerType::getInt8Ty(Ctx))
      .Case("short", IntegerType::getInt16Ty(Ctx))
      .Case("unsigned short", IntegerType::getInt16Ty(Ctx))
      .Case("int", IntegerType::getInt32Ty(Ctx))
      .Case("unsigned int", IntegerType::getInt32Ty(Ctx))
      .Case("unsigned", IntegerType::getInt32Ty(Ctx))
      .Case("unsigned long long", IntegerType::getInt64Ty(Ctx))
      .Case("long long", IntegerType::getInt64Ty(Ctx))
      .Case("_Float16", IntegerType::getHalfTy(Ctx))
      .Case("float", IntegerType::getFloatTy(Ctx))
      .Case("double", IntegerType::getDoubleTy(Ctx))
      .Case("void", IntegerType::getVoidTy(Ctx))
      .Default(nullptr);
}

template <typename T>
static const T *castNodeImpl(const id::Node *N, id::Node::Kind K) {
  assert(N && N->getKind() == K && "unexpected demangler node kind");
  return reinterpret_cast<const T *>(N);
}

#define castNode(NodeObj, NodeKind)                                            \
  castNodeImpl<id::NodeKind>(NodeObj, id::Node::K##NodeKind)

static APInt parseTemplateArg(id::FunctionEncoding *FE, unsigned int N,
                              Type *&Ty, LLVMContext &Ctx,
                              ESIMDIntrinDesc::GenXArgConversion Conv =
                                  ESIMDIntrinDesc::GenXArgConversion::NONE) {
  // parseTemplateArg returns APInt with a certain bitsize
  // This bitsize (primitive size in bits) is deduced by the following rules:
  // If Conv is not None, then bitsize is taken from Conv
  // If Conv is None and Arg is IntegerLiteral, then bitsize is taken from
  // Arg size
  // If Conv is None and Arg is BoolExpr or Enum, the bitsize falls back to 32

  const auto *Nm = castNode(FE->getName(), NameWithTemplateArgs);
  const auto *ArgsN = castNode(Nm->TemplateArgs, TemplateArgs);
  id::NodeArray Args = ArgsN->getParams();
  assert(N < Args.size() && "too few template arguments");
  std::string_view Val;
  switch (Conv) {
  case ESIMDIntrinDesc::GenXArgConversion::NONE:
    // Default fallback case, if we cannot deduce bitsize
    Ty = IntegerType::getInt32Ty(Ctx);
    break;
  case ESIMDIntrinDesc::GenXArgConversion::TO_I1:
    Ty = IntegerType::getInt1Ty(Ctx);
    break;
  case ESIMDIntrinDesc::GenXArgConversion::TO_I8:
    Ty = IntegerType::getInt8Ty(Ctx);
    break;
  case ESIMDIntrinDesc::GenXArgConversion::TO_I16:
    Ty = IntegerType::getInt16Ty(Ctx);
    break;
  case ESIMDIntrinDesc::GenXArgConversion::TO_I32:
    Ty = IntegerType::getInt32Ty(Ctx);
    break;
  case ESIMDIntrinDesc::GenXArgConversion::TO_I64:
    Ty = IntegerType::getInt64Ty(Ctx);
    break;
  }

  switch (Args[N]->getKind()) {
  case id::Node::KIntegerLiteral: {
    auto *ValL = castNode(Args[N], IntegerLiteral);
    const std::string_view &TyStr = ValL->getType();
    if (Conv == ESIMDIntrinDesc::GenXArgConversion::NONE && TyStr.size() != 0)
      // Overwrite Ty with IntegerLiteral's size
      Ty = parsePrimitiveTypeString(StringRef(&*TyStr.begin(), TyStr.size()),
                                    Ctx);
    Val = ValL->getValue();
    break;
  }
  case id::Node::KBoolExpr: {
    auto *ValL = castNode(Args[N], BoolExpr);
    ValL->match([&Val](bool Value) { Value ? Val = "1" : Val = "0"; });
    break;
  }
  case id::Node::KEnumLiteral: {
    auto *CE = castNode(Args[N], EnumLiteral);
    Val = CE->getIntegerValue();
    break;
  }
  default:
    llvm_unreachable_internal("bad esimd intrinsic template parameter");
  }
  return APInt(Ty->getPrimitiveSizeInBits(),
               StringRef(&*Val.begin(), Val.size()), 10);
}

// Returns the value of the 'ArgIndex' parameter of the template
// function called at 'CI'.
static APInt parseTemplateArg(CallInst &CI, int ArgIndex,
                              ESIMDIntrinDesc::GenXArgConversion Conv =
                                  ESIMDIntrinDesc::GenXArgConversion::NONE) {
  Function *F = CI.getCalledFunction();
  llvm::esimd::assert_and_diag(F, "function to translate is invalid");

  StringRef MnglName = F->getName();
  using Demangler = id::ManglingParser<SimpleAllocator>;
  Demangler Parser(MnglName.begin(), MnglName.end());
  id::Node *AST = Parser.parse();
  llvm::esimd::assert_and_diag(
      AST && Parser.ForwardTemplateRefs.empty(),
      "failed to demangle ESIMD intrinsic: ", MnglName);
  llvm::esimd::assert_and_diag(AST->getKind() == id::Node::KFunctionEncoding,
                               "bad ESIMD intrinsic: ", MnglName);

  auto *FE = static_cast<id::FunctionEncoding *>(AST);
  Type *Ty = nullptr;
  return parseTemplateArg(FE, ArgIndex, Ty, CI.getContext(), Conv);
}

// Constructs a GenX intrinsic name suffix based on the original C++ name (stem)
// and the types of its parameters (some intrinsic names have additional
// suffixes depending on the parameter types).
static std::string getESIMDIntrinSuffix(id::FunctionEncoding *FE,
                                        FunctionType *FT,
                                        const ESIMDIntrinDesc::NameRule &Rule) {
  std::string Suff;
  switch (Rule.Kind) {
  case ESIMDIntrinDesc::GenXSuffixRuleKind::BIN_OP: {
    // e.g. ".add"
    Type *Ty = nullptr;
    APInt OpId = parseTemplateArg(FE, Rule.I.TmplArgNo, Ty, FT->getContext());

    switch (OpId.getSExtValue()) {
    case 0x0:
      Suff = ".add";
      break;
    case 0x1:
      Suff = ".sub";
      break;
    case 0x2:
      Suff = ".inc";
      break;
    case 0x3:
      Suff = ".dec";
      break;
    case 0x4:
      Suff = ".min";
      break;
    case 0x5:
      Suff = ".max";
      break;
    case 0x6:
      Suff = ".xchg";
      break;
    case 0x7:
      Suff = ".cmpxchg";
      break;
    case 0x8:
      Suff = ".and";
      break;
    case 0x9:
      Suff = ".or";
      break;
    case 0xa:
      Suff = ".xor";
      break;
    case 0xb:
      Suff = ".imin";
      break;
    case 0xc:
      Suff = ".imax";
      break;
    case 0x10:
      Suff = ".fmax";
      break;
    case 0x11:
      Suff = ".fmin";
      break;
    case 0x12:
      Suff = ".fcmpwr";
      break;
    case 0x13:
      Suff = ".fadd";
      break;
    case 0x14:
      Suff = ".fsub";
      break;
    case 0xff:
      Suff = ".predec";
      break;
    default:
      llvm_unreachable("unknown atomic OP");
    };
    break;
  }
  case ESIMDIntrinDesc::GenXSuffixRuleKind::NUM_KIND: {
    // e.g. "f"
    int No = Rule.I.CallArgNo;
    Type *Ty = No == -1 ? FT->getReturnType() : FT->getParamType(No);
    if (Ty->isVectorTy())
      Ty = cast<VectorType>(Ty)->getElementType();
    assert(Ty->isFloatingPointTy() || Ty->isIntegerTy());
    Suff = Ty->isFloatingPointTy() ? "f" : "i";
    break;
  }
  default:
    // It's ok if there is no suffix.
    break;
  }

  return Suff;
}

static void translateBlockLoad(CallInst &CI, bool IsSLM) {
  IRBuilder<> Builder(&CI);

  constexpr int AlignmentTemplateArgIdx = 2;
  APInt Val = parseTemplateArg(CI, AlignmentTemplateArgIdx,
                               ESIMDIntrinDesc::GenXArgConversion::TO_I64);
  MaybeAlign Align(Val.getZExtValue());

  auto Op0 = CI.getArgOperand(0);
  auto DataType = CI.getType();
  if (IsSLM) {
    // Convert 'uint32_t' to 'addrspace(3)*' pointer.
    auto PtrType = PointerType::get(DataType, 3);
    Op0 = Builder.CreateIntToPtr(Op0, PtrType);
  }

  auto LI = Builder.CreateAlignedLoad(DataType, Op0, Align, CI.getName());
  LI->setDebugLoc(CI.getDebugLoc());
  CI.replaceAllUsesWith(LI);
}

static void translateBlockStore(CallInst &CI, bool IsSLM) {
  IRBuilder<> Builder(&CI);

  constexpr int AlignmentTemplateArgIdx = 2;
  APInt Val = parseTemplateArg(CI, AlignmentTemplateArgIdx,
                               ESIMDIntrinDesc::GenXArgConversion::TO_I64);
  MaybeAlign Align(Val.getZExtValue());

  auto Op0 = CI.getArgOperand(0);
  auto Op1 = CI.getArgOperand(1);
  if (IsSLM) {
    // Convert 'uint32_t' to 'addrspace(3)*' pointer.
    auto DataType = Op1->getType();
    auto PtrType = PointerType::get(DataType, 3);
    Op0 = Builder.CreateIntToPtr(Op0, PtrType);
  }

  auto SI = Builder.CreateAlignedStore(Op1, Op0, Align);
  SI->setDebugLoc(CI.getDebugLoc());
}

static void translateGatherLoad(CallInst &CI, bool IsSLM) {
  IRBuilder<> Builder(&CI);
  constexpr int AlignmentTemplateArgIdx = 2;
  APInt Val = parseTemplateArg(CI, AlignmentTemplateArgIdx,
                               ESIMDIntrinDesc::GenXArgConversion::TO_I64);
  Align AlignValue(Val.getZExtValue());

  auto OffsetsOp = CI.getArgOperand(0);
  auto MaskOp = CI.getArgOperand(1);
  auto PassThroughOp = CI.getArgOperand(2);
  auto DataType = CI.getType();

  // Convert the mask from <N x i16> to <N x i1>.
  Value *Zero = ConstantInt::get(MaskOp->getType(), 0);
  MaskOp = Builder.CreateICmp(ICmpInst::ICMP_NE, MaskOp, Zero);

  // The address space may be 3-SLM, 1-global or private.
  // At the moment of calling 'gather()' operation the pointer passed to it
  // is already 4-generic. Thus, simply use 4-generic for global and private
  // and let GPU BE deduce the actual address space from the use-def graph.
  unsigned AS = IsSLM ? 3 : 4;
  auto ElemType = DataType->getScalarType();
  auto NumElems = (cast<VectorType>(DataType))->getElementCount();
  auto VPtrType = VectorType::get(PointerType::get(ElemType, AS), NumElems);
  auto VPtrOp = Builder.CreateIntToPtr(OffsetsOp, VPtrType);

  auto LI = Builder.CreateMaskedGather(DataType, VPtrOp, AlignValue, MaskOp,
                                       PassThroughOp);
  LI->setDebugLoc(CI.getDebugLoc());
  CI.replaceAllUsesWith(LI);
}

// TODO Specify document behavior for slm_init and nbarrier_init when:
// 1) they are called not from kernels
// 2) there are multiple such calls reachable from a kernel
// 3) when a call in external function linked by the Back-End

// This function sets/updates VCNamedBarrierCount attribute to the kernels
// calling this intrinsic initializing the number of named barriers.
static void translateNbarrierInit(CallInst &CI) {
  auto F = CI.getFunction();
  auto *ArgV = CI.getArgOperand(0);
  llvm::esimd::assert_and_diag(
      isa<ConstantInt>(ArgV), __FILE__,
      " integral constant is expected for named barrier count");

  auto NewVal = cast<llvm::ConstantInt>(ArgV)->getZExtValue();
  assert(NewVal != 0 && "zero named barrier count being requested");
  esimd::UpdateUint64MetaDataToMaxValue SetMaxNBarrierCnt{
      *F->getParent(), genx::KernelMDOp::NBarrierCnt, NewVal};
  // TODO: Keep track of traversed functions to avoid repeating traversals
  // over same function.
  sycl::utils::traverseCallgraphUp(F, SetMaxNBarrierCnt);
}

static void translatePackMask(CallInst &CI) {
  APInt Val = parseTemplateArg(CI, 0);
  unsigned N = Val.getZExtValue();
  Value *Result = CI.getArgOperand(0);
  assert(Result->getType()->isIntOrIntVectorTy());
  Value *Zero = ConstantInt::get(Result->getType(), 0);
  IRBuilder<> Builder(&CI);
  llvm::LLVMContext &Context = CI.getContext();
  // TODO CM_COMPAT
  // In CM non LSB bits in mask elements are ignored, so e.g. '2' is treated as
  // 'false' there. ESIMD adopts C++ semantics, where any non-zero is 'true'.
  // For CM this ICmpInst should be replaced with truncation to i1.
  Result = Builder.CreateICmp(ICmpInst::ICMP_NE, Result, Zero);
  Result = Builder.CreateBitCast(Result, llvm::Type::getIntNTy(Context, N));

  if (N != 32) {
    Result = Builder.CreateCast(llvm::Instruction::ZExt, Result,
                                llvm::Type::getInt32Ty(Context));
  }
  Result->setName(CI.getName());
  cast<llvm::Instruction>(Result)->setDebugLoc(CI.getDebugLoc());
  CI.replaceAllUsesWith(Result);
}

static void translateUnPackMask(CallInst &CI) {
  APInt Val = parseTemplateArg(CI, 0);
  unsigned N = Val.getZExtValue();
  // get N x i1
  assert(CI.arg_size() == 1);
  llvm::Value *Arg0 = CI.getArgOperand(0);
  unsigned Width = Arg0->getType()->getPrimitiveSizeInBits();
  IRBuilder<> Builder(&CI);
  llvm::LLVMContext &Context = CI.getContext();
  if (Width > N) {
    llvm::Type *Ty = llvm::IntegerType::get(Context, N);
    Arg0 = Builder.CreateTrunc(Arg0, Ty);
    if (auto *Trunc = dyn_cast<llvm::Instruction>(Arg0))
      Trunc->setDebugLoc(CI.getDebugLoc());
  }
  assert(Arg0->getType()->getPrimitiveSizeInBits() == N);
  Arg0 = Builder.CreateBitCast(
      Arg0, llvm::FixedVectorType::get(llvm::Type::getInt1Ty(Context), N));

  // get N x i16
  llvm::Value *TransCI = Builder.CreateZExt(
      Arg0, llvm::FixedVectorType::get(llvm::Type::getInt16Ty(Context), N));
  TransCI->takeName(&CI);
  if (llvm::Instruction *TransCInst = dyn_cast<llvm::Instruction>(TransCI))
    TransCInst->setDebugLoc(CI.getDebugLoc());
  CI.replaceAllUsesWith(TransCI);
}

static bool translateVLoad(CallInst &CI, SmallPtrSetImpl<Type *> &GVTS) {
  if (GVTS.find(CI.getType()) != GVTS.end())
    return false;
  IRBuilder<> Builder(&CI);
  auto LI = Builder.CreateLoad(CI.getType(), CI.getArgOperand(0), CI.getName());
  LI->setDebugLoc(CI.getDebugLoc());
  CI.replaceAllUsesWith(LI);
  return true;
}

static bool translateVStore(CallInst &CI, SmallPtrSetImpl<Type *> &GVTS) {
  if (GVTS.find(CI.getOperand(1)->getType()) != GVTS.end())
    return false;
  IRBuilder<> Builder(&CI);
  auto SI = Builder.CreateStore(CI.getArgOperand(1), CI.getArgOperand(0));
  SI->setDebugLoc(CI.getDebugLoc());
  return true;
}

// Newly created GenX intrinsic might have different return type than expected.
// This helper function creates cast operation from GenX intrinsic return type
// to currently expected. Returns pointer to created cast instruction if it
// was created, otherwise returns NewI.
static Instruction *addCastInstIfNeeded(Instruction *OldI, Instruction *NewI,
                                        Type *UseType = nullptr) {
  Type *NITy = NewI->getType();
  Type *OITy = UseType ? UseType : OldI->getType();
  if (OITy != NITy) {
    auto CastOpcode = CastInst::getCastOpcode(NewI, false, OITy, false);
    NewI = CastInst::Create(CastOpcode, NewI, OITy,
                            NewI->getName() + ".cast.ty", OldI);
    NewI->setDebugLoc(OldI->getDebugLoc());
  }
  return NewI;
}

// Translates the following intrinsics:
//   %res = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
//   %res = call double @llvm.fmuladd.f64(double %a, double %b, double %c)
// To
//   %mul = fmul <type> %a, <type> %b
//   %res = fadd <type> %mul, <type> %c
void translateFmuladd(CallInst *CI) {
  assert(CI->getIntrinsicID() == Intrinsic::fmuladd);
  IRBuilder<> Bld(CI);
  auto *Mul = Bld.CreateFMul(CI->getOperand(0), CI->getOperand(1));
  auto *Res = Bld.CreateFAdd(Mul, CI->getOperand(2));
  CI->replaceAllUsesWith(Res);
}

// Translates an LLVM intrinsic to a form, digestable by the BE.
bool translateLLVMIntrinsic(CallInst *CI) {
  Function *F = CI->getCalledFunction();
  llvm::esimd::assert_and_diag(F && F->isIntrinsic(),
                               "malformed llvm intrinsic call");

  switch (F->getIntrinsicID()) {
  case Intrinsic::assume:
    // no translation - it will be simply removed.
    // TODO: make use of 'assume' info in the BE
    break;
  case Intrinsic::fmuladd:
    translateFmuladd(CI);
    break;
  default:
    return false; // "intrinsic wasn't translated, keep the original call"
  }
  return true; // "intrinsic has been translated, erase the original call"
}

/// Replaces the load \p LI of SPIRV global with a compile time known constant
/// when possible. The replaced instructions are stored into the given container
/// \p InstsToErase.
static void
translateSpirvGlobalUses(LoadInst *LI, StringRef SpirvGlobalName,
                         SmallVectorImpl<Instruction *> &InstsToErase) {
  Value *NewInst = nullptr;
  if (SpirvGlobalName == "SubgroupLocalInvocationId") {
    NewInst = llvm::Constant::getNullValue(LI->getType());
  } else if (SpirvGlobalName == "SubgroupSize" ||
             SpirvGlobalName == "SubgroupMaxSize") {
    NewInst = llvm::Constant::getIntegerValue(LI->getType(),
                                              llvm::APInt(32, 1, true));
  }
  if (NewInst) {
    LI->replaceAllUsesWith(NewInst);
    InstsToErase.push_back(LI);
  }
}

static void createESIMDIntrinsicArgs(const ESIMDIntrinDesc &Desc,
                                     SmallVector<Value *, 16> &GenXArgs,
                                     CallInst &CI, id::FunctionEncoding *FE) {
  uint32_t LastCppArgNo = 0; // to implement SRC_CALL_ALL

  for (unsigned int I = 0; I < Desc.ArgRules.size(); ++I) {
    const ESIMDIntrinDesc::ArgRule &Rule = Desc.ArgRules[I];

    switch (Rule.Kind) {
    case ESIMDIntrinDesc::GenXArgRuleKind::SRC_CALL_ARG: {
      Value *Arg = CI.getArgOperand(Rule.I.Arg.CallArgNo);

      switch (Rule.I.Arg.Conv) {
      case ESIMDIntrinDesc::GenXArgConversion::NONE:
        GenXArgs.push_back(Arg);
        break;
      case ESIMDIntrinDesc::GenXArgConversion::TO_I1: {
        // convert N-bit integer to 1-bit integer
        Type *NTy = Arg->getType();
        assert(NTy->isIntOrIntVectorTy());
        Value *Zero = ConstantInt::get(NTy, 0);
        IRBuilder<> Bld(&CI);
        auto *Cmp = Bld.CreateICmp(ICmpInst::ICMP_NE, Arg, Zero);
        GenXArgs.push_back(Cmp);
        break;
      }
      default:
        llvm_unreachable("Unknown ESIMD arg conversion");
      }
      LastCppArgNo = Rule.I.Arg.CallArgNo;
      break;
    }
    case ESIMDIntrinDesc::GenXArgRuleKind::SRC_CALL_ALL:
      assert(LastCppArgNo < CI.arg_size());
      for (uint32_t N = LastCppArgNo; N < CI.arg_size(); ++N)
        GenXArgs.push_back(CI.getArgOperand(N));
      break;
    case ESIMDIntrinDesc::GenXArgRuleKind::SRC_TMPL_ARG: {
      Type *Ty = nullptr;
      APInt Val = parseTemplateArg(FE, Rule.I.Arg.CallArgNo, Ty,
                                   CI.getContext(), Rule.I.Arg.Conv);
      Value *ArgVal = ConstantInt::get(
          Ty, static_cast<uint64_t>(Val.getSExtValue()), true /*signed*/);
      GenXArgs.push_back(ArgVal);
      break;
    }
    case ESIMDIntrinDesc::GenXArgRuleKind::UNDEF: {
      Type *Ty = Rule.I.Arg.CallArgNo == -1
                     ? CI.getType()
                     : CI.getArgOperand(Rule.I.Arg.CallArgNo)->getType();
      GenXArgs.push_back(UndefValue::get(Ty));
      break;
    }
    case ESIMDIntrinDesc::GenXArgRuleKind::CONST_INT8: {
      auto Ty = IntegerType::getInt8Ty(CI.getContext());
      GenXArgs.push_back(llvm::ConstantInt::get(Ty, Rule.I.ArgConst));
      break;
    }
    case ESIMDIntrinDesc::GenXArgRuleKind::CONST_INT16: {
      auto Ty = IntegerType::getInt16Ty(CI.getContext());
      GenXArgs.push_back(llvm::ConstantInt::get(Ty, Rule.I.ArgConst));
      break;
    }
    case ESIMDIntrinDesc::GenXArgRuleKind::CONST_INT32: {
      auto Ty = IntegerType::getInt32Ty(CI.getContext());
      GenXArgs.push_back(llvm::ConstantInt::get(Ty, Rule.I.ArgConst));
      break;
    }
    case ESIMDIntrinDesc::GenXArgRuleKind::CONST_INT64: {
      auto Ty = IntegerType::getInt64Ty(CI.getContext());
      GenXArgs.push_back(llvm::ConstantInt::get(Ty, Rule.I.ArgConst));
      break;
    }
    }
  }
}

// Create a spirv function declaration
// This is used for lowering devicelib functions.
// The function
// 1. Generates spirv function definition
// 2. Converts passed by reference argument of devicelib function into passed by
// value argument of spirv functions
// 3. Assigns proper attributes to generated function
static Function *
createDeviceLibESIMDDeclaration(const ESIMDIntrinDesc &Desc,
                                SmallVector<Value *, 16> &GenXArgs,
                                CallInst &CI) {
  SmallVector<Type *, 16> ArgTypes;
  IRBuilder<> Bld(&CI);
  for (unsigned i = 0; i < GenXArgs.size(); ++i) {
    Type *NTy = llvm::StringSwitch<Type *>(Desc.GenXSpelling)
                    .Case("__spirv_ConvertFToBF16INTEL",
                          Type::getFloatTy(CI.getContext()))
                    .Case("__spirv_ConvertBF16ToFINTEL",
                          Type::getInt16Ty(CI.getContext()))
                    .Default(nullptr);

    auto LI = Bld.CreateLoad(NTy, GenXArgs[i]);
    GenXArgs[i] = LI;
    ArgTypes.push_back(NTy);
  }
  auto *FType = FunctionType::get(CI.getType(), ArgTypes, false);
  Function *F = CI.getModule()->getFunction(Desc.GenXSpelling);
  if (!F) {
    F = Function::Create(FType, GlobalVariable::ExternalLinkage,
                         Desc.GenXSpelling, CI.getModule());
    F->addFnAttr(Attribute::NoUnwind);
    F->addFnAttr(Attribute::Convergent);
    F->setDSOLocal(true);

    F->setCallingConv(CallingConv::SPIR_FUNC);
  }

  return F;
}

// Create a simple function declaration
// This is used for testing purposes, when it is impossible to query
// vc-intrinsics
static Function *createTestESIMDDeclaration(const ESIMDIntrinDesc &Desc,
                                            SmallVector<Value *, 16> &GenXArgs,
                                            CallInst &CI) {
  SmallVector<Type *, 16> ArgTypes;
  for (unsigned i = 0; i < GenXArgs.size(); ++i)
    ArgTypes.push_back(GenXArgs[i]->getType());
  auto *FType = FunctionType::get(CI.getType(), ArgTypes, false);
  auto Name = GenXIntrinsic::getGenXIntrinsicPrefix() + Desc.GenXSpelling;
  return Function::Create(FType, GlobalVariable::ExternalLinkage, Name,
                          CI.getModule());
}

// Demangles and translates given ESIMD intrinsic call instruction. Example
//
// ### Source-level intrinsic:
//
// sycl::_V1::ext::intel::experimental::esimd::__vector_type<int, 16>::type
// __esimd_flat_read<int, 16>(
//     sycl::_V1::ext::intel::experimental::esimd::__vector_type<unsigned long
//     long, 16>::type,
//     sycl::_V1::ext::intel::experimental::esimd::__vector_type<int, 16>::type)
//
// ### Itanium-mangled name:
//
// _Z14__esimd_flat_readIiLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeENS2_IyXT0_EE4typeES5_
//
// ### Itanium demangler IR:
//
// FunctionEncoding(
//  NestedName(
//    NameWithTemplateArgs(
//      NestedName(
//        NestedName(
//          NameType("cm"),
//          NameType("gen")),
//        NameType("__vector_type")),
//      TemplateArgs(
//        {NameType("int"),
//         IntegerLiteral("", "16")})),
//    NameType("type")),
//  NameWithTemplateArgs(
//    NameType("__esimd_flat_read"),
//    TemplateArgs(
//      {NameType("int"),
//       IntegerLiteral("", "16")})),
//  {NestedName(
//     NameWithTemplateArgs(
//       NestedName(
//         NestedName(
//           NameType("cm"),
//           NameType("gen")),
//         NameType("__vector_type")),
//       TemplateArgs(
//         {NameType("unsigned long long"),
//          IntegerLiteral("", "16")})),
//     NameType("type")),
//   NestedName(
//     NameWithTemplateArgs(
//       NestedName(
//         NestedName(
//           NameType("cm"),
//           NameType("gen")),
//         NameType("__vector_type")),
//       TemplateArgs(
//         {NameType("int"),
//          IntegerLiteral("", "16")})),
//     NameType("type"))},
//  <null>,
//  QualNone, FunctionRefQual::FrefQualNone)
//
static void translateESIMDIntrinsicCall(CallInst &CI) {
  using Demangler = id::ManglingParser<SimpleAllocator>;
  Function *F = CI.getCalledFunction();
  llvm::esimd::assert_and_diag(F, "function to translate is invalid");
  std::string MnglNameStr = mangleFunction(F->getName());
  StringRef MnglName = MnglNameStr;

  Demangler Parser(MnglName.begin(), MnglName.end());
  id::Node *AST = Parser.parse();

  llvm::esimd::assert_and_diag(
      AST && Parser.ForwardTemplateRefs.empty(),
      "failed to demangle ESIMD intrinsic: ", MnglName);
  llvm::esimd::assert_and_diag(AST->getKind() == id::Node::KFunctionEncoding,
                               "bad ESIMD intrinsic: ", MnglName);

  auto *FE = static_cast<id::FunctionEncoding *>(AST);
  std::string_view BaseNameV = FE->getName()->getBaseName();

  auto PrefLen = isDevicelibFunction(F->getName())
                     ? 0
                     : StringRef(ESIMD_INTRIN_PREF1).size();
  StringRef BaseName(&*BaseNameV.begin() + PrefLen, BaseNameV.size() - PrefLen);
  const auto &Desc = getIntrinDesc(BaseName);
  if (!Desc.isValid()) // TODO remove this once all intrinsics are supported
    return;

  auto *FTy = F->getFunctionType();
  std::string Suffix = getESIMDIntrinSuffix(FE, FTy, Desc.SuffixRule);
  SmallVector<Value *, 16> GenXArgs;
  createESIMDIntrinsicArgs(Desc, GenXArgs, CI, FE);
  Function *NewFDecl = nullptr;
  bool DoesFunctionReturnStructure =
      isStructureReturningFunction(Desc.GenXSpelling);
  if (isDevicelibFunction(F->getName())) {
    NewFDecl = createDeviceLibESIMDDeclaration(Desc, GenXArgs, CI);
  } else if (Desc.GenXSpelling.rfind("test.src.", 0) == 0) {
    // Special case for testing purposes
    NewFDecl = createTestESIMDDeclaration(Desc, GenXArgs, CI);
  } else {
    auto ID = GenXIntrinsic::lookupGenXIntrinsicID(
        GenXIntrinsic::getGenXIntrinsicPrefix() + Desc.GenXSpelling + Suffix);

    SmallVector<Type *, 16> GenXOverloadedTypes;
    if (GenXIntrinsic::isOverloadedRet(ID)) {
      if (DoesFunctionReturnStructure) {
        // TODO implement more generic handling of returned structure
        // current code assumes that returned code has 2 members of the
        // same type as arguments.
        GenXOverloadedTypes.push_back(GenXArgs[1]->getType());
        GenXOverloadedTypes.push_back(GenXArgs[1]->getType());
      } else {
        GenXOverloadedTypes.push_back(CI.getType());
      }
    }
    for (unsigned i = 0; i < GenXArgs.size(); ++i)
      if (GenXIntrinsic::isOverloadedArg(ID, i))
        GenXOverloadedTypes.push_back(GenXArgs[i]->getType());

    NewFDecl = GenXIntrinsic::getGenXDeclaration(CI.getModule(), ID,
                                                 GenXOverloadedTypes);
  }

  // llvm::Attribute::ReadNone must not be used for call statements anymore.
  Instruction *NewInst = nullptr;
  AddrSpaceCastInst *CastInstruction = nullptr;
  if (DoesFunctionReturnStructure) {
    llvm::esimd::assert_and_diag(
        isa<AddrSpaceCastInst>(GenXArgs[0]),
        "Unexpected instruction for returning a structure from a function.");
    CastInstruction = static_cast<AddrSpaceCastInst *>(GenXArgs[0]);
    // Remove 1st argument that is used to return the structure
    GenXArgs.erase(GenXArgs.begin());
  }

  CallInst *NewCI = IntrinsicInst::Create(
      NewFDecl, GenXArgs,
      NewFDecl->getReturnType()->isVoidTy() ? "" : CI.getName() + ".esimd",
      &CI);
  NewCI->setDebugLoc(CI.getDebugLoc());
  if (DoesFunctionReturnStructure) {
    IRBuilder<> Builder(&CI);

    NewInst = Builder.CreateStore(
        NewCI, Builder.CreateBitCast(CastInstruction->getPointerOperand(),
                                     NewCI->getType()->getPointerTo()));
  } else {
    NewInst = addCastInstIfNeeded(&CI, NewCI);
  }

  CI.replaceAllUsesWith(NewInst);
  CI.eraseFromParent();
}

static std::string getMDString(MDNode *N, unsigned I) {
  if (!N)
    return "";

  Metadata *Op = N->getOperand(I);
  if (!Op)
    return "";

  if (MDString *Str = dyn_cast<MDString>(Op)) {
    return Str->getString().str();
  }

  return "";
}

void generateKernelMetadata(Module &M) {
  if (M.getNamedMetadata(esimd::GENX_KERNEL_METADATA))
    return;

  auto Kernels = M.getOrInsertNamedMetadata(esimd::GENX_KERNEL_METADATA);
  assert(Kernels->getNumOperands() == 0 && "metadata out of sync");

  LLVMContext &Ctx = M.getContext();
  Type *I32Ty = Type::getInt32Ty(Ctx);

  std::string TargetTriple = M.getTargetTriple();
  llvm::Triple T(TargetTriple);
  T.setArchName("genx64");
  TargetTriple = T.str();
  M.setTargetTriple(TargetTriple);

  enum { AK_NORMAL, AK_SAMPLER, AK_SURFACE, AK_VME };
  enum { IK_NORMAL, IK_INPUT, IK_OUTPUT, IK_INPUT_OUTPUT };

  for (auto &F : M.functions()) {
    // Skip non-SIMD kernels.
    if (!esimd::isESIMDKernel(F))
      continue;

    // Metadata node containing N i32s, where N is the number of kernel
    // arguments, and each i32 is the kind of argument,  one of:
    //     0 = general, 1 = sampler, 2 = surface, 3 = vme
    // (the same values as in the "kind" field of an "input_info" record in a
    // vISA kernel.
    SmallVector<Metadata *, 8> ArgKinds;

    // Optional, not supported for compute
    SmallVector<Metadata *, 8> ArgInOutKinds;

    // Metadata node describing N strings where N is the number of kernel
    // arguments, each string describing argument type in OpenCL.
    // required for running on top of OpenCL runtime.
    SmallVector<Metadata *, 8> ArgTypeDescs;

    auto *KernelArgTypes = F.getMetadata("kernel_arg_type");
    auto *KernelArgAccPtrs = F.getMetadata("kernel_arg_accessor_ptr");
    unsigned Idx = 0;

    auto getMD = [](Value *Val) { return esimd::getMetadata(Val); };

    // Iterate argument list to gather argument kinds and generate argument
    // descriptors.
    for (const Argument &Arg : F.args()) {
      int Kind = AK_NORMAL;
      int IKind = IK_NORMAL;

      auto ArgType = getMDString(KernelArgTypes, Idx);

      if (ArgType.find("image1d_t") != std::string::npos ||
          ArgType.find("image2d_t") != std::string::npos ||
          ArgType.find("image3d_t") != std::string::npos) {
        Kind = AK_SURFACE;
        ArgTypeDescs.push_back(MDString::get(Ctx, ArgType));
      } else {
        StringRef ArgDesc = "";

        if (Arg.getType()->isPointerTy()) {
          bool IsAcc = false;
          bool IsLocalAcc = false;

          if (KernelArgAccPtrs) {
            auto *AccMD =
                cast<ConstantAsMetadata>(KernelArgAccPtrs->getOperand(Idx));
            auto AccMDVal = cast<ConstantInt>(AccMD->getValue())->getValue();
            IsAcc = static_cast<unsigned>(AccMDVal.getZExtValue());

            constexpr unsigned LocalAS{3};
            IsLocalAcc =
                IsAcc &&
                cast<PointerType>(Arg.getType())->getAddressSpace() == LocalAS;
          }

          if (IsLocalAcc) {
            // Local accessor doesn't need any changes.
          } else if (IsAcc && !ForceStatelessMem) {
            ArgDesc = "buffer_t";
            Kind = AK_SURFACE;
          } else
            ArgDesc = "svmptr_t";
        }
        ArgTypeDescs.push_back(MDString::get(Ctx, ArgDesc));
      }

      ArgKinds.push_back(getMD(ConstantInt::get(I32Ty, Kind)));
      ArgInOutKinds.push_back(getMD(ConstantInt::get(I32Ty, IKind)));

      Idx++;
    }

    MDNode *Kinds = MDNode::get(Ctx, ArgKinds);
    MDNode *IOKinds = MDNode::get(Ctx, ArgInOutKinds);
    MDNode *ArgDescs = MDNode::get(Ctx, ArgTypeDescs);

    Metadata *MDArgs[] = {
        getMD(&F),
        MDString::get(Ctx, F.getName().str()),
        Kinds,
        getMD(llvm::ConstantInt::getNullValue(I32Ty)), // SLM size in bytes
        getMD(llvm::ConstantInt::getNullValue(I32Ty)), // arg offsets
        IOKinds,
        ArgDescs,
        getMD(llvm::ConstantInt::getNullValue(I32Ty)), // named barrier count
        getMD(llvm::ConstantInt::getNullValue(I32Ty))  // regular barrier count
    };

    // Add this kernel to the root.
    Kernels->addOperand(MDNode::get(Ctx, MDArgs));
    F.addFnAttr("oclrt", "1");
    F.addFnAttr("CMGenxMain");
  }
}

// collect all the vector-types that are used by genx-volatiles
// TODO: can we make the Module argument `const`?
SmallPtrSet<Type *, 4> collectGenXVolatileTypes(Module &M) {
  SmallPtrSet<Type *, 4> GenXVolatileTypeSet;
  for (auto &G : M.globals()) {
    if (!G.hasAttribute("genx_volatile"))
      continue;
    auto GTy = dyn_cast<StructType>(G.getValueType());
    // TODO FIXME relying on type name in LLVM IR is fragile, needs rework
    if (!GTy || !GTy->getName()
                     .rtrim(".0123456789")
                     .endswith("sycl::_V1::ext::intel::esimd::simd"))
      continue;
    assert(GTy->getNumContainedTypes() == 1);
    auto VTy = GTy->getContainedType(0);
    if ((GTy = dyn_cast<StructType>(VTy))) {
      assert(
          GTy->getName()
              .rtrim(".0123456789")
              .endswith("sycl::_V1::ext::intel::esimd::detail::simd_obj_impl"));
      VTy = GTy->getContainedType(0);
    }
    assert(VTy->isVectorTy());
    GenXVolatileTypeSet.insert(VTy);
  }
  return GenXVolatileTypeSet;
}

// genx_volatile variables are special and require vstores instead of stores.
// In most cases, the vstores are called directly in the implementation
// of the simd object operations, but in some cases clang can implicitly
// insert stores, such as after a write in inline assembly. To handle that
// case, lower any stores of genx_volatiles into vstores.
void lowerGlobalStores(Module &M, const SmallPtrSetImpl<Type *> &GVTS) {
  SmallVector<Instruction *, 4> ToErase;
  for (auto &F : M.functions()) {
    for (Instruction &I : instructions(F)) {
      auto SI = dyn_cast_or_null<StoreInst>(&I);
      if (!SI)
        continue;
      if (GVTS.find(SI->getValueOperand()->getType()) == GVTS.end())
        continue;
      SmallVector<Type *, 2> ArgTypes;
      IRBuilder<> Builder(SI);
      ArgTypes.push_back(SI->getPointerOperand()->getType());
      ArgTypes.push_back(SI->getValueOperand()->getType());
      auto *NewFType = FunctionType::get(SI->getType(), ArgTypes, false);
      auto *NewF =
          Function::Create(NewFType, GlobalVariable::ExternalWeakLinkage,
                           ESIMD_INSERTED_VSTORE_FUNC_NAME, M);
      NewF->addFnAttr(Attribute::NoUnwind);
      NewF->addFnAttr(Attribute::Convergent);
      NewF->setDSOLocal(true);
      NewF->setCallingConv(CallingConv::SPIR_FUNC);
      SmallVector<Value *, 2> Args;
      Args.push_back(SI->getPointerOperand());
      Args.push_back(SI->getValueOperand());
      auto *NewCI = Builder.CreateCall(NewFType, NewF, Args);
      NewCI->setDebugLoc(SI->getDebugLoc());
      ToErase.push_back(SI);
    }
  }
  for (auto *Inst : ToErase) {
    Inst->eraseFromParent();
  }
}

// Change in global variables:
//
// Old IR:
// ======
// @vc = global %"class.cm::gen::simd"
//          zeroinitializer, align 64 #0
//
// % call.cm.i.i = tail call<16 x i32> @llvm.genx.vload.v16i32.p4v16i32(
//    <16 x i32> addrspace(4) * getelementptr(
//    % "class.cm::gen::simd",
//    % "class.cm::gen::simd" addrspace(4) *
//    addrspacecast(% "class.cm::gen::simd" * @vc to
//    % "class.cm::gen::simd" addrspace(4) *), i64 0,
//    i32 0))
//
// New IR:
// ======
//
// @0 = dso_local global <16 x i32> zeroinitializer, align 64 #0 <-- New Global
// Variable
//
// % call.cm.i.i = tail call<16 x i32> @llvm.genx.vload.v16i32.p4v16i32(
//        <16 x i32> addrspace(4) * getelementptr(
//        % "class.cm::gen::simd",
//        % "class.cm::gen::simd" addrspace(4) *
//        addrspacecast(% "class.cm::gen::simd" *
//        bitcast(<16 x i32> * @0 to
//        %"class.cm::gen::simd" *) to %
//        "class.cm::gen::simd" addrspace(4) *),
//        i64 0, i32 0))
void lowerGlobalsToVector(Module &M) {
  // Create new global variables of type vector* type
  // when old one is of simd* type.
  DenseMap<GlobalVariable *, GlobalVariable *> OldNewGlobal;
  for (auto &G : M.globals()) {
    Type *GVTy = G.getValueType();
    Type *NewTy = esimd::getVectorTyOrNull(dyn_cast<StructType>(GVTy));
    if (NewTy && !G.user_empty()) {
      auto InitVal =
          G.hasInitializer() && isa<UndefValue>(G.getInitializer())
              ? static_cast<ConstantData *>(UndefValue::get(NewTy))
              : static_cast<ConstantData *>(ConstantAggregateZero::get(NewTy));
      auto NewGlobalVar =
          new GlobalVariable(NewTy, G.isConstant(), G.getLinkage(), InitVal, "",
                             G.getThreadLocalMode(), G.getAddressSpace());
      NewGlobalVar->setExternallyInitialized(G.isExternallyInitialized());
      NewGlobalVar->setVisibility(G.getVisibility());
      NewGlobalVar->copyAttributesFrom(&G);
      NewGlobalVar->takeName(&G);
      NewGlobalVar->copyMetadata(&G, 0);
      M.insertGlobalVariable(NewGlobalVar);
      OldNewGlobal.insert(std::make_pair(&G, NewGlobalVar));
    }
  }

  // Remove old global variables from the program.
  for (auto &G : OldNewGlobal) {
    auto OldGlob = G.first;
    auto NewGlobal = G.second;
    OldGlob->replaceAllUsesWith(
        ConstantExpr::getBitCast(NewGlobal, OldGlob->getType()));
    OldGlob->eraseFromParent();
  }
}

} // namespace

bool SYCLLowerESIMDPass::prepareForAlwaysInliner(Module &M) {

  auto markAlwaysInlined = [](Function &F) -> bool {
    if (F.hasFnAttribute(llvm::Attribute::NoInline))
      F.removeFnAttr(llvm::Attribute::NoInline);
    if (F.hasFnAttribute(llvm::Attribute::InlineHint))
      F.removeFnAttr(llvm::Attribute::InlineHint);
    F.addFnAttr(llvm::Attribute::AlwaysInline);
    return true;
  };
  auto markNoInline = [](Function &F) {
    if (F.hasFnAttribute(llvm::Attribute::AlwaysInline))
      F.removeFnAttr(llvm::Attribute::AlwaysInline);
    if (F.hasFnAttribute(llvm::Attribute::InlineHint))
      F.removeFnAttr(llvm::Attribute::InlineHint);
    F.addFnAttr(llvm::Attribute::NoInline);
  };

  bool ModuleContainsGenXVolatile =
      std::any_of(M.global_begin(), M.global_end(), [](const auto &Global) {
        return Global.hasAttribute("genx_volatile");
      });

  auto requiresInlining = [=](Function &F) {
    // If there are any genx_volatile globals in the module, inline
    // noinline functions because load/store semantics are not valid for
    // these globals and we cannot know for sure if the load/store target
    // is one of these globals without inlining.
    if (ModuleContainsGenXVolatile)
      return true;

    // Otherwise, only inline esimd namespace functions.
    StringRef MangledName = F.getName();
    id::ManglingParser<SimpleAllocator> Parser(MangledName.begin(),
                                               MangledName.end());
    id::Node *AST = Parser.parse();
    if (!AST || AST->getKind() != id::Node::KFunctionEncoding)
      return false;

    auto *FE = static_cast<id::FunctionEncoding *>(AST);
    const id::Node *NameNode = FE->getName();
    if (!NameNode)
      return false;

    if (NameNode->getKind() == id::Node::KLocalName)
      return false;

    id::OutputBuffer NameBuf;
    NameNode->print(NameBuf);
    StringRef Name(NameBuf.getBuffer(), NameBuf.getCurrentPosition());

    return Name.starts_with("sycl::_V1::ext::intel::esimd::") ||
           Name.starts_with("sycl::_V1::ext::intel::experimental::esimd::");
  };
  bool NeedInline = false;
  for (auto &F : M) {
    // If some function already has 'alwaysinline' attribute, then request
    // inliner pass.
    if (F.hasFnAttribute(Attribute::AlwaysInline)) {
      NeedInline = true;
      continue;
    }

    // VC BE forbids 'alwaysinline' and "VCStackCall" on the same function.
    // Such function may be used in other module, we cannot remove it
    // after inlining.
    if (F.hasFnAttribute(llvm::genx::VCFunctionMD::VCStackCall))
      continue;

    if (isESIMDKernel(F))
      continue;

    if (isSlmAllocatorConstructor(F) || isSlmAllocatorDestructor(F)) {
      // slm_allocator constructor and destructor must be inlined
      // to help SLM reservation analysis.
      NeedInline |= markAlwaysInlined(F);
      continue;
    }

    if (isAssertFail(F)) {
      markNoInline(F);
      continue;
    }

    // TODO: The next code and comment was placed to ESIMDLoweringPass
    // 2 years ago, when GPU VC BE did not support function calls and
    // required everything to be inlined right into the kernel unless
    // it had noinline or VCStackCall attrubute.
    // This code migrated to here without changes, but... VC BE does support
    //  the calls of spir_func these days, so this code needs re-visiting.
    if (!F.hasFnAttribute(Attribute::NoInline) || requiresInlining(F))
      NeedInline |= markAlwaysInlined(F);

    if (!isSlmInit(F))
      continue;

    for (User *U : F.users()) {
      auto *FCall = dyn_cast<CallInst>(U);
      if (FCall && FCall->getCalledFunction() == &F) {
        Function *GenF = FCall->getFunction();
        // The original kernel (UserK) if often automatically separated into
        // a spir_func (GenF) that is then called from spir_kernel (GenK).
        // When that happens, the calls of slm_init<N>() originally placed
        // in 'UserK' get moved to spir_func 'GenF', which creates wrong IR
        // because slm_init() must be called only from a kernel.
        // Fix it here: If 'GenF' has only 1 caller spir_kernel 'GenK',
        // then inline 'GenF' to move slm_init call from spir_kernel 'GenK'.
        SmallPtrSet<Function *, 1> GenFCallers;
        for (User *GenFU : GenF->users()) {
          auto *GenFCall = dyn_cast<CallInst>(GenFU);
          if (GenFCall && GenFCall->getCalledFunction() == GenF) {
            Function *GenK = GenFCall->getFunction();
            GenFCallers.insert(GenK);
          } else {
            // Unexpected user of GenF. Do not require GenF inlining.
            GenFCallers.clear();
            break;
          }
        } // end for (User *GenFU : GenF->users())
        if (GenFCallers.size() == 1 && isESIMDKernel(**GenFCallers.begin()))
          NeedInline |= markAlwaysInlined(*GenF);
      }
    } // end for (User *U : F.users())
  }
  return NeedInline;
}

/// Remove the attribute \p Attr from the given function \p F.
/// Adds the memory effect \p Memef to the calls \p F.
static void fixFunctionAttribute(Function &F, Attribute::AttrKind Attr,
                                 MemoryEffects MemEf) {
  if (!F.getFnAttribute(Attr).isValid())
    return;

  for (auto &U : F.uses()) {
    if (auto *Call = dyn_cast<CallInst>(&*U))
      Call->setMemoryEffects(MemEf);
  }
  F.removeFnAttr(Attr);
}

/// Replaces the function attributes ReadNone/ReadOnly/WriteOnly
/// with the corresponding memory effects on function calls.
static void fixFunctionReadWriteAttributes(Module &M) {
  // llvm::Attribute::ReadNone/ReadOnly/WriteOnly
  // must not be used for call statements anymore.
  for (auto &&F : M) {
    fixFunctionAttribute(F, llvm::Attribute::ReadNone,
                         llvm::MemoryEffects::none());
    fixFunctionAttribute(F, llvm::Attribute::ReadOnly,
                         llvm::MemoryEffects::readOnly());
    fixFunctionAttribute(F, llvm::Attribute::WriteOnly,
                         llvm::MemoryEffects::writeOnly());
  }
}

PreservedAnalyses SYCLLowerESIMDPass::run(Module &M,
                                          ModuleAnalysisManager &MAM) {
  // AlwaysInlinerPass is required for correctness.
  bool ForceInline = prepareForAlwaysInliner(M);
  if (ForceInline) {
    ModulePassManager MPM;
    MPM.addPass(AlwaysInlinerPass{});
    MPM.run(M, MAM);
  }

  generateKernelMetadata(M);
  // This function needs to run after generateKernelMetadata, as it
  // uses the generated metadata:
  size_t AmountOfESIMDIntrCalls = lowerSLMReservationCalls(M);
  SmallPtrSet<Type *, 4> GVTS = collectGenXVolatileTypes(M);
  lowerGlobalStores(M, GVTS);
  lowerGlobalsToVector(M);
  for (auto &F : M.functions()) {
    AmountOfESIMDIntrCalls += this->runOnFunction(F, GVTS);
  }

  fixFunctionReadWriteAttributes(M);

  // TODO FIXME ESIMD figure out less conservative result
  return AmountOfESIMDIntrCalls > 0 || ForceInline ? PreservedAnalyses::none()
                                                   : PreservedAnalyses::all();
}

size_t SYCLLowerESIMDPass::runOnFunction(Function &F,
                                         SmallPtrSetImpl<Type *> &GVTS) {
  SmallVector<CallInst *, 32> ESIMDIntrCalls;
  SmallVector<Instruction *, 8> ToErase;

  // The VC backend doesn't support debugging, and trying to use
  // non-optimized code often produces crashes or wrong answers.
  // The recommendation from the VC team was always optimize code,
  // even if the user requested no optimization. We already drop
  // debugging flags in the SYCL runtime, so also drop optnone and
  // noinline here.
  if (isESIMD(F) && F.hasFnAttribute(Attribute::OptimizeNone)) {
    F.removeFnAttr(Attribute::OptimizeNone);
    F.removeFnAttr(Attribute::NoInline);
  }

  for (Instruction &I : instructions(F)) {
    if (auto CastOp = dyn_cast<llvm::CastInst>(&I)) {
      llvm::Type *DstTy = CastOp->getDestTy();
      auto CastOpcode = CastOp->getOpcode();
      if (isa<FixedVectorType>(DstTy) &&
          ((CastOpcode == llvm::Instruction::FPToUI &&
            DstTy->getScalarType()->getPrimitiveSizeInBits() <= 32) ||
           (CastOpcode == llvm::Instruction::FPToSI &&
            DstTy->getScalarType()->getPrimitiveSizeInBits() < 32))) {
        IRBuilder<> Builder(&I);
        llvm::Value *Src = CastOp->getOperand(0);
        auto TmpTy = llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(DstTy->getContext()),
            cast<FixedVectorType>(DstTy)->getNumElements());
        if (CastOpcode == llvm::Instruction::FPToUI) {
          Src = Builder.CreateFPToUI(Src, TmpTy);
        } else {
          Src = Builder.CreateFPToSI(Src, TmpTy);
        }

        llvm::Instruction::CastOps TruncOp = llvm::Instruction::Trunc;
        llvm::Value *NewDst = Builder.CreateCast(TruncOp, Src, DstTy);
        CastOp->replaceAllUsesWith(NewDst);
        ToErase.push_back(CastOp);
      }
    }

    auto *CI = dyn_cast<CallInst>(&I);
    Function *Callee = nullptr;
    if (CI && (Callee = CI->getCalledFunction())) {
      // TODO workaround for ESIMD BE until it starts supporting @llvm.assume
      if (Callee->isIntrinsic()) {
        if (translateLLVMIntrinsic(CI)) {
          ToErase.push_back(CI);
        }
        continue;
      }
      StringRef Name = Callee->getName();

      // See if the Name represents an ESIMD intrinsic and demangle only if it
      // does.
      if (!Name.consume_front(ESIMD_INTRIN_PREF0) && !isDevicelibFunction(Name))
        continue;
      // now skip the digits
      Name = Name.drop_while([](char C) { return std::isdigit(C); });

      // process ESIMD builtins that go through special handling instead of
      // the translation procedure

      if (Name.startswith("__esimd_svm_block_ld") ||
          Name.startswith("__esimd_slm_block_ld")) {
        translateBlockLoad(*CI, Name.startswith("__esimd_slm_block_ld"));
        ToErase.push_back(CI);
        continue;
      }
      if (Name.startswith("__esimd_svm_block_st") ||
          Name.startswith("__esimd_slm_block_st")) {
        translateBlockStore(*CI, Name.startswith("__esimd_slm_block_st"));
        ToErase.push_back(CI);
        continue;
      }
      if (Name.startswith("__esimd_gather_ld") ||
          Name.startswith("__esimd_slm_gather_ld")) {
        translateGatherLoad(*CI, Name.startswith("__esimd_slm_gather_ld"));
        ToErase.push_back(CI);
        continue;
      }

      if (Name.startswith("__esimd_nbarrier_init")) {
        translateNbarrierInit(*CI);
        ToErase.push_back(CI);
        continue;
      }
      if (Name.startswith("__esimd_pack_mask")) {
        translatePackMask(*CI);
        ToErase.push_back(CI);
        continue;
      }
      if (Name.startswith("__esimd_unpack_mask")) {
        translateUnPackMask(*CI);
        ToErase.push_back(CI);
        continue;
      }
      // If vload/vstore is not about the vector-types used by
      // those globals marked as genx_volatile, We can translate
      // them directly into generic load/store inst. In this way
      // those insts can be optimized by llvm ASAP.
      if (Name.startswith("__esimd_vload")) {
        if (translateVLoad(*CI, GVTS)) {
          ToErase.push_back(CI);
          continue;
        }
      }
      if (Name.startswith("__esimd_vstore")) {
        if (translateVStore(*CI, GVTS)) {
          ToErase.push_back(CI);
          continue;
        }
      }

      if (Name.empty() ||
          (!Name.startswith(ESIMD_INTRIN_PREF1) && !isDevicelibFunction(Name)))
        continue;
      // this is ESIMD intrinsic - record for later translation
      ESIMDIntrCalls.push_back(CI);
    }

    // Translate loads from SPIRV builtin globals into GenX intrinsics
    auto *LI = dyn_cast<LoadInst>(&I);
    if (LI) {
      Value *LoadPtrOp = LI->getPointerOperand();
      Value *SpirvGlobal = nullptr;
      // Look through constant expressions to find SPIRV builtin globals
      // It may come with or without cast.
      auto *CE = dyn_cast<ConstantExpr>(LoadPtrOp);
      auto *GEPCE = dyn_cast<GetElementPtrConstantExpr>(LoadPtrOp);
      if (GEPCE) {
        SpirvGlobal = GEPCE->getOperand(0);
      } else if (CE) {
        assert(CE->isCast() && "ConstExpr should be a cast");
        SpirvGlobal = CE->getOperand(0);
      } else {
        SpirvGlobal = LoadPtrOp;
      }

      if (!isa<GlobalVariable>(SpirvGlobal) ||
          !SpirvGlobal->getName().startswith(SPIRV_INTRIN_PREF))
        continue;

      auto PrefLen = StringRef(SPIRV_INTRIN_PREF).size();

      // Translate all uses of the load instruction from SPIRV builtin global.
      // Replaces the original global load and it is uses and stores the old
      // instructions to ToErase.
      translateSpirvGlobalUses(LI, SpirvGlobal->getName().drop_front(PrefLen),
                               ToErase);
    }
  }
  // Now demangle and translate found ESIMD intrinsic calls
  for (auto *CI : ESIMDIntrCalls) {
    translateESIMDIntrinsicCall(*CI);
  }
  for (auto *CI : ToErase) {
    CI->eraseFromParent();
  }

  return ESIMDIntrCalls.size();
}
