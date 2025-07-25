//===- AddressSanitizer.cpp - memory error detector -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address basic correctness
// checker.
// Details of the algorithm:
//  https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm
//
// FIXME: This sanitizer does not yet handle scalable vectors
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include "llvm/Transforms/Instrumentation/SPIRVSanitizerCommonUtils.h"
#include "llvm/Transforms/Utils/ASanStackFrameLayout.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "asan"

static const uint64_t kDefaultShadowScale = 3;
static const uint64_t kDefaultShadowOffset32 = 1ULL << 29;
static const uint64_t kDefaultShadowOffset64 = 1ULL << 44;
static const uint64_t kDynamicShadowSentinel =
    std::numeric_limits<uint64_t>::max();
static const uint64_t kSmallX86_64ShadowOffsetBase = 0x7FFFFFFF;  // < 2G.
static const uint64_t kSmallX86_64ShadowOffsetAlignMask = ~0xFFFULL;
static const uint64_t kLinuxKasan_ShadowOffset64 = 0xdffffc0000000000;
static const uint64_t kPPC64_ShadowOffset64 = 1ULL << 44;
static const uint64_t kSystemZ_ShadowOffset64 = 1ULL << 52;
static const uint64_t kMIPS_ShadowOffsetN32 = 1ULL << 29;
static const uint64_t kMIPS32_ShadowOffset32 = 0x0aaa0000;
static const uint64_t kMIPS64_ShadowOffset64 = 1ULL << 37;
static const uint64_t kAArch64_ShadowOffset64 = 1ULL << 36;
static const uint64_t kLoongArch64_ShadowOffset64 = 1ULL << 46;
static const uint64_t kRISCV64_ShadowOffset64 = kDynamicShadowSentinel;
static const uint64_t kFreeBSD_ShadowOffset32 = 1ULL << 30;
static const uint64_t kFreeBSD_ShadowOffset64 = 1ULL << 46;
static const uint64_t kFreeBSDAArch64_ShadowOffset64 = 1ULL << 47;
static const uint64_t kFreeBSDKasan_ShadowOffset64 = 0xdffff7c000000000;
static const uint64_t kNetBSD_ShadowOffset32 = 1ULL << 30;
static const uint64_t kNetBSD_ShadowOffset64 = 1ULL << 46;
static const uint64_t kNetBSDKasan_ShadowOffset64 = 0xdfff900000000000;
static const uint64_t kPS_ShadowOffset64 = 1ULL << 40;
static const uint64_t kWindowsShadowOffset32 = 3ULL << 28;
static const uint64_t kWebAssemblyShadowOffset = 0;

// The shadow memory space is dynamically allocated.
static const uint64_t kWindowsShadowOffset64 = kDynamicShadowSentinel;

static const size_t kMinStackMallocSize = 1 << 6;   // 64B
static const size_t kMaxStackMallocSize = 1 << 16;  // 64K
static const uintptr_t kCurrentStackFrameMagic = 0x41B58AB3;
static const uintptr_t kRetiredStackFrameMagic = 0x45E0360E;

const char kAsanModuleCtorName[] = "asan.module_ctor";
const char kAsanModuleDtorName[] = "asan.module_dtor";
static const uint64_t kAsanCtorAndDtorPriority = 1;
// On Emscripten, the system needs more than one priorities for constructors.
static const uint64_t kAsanEmscriptenCtorAndDtorPriority = 50;
const char kAsanReportErrorTemplate[] = "__asan_report_";
const char kAsanRegisterGlobalsName[] = "__asan_register_globals";
const char kAsanUnregisterGlobalsName[] = "__asan_unregister_globals";
const char kAsanRegisterImageGlobalsName[] = "__asan_register_image_globals";
const char kAsanUnregisterImageGlobalsName[] =
    "__asan_unregister_image_globals";
const char kAsanRegisterElfGlobalsName[] = "__asan_register_elf_globals";
const char kAsanUnregisterElfGlobalsName[] = "__asan_unregister_elf_globals";
const char kAsanPoisonGlobalsName[] = "__asan_before_dynamic_init";
const char kAsanUnpoisonGlobalsName[] = "__asan_after_dynamic_init";
const char kAsanInitName[] = "__asan_init";
const char kAsanVersionCheckNamePrefix[] = "__asan_version_mismatch_check_v";
const char kAsanPtrCmp[] = "__sanitizer_ptr_cmp";
const char kAsanPtrSub[] = "__sanitizer_ptr_sub";
const char kAsanHandleNoReturnName[] = "__asan_handle_no_return";
static const int kMaxAsanStackMallocSizeClass = 10;
const char kAsanStackMallocNameTemplate[] = "__asan_stack_malloc_";
const char kAsanStackMallocAlwaysNameTemplate[] =
    "__asan_stack_malloc_always_";
const char kAsanStackFreeNameTemplate[] = "__asan_stack_free_";
const char kAsanGenPrefix[] = "___asan_gen_";
const char kODRGenPrefix[] = "__odr_asan_gen_";
const char kSanCovGenPrefix[] = "__sancov_gen_";
const char kAsanSetShadowPrefix[] = "__asan_set_shadow_";
const char kAsanPoisonStackMemoryName[] = "__asan_poison_stack_memory";
const char kAsanUnpoisonStackMemoryName[] = "__asan_unpoison_stack_memory";

// ASan version script has __asan_* wildcard. Triple underscore prevents a
// linker (gold) warning about attempting to export a local symbol.
const char kAsanGlobalsRegisteredFlagName[] = "___asan_globals_registered";

const char kAsanOptionDetectUseAfterReturn[] =
    "__asan_option_detect_stack_use_after_return";

const char kAsanShadowMemoryDynamicAddress[] =
    "__asan_shadow_memory_dynamic_address";

const char kAsanAllocaPoison[] = "__asan_alloca_poison";
const char kAsanAllocasUnpoison[] = "__asan_allocas_unpoison";

const char kAMDGPUAddressSharedName[] = "llvm.amdgcn.is.shared";
const char kAMDGPUAddressPrivateName[] = "llvm.amdgcn.is.private";
const char kAMDGPUBallotName[] = "llvm.amdgcn.ballot.i64";
const char kAMDGPUUnreachableName[] = "llvm.amdgcn.unreachable";

const char kAsanMemToShadow[] = "__asan_mem_to_shadow";

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;
static const size_t kNumberOfAddressSpace = 5;

static const uint64_t kAllocaRzSize = 32;

// ASanAccessInfo implementation constants.
constexpr size_t kCompileKernelShift = 0;
constexpr size_t kCompileKernelMask = 0x1;
constexpr size_t kAccessSizeIndexShift = 1;
constexpr size_t kAccessSizeIndexMask = 0xf;
constexpr size_t kIsWriteShift = 5;
constexpr size_t kIsWriteMask = 0x1;

// Command-line flags.

static cl::opt<bool> ClEnableKasan(
    "asan-kernel", cl::desc("Enable KernelAddressSanitizer instrumentation"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClRecover(
    "asan-recover",
    cl::desc("Enable recovery mode (continue-after-error)."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClInsertVersionCheck(
    "asan-guard-against-version-mismatch",
    cl::desc("Guard against compiler/runtime version mismatch."), cl::Hidden,
    cl::init(true));

// This flag may need to be replaced with -f[no-]asan-reads.
static cl::opt<bool> ClInstrumentReads("asan-instrument-reads",
                                       cl::desc("instrument read instructions"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool> ClInstrumentWrites(
    "asan-instrument-writes", cl::desc("instrument write instructions"),
    cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClUseStackSafety("asan-use-stack-safety", cl::Hidden, cl::init(true),
                     cl::Hidden, cl::desc("Use Stack Safety analysis results"),
                     cl::Optional);

static cl::opt<bool> ClInstrumentAtomics(
    "asan-instrument-atomics",
    cl::desc("instrument atomic instructions (rmw, cmpxchg)"), cl::Hidden,
    cl::init(true));

static cl::opt<bool>
    ClInstrumentByval("asan-instrument-byval",
                      cl::desc("instrument byval call arguments"), cl::Hidden,
                      cl::init(true));

static cl::opt<bool> ClAlwaysSlowPath(
    "asan-always-slow-path",
    cl::desc("use instrumentation with slow path for all accesses"), cl::Hidden,
    cl::init(false));

static cl::opt<bool> ClForceDynamicShadow(
    "asan-force-dynamic-shadow",
    cl::desc("Load shadow address into a local variable for each function"),
    cl::Hidden, cl::init(false));

static cl::opt<bool>
    ClWithIfunc("asan-with-ifunc",
                cl::desc("Access dynamic shadow through an ifunc global on "
                         "platforms that support this"),
                cl::Hidden, cl::init(true));

static cl::opt<bool> ClWithIfuncSuppressRemat(
    "asan-with-ifunc-suppress-remat",
    cl::desc("Suppress rematerialization of dynamic shadow address by passing "
             "it through inline asm in prologue."),
    cl::Hidden, cl::init(true));

// This flag limits the number of instructions to be instrumented
// in any given BB. Normally, this should be set to unlimited (INT_MAX),
// but due to http://llvm.org/bugs/show_bug.cgi?id=12652 we temporary
// set it to 10000.
static cl::opt<int> ClMaxInsnsToInstrumentPerBB(
    "asan-max-ins-per-bb", cl::init(10000),
    cl::desc("maximal number of instructions to instrument in any given BB"),
    cl::Hidden);

// This flag may need to be replaced with -f[no]asan-stack.
static cl::opt<bool> ClStack("asan-stack", cl::desc("Handle stack memory"),
                             cl::Hidden, cl::init(true));
static cl::opt<uint32_t> ClMaxInlinePoisoningSize(
    "asan-max-inline-poisoning-size",
    cl::desc(
        "Inline shadow poisoning for blocks up to the given size in bytes."),
    cl::Hidden, cl::init(64));

static cl::opt<AsanDetectStackUseAfterReturnMode> ClUseAfterReturn(
    "asan-use-after-return",
    cl::desc("Sets the mode of detection for stack-use-after-return."),
    cl::values(
        clEnumValN(AsanDetectStackUseAfterReturnMode::Never, "never",
                   "Never detect stack use after return."),
        clEnumValN(
            AsanDetectStackUseAfterReturnMode::Runtime, "runtime",
            "Detect stack use after return if "
            "binary flag 'ASAN_OPTIONS=detect_stack_use_after_return' is set."),
        clEnumValN(AsanDetectStackUseAfterReturnMode::Always, "always",
                   "Always detect stack use after return.")),
    cl::Hidden, cl::init(AsanDetectStackUseAfterReturnMode::Runtime));

static cl::opt<bool> ClRedzoneByvalArgs("asan-redzone-byval-args",
                                        cl::desc("Create redzones for byval "
                                                 "arguments (extra copy "
                                                 "required)"), cl::Hidden,
                                        cl::init(true));

static cl::opt<bool> ClUseAfterScope("asan-use-after-scope",
                                     cl::desc("Check stack-use-after-scope"),
                                     cl::Hidden, cl::init(false));

// This flag may need to be replaced with -f[no]asan-globals.
static cl::opt<bool> ClGlobals("asan-globals",
                               cl::desc("Handle global objects"), cl::Hidden,
                               cl::init(true));

static cl::opt<bool> ClInitializers("asan-initialization-order",
                                    cl::desc("Handle C++ initializer order"),
                                    cl::Hidden, cl::init(true));

static cl::opt<bool> ClInvalidPointerPairs(
    "asan-detect-invalid-pointer-pair",
    cl::desc("Instrument <, <=, >, >=, - with pointer operands"), cl::Hidden,
    cl::init(false));

static cl::opt<bool> ClInvalidPointerCmp(
    "asan-detect-invalid-pointer-cmp",
    cl::desc("Instrument <, <=, >, >= with pointer operands"), cl::Hidden,
    cl::init(false));

static cl::opt<bool> ClInvalidPointerSub(
    "asan-detect-invalid-pointer-sub",
    cl::desc("Instrument - operations with pointer operands"), cl::Hidden,
    cl::init(false));

static cl::opt<unsigned> ClRealignStack(
    "asan-realign-stack",
    cl::desc("Realign stack to the value of this flag (power of two)"),
    cl::Hidden, cl::init(32));

static cl::opt<int> ClInstrumentationWithCallsThreshold(
    "asan-instrumentation-with-call-threshold",
    cl::desc("If the function being instrumented contains more than "
             "this number of memory accesses, use callbacks instead of "
             "inline checks (-1 means never use callbacks)."),
    cl::Hidden, cl::init(7000));

static cl::opt<std::string> ClMemoryAccessCallbackPrefix(
    "asan-memory-access-callback-prefix",
    cl::desc("Prefix for memory access callbacks"), cl::Hidden,
    cl::init("__asan_"));

static cl::opt<bool> ClKasanMemIntrinCallbackPrefix(
    "asan-kernel-mem-intrinsic-prefix",
    cl::desc("Use prefix for memory intrinsics in KASAN mode"), cl::Hidden,
    cl::init(false));

static cl::opt<bool>
    ClInstrumentDynamicAllocas("asan-instrument-dynamic-allocas",
                               cl::desc("instrument dynamic allocas"),
                               cl::Hidden, cl::init(true));

static cl::opt<bool> ClSkipPromotableAllocas(
    "asan-skip-promotable-allocas",
    cl::desc("Do not instrument promotable allocas"), cl::Hidden,
    cl::init(true));

static cl::opt<AsanCtorKind> ClConstructorKind(
    "asan-constructor-kind",
    cl::desc("Sets the ASan constructor kind"),
    cl::values(clEnumValN(AsanCtorKind::None, "none", "No constructors"),
               clEnumValN(AsanCtorKind::Global, "global",
                          "Use global constructors")),
    cl::init(AsanCtorKind::Global), cl::Hidden);
// These flags allow to change the shadow mapping.
// The shadow mapping looks like
//    Shadow = (Mem >> scale) + offset

static cl::opt<int> ClMappingScale("asan-mapping-scale",
                                   cl::desc("scale of asan shadow mapping"),
                                   cl::Hidden, cl::init(0));

static cl::opt<uint64_t>
    ClMappingOffset("asan-mapping-offset",
                    cl::desc("offset of asan shadow mapping [EXPERIMENTAL]"),
                    cl::Hidden, cl::init(0));

// Optimization flags. Not user visible, used mostly for testing
// and benchmarking the tool.

static cl::opt<bool> ClOpt("asan-opt", cl::desc("Optimize instrumentation"),
                           cl::Hidden, cl::init(true));

static cl::opt<bool> ClOptimizeCallbacks("asan-optimize-callbacks",
                                         cl::desc("Optimize callbacks"),
                                         cl::Hidden, cl::init(false));

static cl::opt<bool> ClOptSameTemp(
    "asan-opt-same-temp", cl::desc("Instrument the same temp just once"),
    cl::Hidden, cl::init(true));

static cl::opt<bool> ClOptGlobals("asan-opt-globals",
                                  cl::desc("Don't instrument scalar globals"),
                                  cl::Hidden, cl::init(true));

static cl::opt<bool> ClOptStack(
    "asan-opt-stack", cl::desc("Don't instrument scalar stack variables"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClDynamicAllocaStack(
    "asan-stack-dynamic-alloca",
    cl::desc("Use dynamic alloca to represent stack variables"), cl::Hidden,
    cl::init(true));

static cl::opt<uint32_t> ClForceExperiment(
    "asan-force-experiment",
    cl::desc("Force optimization experiment (for testing)"), cl::Hidden,
    cl::init(0));

static cl::opt<bool>
    ClUsePrivateAlias("asan-use-private-alias",
                      cl::desc("Use private aliases for global variables"),
                      cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClUseOdrIndicator("asan-use-odr-indicator",
                      cl::desc("Use odr indicators to improve ODR reporting"),
                      cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClUseGlobalsGC("asan-globals-live-support",
                   cl::desc("Use linker features to support dead "
                            "code stripping of globals"),
                   cl::Hidden, cl::init(true));

// This is on by default even though there is a bug in gold:
// https://sourceware.org/bugzilla/show_bug.cgi?id=19002
static cl::opt<bool>
    ClWithComdat("asan-with-comdat",
                 cl::desc("Place ASan constructors in comdat sections"),
                 cl::Hidden, cl::init(true));

static cl::opt<AsanDtorKind> ClOverrideDestructorKind(
    "asan-destructor-kind",
    cl::desc("Sets the ASan destructor kind. The default is to use the value "
             "provided to the pass constructor"),
    cl::values(clEnumValN(AsanDtorKind::None, "none", "No destructors"),
               clEnumValN(AsanDtorKind::Global, "global",
                          "Use global destructors")),
    cl::init(AsanDtorKind::Invalid), cl::Hidden);

// SYCL flags
static cl::opt<bool>
    ClSpirOffloadPrivates("asan-spir-privates",
                          cl::desc("instrument private pointer"), cl::Hidden,
                          cl::init(true));

static cl::opt<bool> ClSpirOffloadGlobals("asan-spir-globals",
                                          cl::desc("instrument global pointer"),
                                          cl::Hidden, cl::init(true));

static cl::opt<bool> ClSpirOffloadLocals("asan-spir-locals",
                                         cl::desc("instrument local pointer"),
                                         cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClSpirOffloadGenerics("asan-spir-generics",
                          cl::desc("instrument generic pointer"), cl::Hidden,
                          cl::init(true));

static cl::opt<bool> ClDeviceGlobals("asan-device-globals",
                                     cl::desc("instrument device globals"),
                                     cl::Hidden, cl::init(true));

// Debug flags.

static cl::opt<int> ClDebug("asan-debug", cl::desc("debug"), cl::Hidden,
                            cl::init(0));

static cl::opt<int> ClDebugStack("asan-debug-stack", cl::desc("debug stack"),
                                 cl::Hidden, cl::init(0));

static cl::opt<std::string> ClDebugFunc("asan-debug-func", cl::Hidden,
                                        cl::desc("Debug func"));

static cl::opt<int> ClDebugMin("asan-debug-min", cl::desc("Debug min inst"),
                               cl::Hidden, cl::init(-1));

static cl::opt<int> ClDebugMax("asan-debug-max", cl::desc("Debug max inst"),
                               cl::Hidden, cl::init(-1));

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOptimizedAccessesToGlobalVar,
          "Number of optimized accesses to global vars");
STATISTIC(NumOptimizedAccessesToStackVar,
          "Number of optimized accesses to stack vars");

namespace {

/// This struct defines the shadow mapping using the rule:
///   shadow = (mem >> Scale) ADD-or-OR Offset.
/// If InGlobal is true, then
///   extern char __asan_shadow[];
///   shadow = (mem >> Scale) + &__asan_shadow
struct ShadowMapping {
  int Scale;
  uint64_t Offset;
  bool OrShadowOffset;
  bool InGlobal;
};

} // end anonymous namespace

static ShadowMapping getShadowMapping(const Triple &TargetTriple, int LongSize,
                                      bool IsKasan) {
  bool IsAndroid = TargetTriple.isAndroid();
  bool IsIOS = TargetTriple.isiOS() || TargetTriple.isWatchOS() ||
               TargetTriple.isDriverKit();
  bool IsMacOS = TargetTriple.isMacOSX();
  bool IsFreeBSD = TargetTriple.isOSFreeBSD();
  bool IsNetBSD = TargetTriple.isOSNetBSD();
  bool IsPS = TargetTriple.isPS();
  bool IsLinux = TargetTriple.isOSLinux();
  bool IsPPC64 = TargetTriple.getArch() == Triple::ppc64 ||
                 TargetTriple.getArch() == Triple::ppc64le;
  bool IsSystemZ = TargetTriple.getArch() == Triple::systemz;
  bool IsX86_64 = TargetTriple.getArch() == Triple::x86_64;
  bool IsMIPSN32ABI = TargetTriple.isABIN32();
  bool IsMIPS32 = TargetTriple.isMIPS32();
  bool IsMIPS64 = TargetTriple.isMIPS64();
  bool IsArmOrThumb = TargetTriple.isARM() || TargetTriple.isThumb();
  bool IsAArch64 = TargetTriple.getArch() == Triple::aarch64 ||
                   TargetTriple.getArch() == Triple::aarch64_be;
  bool IsLoongArch64 = TargetTriple.isLoongArch64();
  bool IsRISCV64 = TargetTriple.getArch() == Triple::riscv64;
  bool IsWindows = TargetTriple.isOSWindows();
  bool IsFuchsia = TargetTriple.isOSFuchsia();
  bool IsAMDGPU = TargetTriple.isAMDGPU();
  bool IsHaiku = TargetTriple.isOSHaiku();
  bool IsWasm = TargetTriple.isWasm();

  ShadowMapping Mapping;

  Mapping.Scale = kDefaultShadowScale;
  if (ClMappingScale.getNumOccurrences() > 0) {
    Mapping.Scale = ClMappingScale;
  }

  if (LongSize == 32) {
    if (IsAndroid)
      Mapping.Offset = kDynamicShadowSentinel;
    else if (IsMIPSN32ABI)
      Mapping.Offset = kMIPS_ShadowOffsetN32;
    else if (IsMIPS32)
      Mapping.Offset = kMIPS32_ShadowOffset32;
    else if (IsFreeBSD)
      Mapping.Offset = kFreeBSD_ShadowOffset32;
    else if (IsNetBSD)
      Mapping.Offset = kNetBSD_ShadowOffset32;
    else if (IsIOS)
      Mapping.Offset = kDynamicShadowSentinel;
    else if (IsWindows)
      Mapping.Offset = kWindowsShadowOffset32;
    else if (IsWasm)
      Mapping.Offset = kWebAssemblyShadowOffset;
    else
      Mapping.Offset = kDefaultShadowOffset32;
  } else {  // LongSize == 64
    // Fuchsia is always PIE, which means that the beginning of the address
    // space is always available.
    if (IsFuchsia)
      Mapping.Offset = 0;
    else if (IsPPC64)
      Mapping.Offset = kPPC64_ShadowOffset64;
    else if (IsSystemZ)
      Mapping.Offset = kSystemZ_ShadowOffset64;
    else if (IsFreeBSD && IsAArch64)
        Mapping.Offset = kFreeBSDAArch64_ShadowOffset64;
    else if (IsFreeBSD && !IsMIPS64) {
      if (IsKasan)
        Mapping.Offset = kFreeBSDKasan_ShadowOffset64;
      else
        Mapping.Offset = kFreeBSD_ShadowOffset64;
    } else if (IsNetBSD) {
      if (IsKasan)
        Mapping.Offset = kNetBSDKasan_ShadowOffset64;
      else
        Mapping.Offset = kNetBSD_ShadowOffset64;
    } else if (IsPS)
      Mapping.Offset = kPS_ShadowOffset64;
    else if (IsLinux && IsX86_64) {
      if (IsKasan)
        Mapping.Offset = kLinuxKasan_ShadowOffset64;
      else
        Mapping.Offset = (kSmallX86_64ShadowOffsetBase &
                          (kSmallX86_64ShadowOffsetAlignMask << Mapping.Scale));
    } else if (IsWindows && IsX86_64) {
      Mapping.Offset = kWindowsShadowOffset64;
    } else if (IsMIPS64)
      Mapping.Offset = kMIPS64_ShadowOffset64;
    else if (IsIOS)
      Mapping.Offset = kDynamicShadowSentinel;
    else if (IsMacOS && IsAArch64)
      Mapping.Offset = kDynamicShadowSentinel;
    else if (IsAArch64)
      Mapping.Offset = kAArch64_ShadowOffset64;
    else if (IsLoongArch64)
      Mapping.Offset = kLoongArch64_ShadowOffset64;
    else if (IsRISCV64)
      Mapping.Offset = kRISCV64_ShadowOffset64;
    else if (IsAMDGPU)
      Mapping.Offset = (kSmallX86_64ShadowOffsetBase &
                        (kSmallX86_64ShadowOffsetAlignMask << Mapping.Scale));
    else if (IsHaiku && IsX86_64)
      Mapping.Offset = (kSmallX86_64ShadowOffsetBase &
                        (kSmallX86_64ShadowOffsetAlignMask << Mapping.Scale));
    else
      Mapping.Offset = kDefaultShadowOffset64;
  }

  if (ClForceDynamicShadow) {
    Mapping.Offset = kDynamicShadowSentinel;
  }

  if (ClMappingOffset.getNumOccurrences() > 0) {
    Mapping.Offset = ClMappingOffset;
  }

  // OR-ing shadow offset if more efficient (at least on x86) if the offset
  // is a power of two, but on ppc64 and loongarch64 we have to use add since
  // the shadow offset is not necessarily 1/8-th of the address space.  On
  // SystemZ, we could OR the constant in a single instruction, but it's more
  // efficient to load it once and use indexed addressing.
  Mapping.OrShadowOffset = !IsAArch64 && !IsPPC64 && !IsSystemZ && !IsPS &&
                           !IsRISCV64 && !IsLoongArch64 &&
                           !(Mapping.Offset & (Mapping.Offset - 1)) &&
                           Mapping.Offset != kDynamicShadowSentinel;
  bool IsAndroidWithIfuncSupport =
      IsAndroid && !TargetTriple.isAndroidVersionLT(21);
  Mapping.InGlobal = ClWithIfunc && IsAndroidWithIfuncSupport && IsArmOrThumb;

  return Mapping;
}

namespace llvm {
void getAddressSanitizerParams(const Triple &TargetTriple, int LongSize,
                               bool IsKasan, uint64_t *ShadowBase,
                               int *MappingScale, bool *OrShadowOffset) {
  auto Mapping = getShadowMapping(TargetTriple, LongSize, IsKasan);
  *ShadowBase = Mapping.Offset;
  *MappingScale = Mapping.Scale;
  *OrShadowOffset = Mapping.OrShadowOffset;
}

void removeASanIncompatibleFnAttributes(Function &F, bool ReadsArgMem) {
  // Sanitizer checks read from shadow, which invalidates memory(argmem: *).
  //
  // This is not only true for sanitized functions, because AttrInfer can
  // infer those attributes on libc functions, which is not true if those
  // are instrumented (Android) or intercepted.
  //
  // We might want to model ASan shadow memory more opaquely to get rid of
  // this problem altogether, by hiding the shadow memory write in an
  // intrinsic, essentially like in the AArch64StackTagging pass. But that's
  // for another day.

  // The API is weird. `onlyReadsMemory` actually means "does not write", and
  // `onlyWritesMemory` actually means "does not read". So we reconstruct
  // "accesses memory" && "does not read" <=> "writes".
  bool Changed = false;
  if (!F.doesNotAccessMemory()) {
    bool WritesMemory = !F.onlyReadsMemory();
    bool ReadsMemory = !F.onlyWritesMemory();
    if ((WritesMemory && !ReadsMemory) || F.onlyAccessesArgMemory()) {
      F.removeFnAttr(Attribute::Memory);
      Changed = true;
    }
  }
  if (ReadsArgMem) {
    for (Argument &A : F.args()) {
      if (A.hasAttribute(Attribute::WriteOnly)) {
        A.removeAttr(Attribute::WriteOnly);
        Changed = true;
      }
    }
  }
  if (Changed) {
    // nobuiltin makes sure later passes don't restore assumptions about
    // the function.
    F.addFnAttr(Attribute::NoBuiltin);
  }
}

ASanAccessInfo::ASanAccessInfo(int32_t Packed)
    : Packed(Packed),
      AccessSizeIndex((Packed >> kAccessSizeIndexShift) & kAccessSizeIndexMask),
      IsWrite((Packed >> kIsWriteShift) & kIsWriteMask),
      CompileKernel((Packed >> kCompileKernelShift) & kCompileKernelMask) {}

ASanAccessInfo::ASanAccessInfo(bool IsWrite, bool CompileKernel,
                               uint8_t AccessSizeIndex)
    : Packed((IsWrite << kIsWriteShift) +
             (CompileKernel << kCompileKernelShift) +
             (AccessSizeIndex << kAccessSizeIndexShift)),
      AccessSizeIndex(AccessSizeIndex), IsWrite(IsWrite),
      CompileKernel(CompileKernel) {}

} // namespace llvm

static uint64_t getRedzoneSizeForScale(int MappingScale) {
  // Redzone used for stack and globals is at least 32 bytes.
  // For scales 6 and 7, the redzone has to be 64 and 128 bytes respectively.
  return std::max(32U, 1U << MappingScale);
}

static uint64_t GetCtorAndDtorPriority(Triple &TargetTriple) {
  if (TargetTriple.isOSEmscripten()) {
    return kAsanEmscriptenCtorAndDtorPriority;
  } else {
    return kAsanCtorAndDtorPriority;
  }
}

static Twine genName(StringRef suffix) {
  return Twine(kAsanGenPrefix) + suffix;
}

namespace {
/// Helper RAII class to post-process inserted asan runtime calls during a
/// pass on a single Function. Upon end of scope, detects and applies the
/// required funclet OpBundle.
class RuntimeCallInserter {
  Function *OwnerFn = nullptr;
  bool TrackInsertedCalls = false;
  SmallVector<CallInst *> InsertedCalls;

public:
  RuntimeCallInserter(Function &Fn) : OwnerFn(&Fn) {
    if (Fn.hasPersonalityFn()) {
      auto Personality = classifyEHPersonality(Fn.getPersonalityFn());
      if (isScopedEHPersonality(Personality))
        TrackInsertedCalls = true;
    }
  }

  ~RuntimeCallInserter() {
    if (InsertedCalls.empty())
      return;
    assert(TrackInsertedCalls && "Calls were wrongly tracked");

    DenseMap<BasicBlock *, ColorVector> BlockColors = colorEHFunclets(*OwnerFn);
    for (CallInst *CI : InsertedCalls) {
      BasicBlock *BB = CI->getParent();
      assert(BB && "Instruction doesn't belong to a BasicBlock");
      assert(BB->getParent() == OwnerFn &&
             "Instruction doesn't belong to the expected Function!");

      ColorVector &Colors = BlockColors[BB];
      // funclet opbundles are only valid in monochromatic BBs.
      // Note that unreachable BBs are seen as colorless by colorEHFunclets()
      // and will be DCE'ed later.
      if (Colors.empty())
        continue;
      if (Colors.size() != 1) {
        OwnerFn->getContext().emitError(
            "Instruction's BasicBlock is not monochromatic");
        continue;
      }

      BasicBlock *Color = Colors.front();
      BasicBlock::iterator EHPadIt = Color->getFirstNonPHIIt();

      if (EHPadIt != Color->end() && EHPadIt->isEHPad()) {
        // Replace CI with a clone with an added funclet OperandBundle
        OperandBundleDef OB("funclet", &*EHPadIt);
        auto *NewCall = CallBase::addOperandBundle(CI, LLVMContext::OB_funclet,
                                                   OB, CI->getIterator());
        NewCall->copyMetadata(*CI);
        CI->replaceAllUsesWith(NewCall);
        CI->eraseFromParent();
      }
    }
  }

  CallInst *createRuntimeCall(IRBuilder<> &IRB, FunctionCallee Callee,
                              ArrayRef<Value *> Args = {},
                              const Twine &Name = "") {
    assert(IRB.GetInsertBlock()->getParent() == OwnerFn);

    CallInst *Inst = IRB.CreateCall(Callee, Args, Name, nullptr);
    if (TrackInsertedCalls)
      InsertedCalls.push_back(Inst);
    return Inst;
  }
};

/// AddressSanitizer: instrument the code in module to find memory bugs.
struct AddressSanitizer {
  AddressSanitizer(Module &M, const StackSafetyGlobalInfo *SSGI,
                   int InstrumentationWithCallsThreshold,
                   uint32_t MaxInlinePoisoningSize, bool CompileKernel = false,
                   bool Recover = false, bool UseAfterScope = false,
                   AsanDetectStackUseAfterReturnMode UseAfterReturn =
                       AsanDetectStackUseAfterReturnMode::Runtime)
      : M(M),
        CompileKernel(ClEnableKasan.getNumOccurrences() > 0 ? ClEnableKasan
                                                            : CompileKernel),
        Recover(ClRecover.getNumOccurrences() > 0 ? ClRecover : Recover),
        UseAfterScope(UseAfterScope || ClUseAfterScope),
        UseAfterReturn(ClUseAfterReturn.getNumOccurrences() ? ClUseAfterReturn
                                                            : UseAfterReturn),
        SSGI(SSGI),
        InstrumentationWithCallsThreshold(
            ClInstrumentationWithCallsThreshold.getNumOccurrences() > 0
                ? ClInstrumentationWithCallsThreshold
                : InstrumentationWithCallsThreshold),
        MaxInlinePoisoningSize(ClMaxInlinePoisoningSize.getNumOccurrences() > 0
                                   ? ClMaxInlinePoisoningSize
                                   : MaxInlinePoisoningSize) {
    C = &(M.getContext());
    DL = &M.getDataLayout();
    LongSize = M.getDataLayout().getPointerSizeInBits();
    IntptrTy = Type::getIntNTy(*C, LongSize);
    PtrTy = PointerType::getUnqual(*C);
    Int32Ty = Type::getInt32Ty(*C);
    TargetTriple = M.getTargetTriple();

    Mapping = getShadowMapping(TargetTriple, LongSize, this->CompileKernel);

    assert(this->UseAfterReturn != AsanDetectStackUseAfterReturnMode::Invalid);
  }

  TypeSize getAllocaSizeInBytes(const AllocaInst &AI) const {
    return *AI.getAllocationSize(AI.getDataLayout());
  }

  /// Check if we want (and can) handle this alloca.
  bool isInterestingAlloca(const AllocaInst &AI);

  bool ignoreAccess(Instruction *Inst, Value *Ptr);
  void getInterestingMemoryOperands(
      Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting);

  void instrumentMop(ObjectSizeOffsetVisitor &ObjSizeVis,
                     InterestingMemoryOperand &O, bool UseCalls,
                     const DataLayout &DL, RuntimeCallInserter &RTCI);
  void instrumentPointerComparisonOrSubtraction(Instruction *I,
                                                RuntimeCallInserter &RTCI);
  void instrumentAddress(Instruction *OrigIns, Instruction *InsertBefore,
                         Value *Addr, MaybeAlign Alignment,
                         uint32_t TypeStoreSize, bool IsWrite,
                         Value *SizeArgument, bool UseCalls, uint32_t Exp,
                         RuntimeCallInserter &RTCI);
  Instruction *instrumentAMDGPUAddress(Instruction *OrigIns,
                                       Instruction *InsertBefore, Value *Addr,
                                       uint32_t TypeStoreSize, bool IsWrite,
                                       Value *SizeArgument);
  Instruction *genAMDGPUReportBlock(IRBuilder<> &IRB, Value *Cond,
                                    bool Recover);
  void instrumentUnusualSizeOrAlignment(Instruction *I,
                                        Instruction *InsertBefore, Value *Addr,
                                        TypeSize TypeStoreSize, bool IsWrite,
                                        Value *SizeArgument, bool UseCalls,
                                        uint32_t Exp,
                                        RuntimeCallInserter &RTCI);
  void instrumentMaskedLoadOrStore(AddressSanitizer *Pass, const DataLayout &DL,
                                   Type *IntptrTy, Value *Mask, Value *EVL,
                                   Value *Stride, Instruction *I, Value *Addr,
                                   MaybeAlign Alignment, unsigned Granularity,
                                   Type *OpType, bool IsWrite,
                                   Value *SizeArgument, bool UseCalls,
                                   uint32_t Exp, RuntimeCallInserter &RTCI);
  Value *createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                           Value *ShadowValue, uint32_t TypeStoreSize);
  Instruction *generateCrashCode(Instruction *InsertBefore, Value *Addr,
                                 bool IsWrite, size_t AccessSizeIndex,
                                 Value *SizeArgument, uint32_t Exp,
                                 RuntimeCallInserter &RTCI);
  void instrumentMemIntrinsic(MemIntrinsic *MI, RuntimeCallInserter &RTCI);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB,
                     uint32_t AddressSpace = kSpirOffloadPrivateAS);
  bool suppressInstrumentationSiteForDebug(int &Instrumented);
  bool instrumentFunction(Function &F, const TargetLibraryInfo *TLI);
  bool maybeInsertAsanInitAtFunctionEntry(Function &F);
  bool maybeInsertDynamicShadowAtFunctionEntry(Function &F);
  void markEscapedLocalAllocas(Function &F);
  bool instrumentSyclDynamicLocalMemory(Function &F);
  void instrumentInitAsanLaunchInfo(Function &F, const TargetLibraryInfo *TLI);

  void AppendDebugInfoToArgs(Instruction *InsertBefore, Value *Addr,
                             SmallVectorImpl<Value *> &Args);

private:
  friend struct FunctionStackPoisoner;

  void initializeCallbacks(const TargetLibraryInfo *TLI);

  bool LooksLikeCodeInBug11395(Instruction *I);
  bool GlobalIsLinkerInitialized(GlobalVariable *G);
  bool isSafeAccess(ObjectSizeOffsetVisitor &ObjSizeVis, Value *Addr,
                    TypeSize TypeStoreSize) const;

  /// Helper to cleanup per-function state.
  struct FunctionStateRAII {
    AddressSanitizer *Pass;

    FunctionStateRAII(AddressSanitizer *Pass) : Pass(Pass) {
      assert(Pass->ProcessedAllocas.empty() &&
             "last pass forgot to clear cache");
      assert(!Pass->LocalDynamicShadow);
    }

    ~FunctionStateRAII() {
      Pass->LocalDynamicShadow = nullptr;
      Pass->ProcessedAllocas.clear();
    }
  };

  Module &M;
  LLVMContext *C;
  const DataLayout *DL;
  Triple TargetTriple;
  int LongSize;
  bool CompileKernel;
  bool Recover;
  bool UseAfterScope;
  AsanDetectStackUseAfterReturnMode UseAfterReturn;
  Type *IntptrTy;
  Type *Int32Ty;
  PointerType *PtrTy;
  ShadowMapping Mapping;
  FunctionCallee AsanHandleNoReturnFunc;
  FunctionCallee AsanPtrCmpFunction, AsanPtrSubFunction;
  FunctionCallee AsanSetShadowDynamicLocalFunc;
  FunctionCallee AsanUnpoisonShadowDynamicLocalFunc;
  Constant *AsanShadowGlobal;
  Constant *AsanLaunchInfo;

  // These arrays is indexed by AccessIsWrite, Experiment and log2(AccessSize).
  FunctionCallee AsanErrorCallback[2][2][kNumberOfAccessSizes];
  FunctionCallee AsanMemoryAccessCallback[2][2][kNumberOfAccessSizes];
  FunctionCallee AsanMemoryAccessCallbackAS[2][2][kNumberOfAccessSizes]
                                           [kNumberOfAddressSpace];

  // These arrays is indexed by AccessIsWrite and Experiment.
  FunctionCallee AsanErrorCallbackSized[2][2];
  FunctionCallee AsanMemoryAccessCallbackSized[2][2];
  FunctionCallee AsanMemoryAccessCallbackSizedAS[2][2][kNumberOfAddressSpace];

  FunctionCallee AsanMemmove, AsanMemcpy, AsanMemset;
  Value *LocalDynamicShadow = nullptr;
  const StackSafetyGlobalInfo *SSGI;
  DenseMap<const AllocaInst *, bool> ProcessedAllocas;

  FunctionCallee AMDGPUAddressShared;
  FunctionCallee AMDGPUAddressPrivate;
  int InstrumentationWithCallsThreshold;
  uint32_t MaxInlinePoisoningSize;

  FunctionCallee AsanMemToShadow;
};

class ModuleAddressSanitizer {
public:
  ModuleAddressSanitizer(Module &M, bool InsertVersionCheck,
                         bool CompileKernel = false, bool Recover = false,
                         bool UseGlobalsGC = true, bool UseOdrIndicator = true,
                         AsanDtorKind DestructorKind = AsanDtorKind::Global,
                         AsanCtorKind ConstructorKind = AsanCtorKind::Global)
      : M(M),
        CompileKernel(ClEnableKasan.getNumOccurrences() > 0 ? ClEnableKasan
                                                            : CompileKernel),
        InsertVersionCheck(ClInsertVersionCheck.getNumOccurrences() > 0
                               ? ClInsertVersionCheck
                               : InsertVersionCheck),
        Recover(ClRecover.getNumOccurrences() > 0 ? ClRecover : Recover),
        UseGlobalsGC(UseGlobalsGC && ClUseGlobalsGC && !this->CompileKernel),
        // Enable aliases as they should have no downside with ODR indicators.
        UsePrivateAlias(ClUsePrivateAlias.getNumOccurrences() > 0
                            ? ClUsePrivateAlias
                            : UseOdrIndicator),
        UseOdrIndicator(ClUseOdrIndicator.getNumOccurrences() > 0
                            ? ClUseOdrIndicator
                            : UseOdrIndicator),
        // Not a typo: ClWithComdat is almost completely pointless without
        // ClUseGlobalsGC (because then it only works on modules without
        // globals, which are rare); it is a prerequisite for ClUseGlobalsGC;
        // and both suffer from gold PR19002 for which UseGlobalsGC constructor
        // argument is designed as workaround. Therefore, disable both
        // ClWithComdat and ClUseGlobalsGC unless the frontend says it's ok to
        // do globals-gc.
        UseCtorComdat(UseGlobalsGC && ClWithComdat && !this->CompileKernel),
        DestructorKind(DestructorKind),
        ConstructorKind(ClConstructorKind.getNumOccurrences() > 0
                            ? ClConstructorKind
                            : ConstructorKind) {
    C = &(M.getContext());
    int LongSize = M.getDataLayout().getPointerSizeInBits();
    IntptrTy = Type::getIntNTy(*C, LongSize);
    PtrTy = PointerType::getUnqual(*C);
    TargetTriple = M.getTargetTriple();
    Mapping = getShadowMapping(TargetTriple, LongSize, this->CompileKernel);

    if (ClOverrideDestructorKind != AsanDtorKind::Invalid)
      this->DestructorKind = ClOverrideDestructorKind;
    assert(this->DestructorKind != AsanDtorKind::Invalid);
  }

  bool instrumentModule();

private:
  void initializeCallbacks();

  void instrumentDeviceGlobal(IRBuilder<> &IRB);
  void instrumentSyclStaticLocalMemory(IRBuilder<> &IRB);
  void initializeRetVecMap(Function *F);
  void initializeKernelCallerMap(Function *F);

  void instrumentGlobals(IRBuilder<> &IRB, bool *CtorComdat);
  void InstrumentGlobalsCOFF(IRBuilder<> &IRB,
                             ArrayRef<GlobalVariable *> ExtendedGlobals,
                             ArrayRef<Constant *> MetadataInitializers);
  void instrumentGlobalsELF(IRBuilder<> &IRB,
                            ArrayRef<GlobalVariable *> ExtendedGlobals,
                            ArrayRef<Constant *> MetadataInitializers,
                            const std::string &UniqueModuleId);
  void InstrumentGlobalsMachO(IRBuilder<> &IRB,
                              ArrayRef<GlobalVariable *> ExtendedGlobals,
                              ArrayRef<Constant *> MetadataInitializers);
  void
  InstrumentGlobalsWithMetadataArray(IRBuilder<> &IRB,
                                     ArrayRef<GlobalVariable *> ExtendedGlobals,
                                     ArrayRef<Constant *> MetadataInitializers);

  GlobalVariable *CreateMetadataGlobal(Constant *Initializer,
                                       StringRef OriginalName);
  void SetComdatForGlobalMetadata(GlobalVariable *G, GlobalVariable *Metadata,
                                  StringRef InternalSuffix);
  Instruction *CreateAsanModuleDtor();

  const GlobalVariable *getExcludedAliasedGlobal(const GlobalAlias &GA) const;
  bool shouldInstrumentGlobal(GlobalVariable *G) const;
  bool ShouldUseMachOGlobalsSection() const;
  StringRef getGlobalMetadataSection() const;
  void poisonOneInitializer(Function &GlobalInit);
  void createInitializerPoisonCalls();
  uint64_t getMinRedzoneSizeForGlobal() const {
    return getRedzoneSizeForScale(Mapping.Scale);
  }
  uint64_t getRedzoneSizeForGlobal(uint64_t SizeInBytes) const;
  int GetAsanVersion() const;
  GlobalVariable *getOrCreateModuleName();

  Module &M;
  bool CompileKernel;
  bool InsertVersionCheck;
  bool Recover;
  bool UseGlobalsGC;
  bool UsePrivateAlias;
  bool UseOdrIndicator;
  bool UseCtorComdat;
  AsanDtorKind DestructorKind;
  AsanCtorKind ConstructorKind;
  Type *IntptrTy;
  PointerType *PtrTy;
  LLVMContext *C;
  Triple TargetTriple;
  ShadowMapping Mapping;
  FunctionCallee AsanPoisonGlobals;
  FunctionCallee AsanUnpoisonGlobals;
  FunctionCallee AsanRegisterGlobals;
  FunctionCallee AsanUnregisterGlobals;
  FunctionCallee AsanRegisterImageGlobals;
  FunctionCallee AsanUnregisterImageGlobals;
  FunctionCallee AsanRegisterElfGlobals;
  FunctionCallee AsanUnregisterElfGlobals;
  FunctionCallee AsanSetShadowStaticLocalFunc;
  FunctionCallee AsanUnpoisonShadowStaticLocalFunc;

  Function *AsanCtorFunction = nullptr;
  Function *AsanDtorFunction = nullptr;
  GlobalVariable *ModuleName = nullptr;

  DenseMap<Function *, SmallVector<Instruction *, 8>> KernelToRetVecMap;
  DenseMap<Function *, DenseSet<Function *>> FuncToKernelCallerMap;
};

// Stack poisoning does not play well with exception handling.
// When an exception is thrown, we essentially bypass the code
// that unpoisones the stack. This is why the run-time library has
// to intercept __cxa_throw (as well as longjmp, etc) and unpoison the entire
// stack in the interceptor. This however does not work inside the
// actual function which catches the exception. Most likely because the
// compiler hoists the load of the shadow value somewhere too high.
// This causes asan to report a non-existing bug on 453.povray.
// It sounds like an LLVM bug.
struct FunctionStackPoisoner : public InstVisitor<FunctionStackPoisoner> {
  Function &F;
  AddressSanitizer &ASan;
  RuntimeCallInserter &RTCI;
  DIBuilder DIB;
  LLVMContext *C;
  Type *IntptrTy;
  Type *IntptrPtrTy;
  ShadowMapping Mapping;

  SmallVector<AllocaInst *, 16> AllocaVec;
  SmallVector<AllocaInst *, 16> StaticAllocasToMoveUp;
  SmallVector<Instruction *, 8> RetVec;

  FunctionCallee AsanStackMallocFunc[kMaxAsanStackMallocSizeClass + 1],
      AsanStackFreeFunc[kMaxAsanStackMallocSizeClass + 1];
  FunctionCallee AsanSetShadowFunc[0x100] = {};
  FunctionCallee AsanSetShadowPrivateFunc;
  FunctionCallee AsanPoisonStackMemoryFunc, AsanUnpoisonStackMemoryFunc;
  FunctionCallee AsanAllocaPoisonFunc, AsanAllocasUnpoisonFunc;

  // Stores a place and arguments of poisoning/unpoisoning call for alloca.
  struct AllocaPoisonCall {
    IntrinsicInst *InsBefore;
    AllocaInst *AI;
    uint64_t Size;
    bool DoPoison;
  };
  SmallVector<AllocaPoisonCall, 8> DynamicAllocaPoisonCallVec;
  SmallVector<AllocaPoisonCall, 8> StaticAllocaPoisonCallVec;
  bool HasUntracedLifetimeIntrinsic = false;

  SmallVector<AllocaInst *, 1> DynamicAllocaVec;
  SmallVector<IntrinsicInst *, 1> StackRestoreVec;
  AllocaInst *DynamicAllocaLayout = nullptr;
  IntrinsicInst *LocalEscapeCall = nullptr;

  bool HasInlineAsm = false;
  bool HasReturnsTwiceCall = false;
  bool PoisonStack;

  FunctionStackPoisoner(Function &F, AddressSanitizer &ASan,
                        RuntimeCallInserter &RTCI)
      : F(F), ASan(ASan), RTCI(RTCI),
        DIB(*F.getParent(), /*AllowUnresolved*/ false), C(ASan.C),
        IntptrTy(ASan.IntptrTy),
        IntptrPtrTy(PointerType::get(IntptrTy->getContext(), 0)),
        Mapping(ASan.Mapping),
        PoisonStack(
          F.getParent()->getTargetTriple().isSPIROrSPIRV()
          ? ClSpirOffloadPrivates
          : (ClStack &&
             !F.getParent()->getTargetTriple().isAMDGPU())) {}

  bool runOnFunction() {
    if (!PoisonStack)
      return false;

    if (ClRedzoneByvalArgs)
      copyArgsPassedByValToAllocas();

    // Collect alloca, ret, lifetime instructions etc.
    for (BasicBlock *BB : depth_first(&F.getEntryBlock())) visit(*BB);

    if (AllocaVec.empty() && DynamicAllocaVec.empty()) return false;

    initializeCallbacks(*F.getParent());

    if (HasUntracedLifetimeIntrinsic) {
      // If there are lifetime intrinsics which couldn't be traced back to an
      // alloca, we may not know exactly when a variable enters scope, and
      // therefore should "fail safe" by not poisoning them.
      StaticAllocaPoisonCallVec.clear();
      DynamicAllocaPoisonCallVec.clear();
    }

    processDynamicAllocas();
    processStaticAllocas();

    if (ClDebugStack) {
      LLVM_DEBUG(dbgs() << F);
    }
    return true;
  }

  // Arguments marked with the "byval" attribute are implicitly copied without
  // using an alloca instruction.  To produce redzones for those arguments, we
  // copy them a second time into memory allocated with an alloca instruction.
  void copyArgsPassedByValToAllocas();

  // Finds all Alloca instructions and puts
  // poisoned red zones around all of them.
  // Then unpoison everything back before the function returns.
  void processStaticAllocas();
  void processDynamicAllocas();

  void createDynamicAllocasInitStorage();

  // ----------------------- Visitors.
  /// Collect all Ret instructions, or the musttail call instruction if it
  /// precedes the return instruction.
  void visitReturnInst(ReturnInst &RI) {
    if (CallInst *CI = RI.getParent()->getTerminatingMustTailCall())
      RetVec.push_back(CI);
    else
      RetVec.push_back(&RI);
  }

  /// Collect all Resume instructions.
  void visitResumeInst(ResumeInst &RI) { RetVec.push_back(&RI); }

  /// Collect all CatchReturnInst instructions.
  void visitCleanupReturnInst(CleanupReturnInst &CRI) { RetVec.push_back(&CRI); }

  void unpoisonDynamicAllocasBeforeInst(Instruction *InstBefore,
                                        Value *SavedStack) {
    IRBuilder<> IRB(InstBefore);
    Value *DynamicAreaPtr = IRB.CreatePtrToInt(SavedStack, IntptrTy);
    // When we insert _asan_allocas_unpoison before @llvm.stackrestore, we
    // need to adjust extracted SP to compute the address of the most recent
    // alloca. We have a special @llvm.get.dynamic.area.offset intrinsic for
    // this purpose.
    if (!isa<ReturnInst>(InstBefore)) {
      Value *DynamicAreaOffset = IRB.CreateIntrinsic(
          Intrinsic::get_dynamic_area_offset, {IntptrTy}, {});

      DynamicAreaPtr = IRB.CreateAdd(IRB.CreatePtrToInt(SavedStack, IntptrTy),
                                     DynamicAreaOffset);
    }

    RTCI.createRuntimeCall(
        IRB, AsanAllocasUnpoisonFunc,
        {IRB.CreateLoad(IntptrTy, DynamicAllocaLayout), DynamicAreaPtr});
  }

  // Unpoison dynamic allocas redzones.
  void unpoisonDynamicAllocas() {
    for (Instruction *Ret : RetVec)
      unpoisonDynamicAllocasBeforeInst(Ret, DynamicAllocaLayout);

    for (Instruction *StackRestoreInst : StackRestoreVec)
      unpoisonDynamicAllocasBeforeInst(StackRestoreInst,
                                       StackRestoreInst->getOperand(0));
  }

  // Deploy and poison redzones around dynamic alloca call. To do this, we
  // should replace this call with another one with changed parameters and
  // replace all its uses with new address, so
  //   addr = alloca type, old_size, align
  // is replaced by
  //   new_size = (old_size + additional_size) * sizeof(type)
  //   tmp = alloca i8, new_size, max(align, 32)
  //   addr = tmp + 32 (first 32 bytes are for the left redzone).
  // Additional_size is added to make new memory allocation contain not only
  // requested memory, but also left, partial and right redzones.
  void handleDynamicAllocaCall(AllocaInst *AI);

  /// Collect Alloca instructions we want (and can) handle.
  void visitAllocaInst(AllocaInst &AI) {
    // FIXME: Handle scalable vectors instead of ignoring them.
    const Type *AllocaType = AI.getAllocatedType();
    const auto *STy = dyn_cast<StructType>(AllocaType);
    if (!ASan.isInterestingAlloca(AI) || isa<ScalableVectorType>(AllocaType) ||
        (STy && STy->containsHomogeneousScalableVectorTypes())) {
      if (AI.isStaticAlloca()) {
        // Skip over allocas that are present *before* the first instrumented
        // alloca, we don't want to move those around.
        if (AllocaVec.empty())
          return;

        StaticAllocasToMoveUp.push_back(&AI);
      }
      return;
    }

    if (!AI.isStaticAlloca())
      DynamicAllocaVec.push_back(&AI);
    else
      AllocaVec.push_back(&AI);
  }

  /// Collect lifetime intrinsic calls to check for use-after-scope
  /// errors.
  void visitIntrinsicInst(IntrinsicInst &II) {
    Intrinsic::ID ID = II.getIntrinsicID();
    if (ID == Intrinsic::stackrestore) StackRestoreVec.push_back(&II);
    if (ID == Intrinsic::localescape) LocalEscapeCall = &II;
    if (!ASan.UseAfterScope)
      return;
    if (!II.isLifetimeStartOrEnd())
      return;
    // Found lifetime intrinsic, add ASan instrumentation if necessary.
    auto *Size = cast<ConstantInt>(II.getArgOperand(0));
    // If size argument is undefined, don't do anything.
    if (Size->isMinusOne()) return;
    // Check that size doesn't saturate uint64_t and can
    // be stored in IntptrTy.
    const uint64_t SizeValue = Size->getValue().getLimitedValue();
    if (SizeValue == ~0ULL ||
        !ConstantInt::isValueValidForType(IntptrTy, SizeValue))
      return;
    // Find alloca instruction that corresponds to llvm.lifetime argument.
    // Currently we can only handle lifetime markers pointing to the
    // beginning of the alloca.
    AllocaInst *AI = findAllocaForValue(II.getArgOperand(1), true);
    if (!AI) {
      HasUntracedLifetimeIntrinsic = true;
      return;
    }
    // We're interested only in allocas we can handle.
    if (!ASan.isInterestingAlloca(*AI))
      return;
    bool DoPoison = (ID == Intrinsic::lifetime_end);
    AllocaPoisonCall APC = {&II, AI, SizeValue, DoPoison};
    if (AI->isStaticAlloca())
      StaticAllocaPoisonCallVec.push_back(APC);
    else if (ClInstrumentDynamicAllocas)
      DynamicAllocaPoisonCallVec.push_back(APC);
  }

  void visitCallBase(CallBase &CB) {
    if (CallInst *CI = dyn_cast<CallInst>(&CB)) {
      HasInlineAsm |= CI->isInlineAsm() && &CB != ASan.LocalDynamicShadow;
      HasReturnsTwiceCall |= CI->canReturnTwice();
    }
  }

  // ---------------------- Helpers.
  void initializeCallbacks(Module &M);

  // Copies bytes from ShadowBytes into shadow memory for indexes where
  // ShadowMask is not zero. If ShadowMask[i] is zero, we assume that
  // ShadowBytes[i] is constantly zero and doesn't need to be overwritten.
  void copyToShadow(ArrayRef<uint8_t> ShadowMask, ArrayRef<uint8_t> ShadowBytes,
                    IRBuilder<> &IRB, Value *ShadowBase,
                    bool ForceOutline = false);
  void copyToShadow(ArrayRef<uint8_t> ShadowMask, ArrayRef<uint8_t> ShadowBytes,
                    size_t Begin, size_t End, IRBuilder<> &IRB,
                    Value *ShadowBase, bool ForceOutline = false);
  void copyToShadowInline(ArrayRef<uint8_t> ShadowMask,
                          ArrayRef<uint8_t> ShadowBytes, size_t Begin,
                          size_t End, IRBuilder<> &IRB, Value *ShadowBase);

  void poisonAlloca(Value *V, uint64_t Size, IRBuilder<> &IRB, bool DoPoison);

  Value *createAllocaForLayout(IRBuilder<> &IRB, const ASanStackFrameLayout &L,
                               bool Dynamic);
  PHINode *createPHI(IRBuilder<> &IRB, Value *Cond, Value *ValueIfTrue,
                     Instruction *ThenTerm, Value *ValueIfFalse);
};

class AddressSanitizerOnSpirv {
public:
  explicit AddressSanitizerOnSpirv(Module *M) : M(M), C(&M->getContext()) {
    const DataLayout &DL = M->getDataLayout();
    IntptrTy = DL.getIntPtrType(*C);
    Int32Ty = Type::getInt32Ty(*C);
  }

  void initializeCallbacks() {
    IRBuilder<> IRB(*C);

    // __msan_set_private_base(
    //   as(0) void *  ptr
    // )
    AsanSetPrivateBaseFunc =
        M->getOrInsertFunction("__asan_set_private_base", IRB.getVoidTy(),
                               PointerType::get(*C, kSpirOffloadPrivateAS));
  }

  void instrumentPrivateBase(Function &F) {
    if (!ClSpirOffloadPrivates)
      return;

    IRBuilder<> IRB(&F.getEntryBlock().front());
    AllocaInst *PrivateBase = IRB.CreateAlloca(
        IRB.getInt8Ty(), ConstantInt::get(Int32Ty, 1), "__private_base");
    IRB.CreateCall(AsanSetPrivateBaseFunc, {PrivateBase});
  }

private:
  Module *M;
  LLVMContext *C;

  Type *IntptrTy;
  Type *Int32Ty;

  FunctionCallee AsanSetPrivateBaseFunc;
};

} // end anonymous namespace

static StringMap<GlobalVariable *> GlobalStringMap;

static GlobalVariable *GetOrCreateGlobalString(Module &M, StringRef Name,
                                               StringRef Value,
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

// Append a new argument "__asan_launch" to user's spir_kernels
static void ExtendSpirKernelArgs(Module &M, FunctionAnalysisManager &FAM,
                                 bool HasESIMD) {
  SmallVector<Function *> SpirFixupKernels;
  SmallVector<Constant *, 8> SpirKernelsMetadata;
  SmallVector<uint8_t, 256> KernelNamesBytes;

  const auto &DL = M.getDataLayout();
  Type *IntptrTy = DL.getIntPtrType(M.getContext());

  // SpirKernelsMetadata only saves fixed kernels, and is described by
  // following structure:
  //  uptr unmangled_kernel_name
  //  uptr unmangled_kernel_name_size
  StructType *StructTy = StructType::get(IntptrTy, IntptrTy);

  if (!HasESIMD)
    for (Function &F : M) {
      if (!F.hasFnAttribute(Attribute::SanitizeAddress) ||
          F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
        continue;

      if (F.getName().contains("__sycl_service_kernel__")) {
        F.addFnAttr(Attribute::DisableSanitizerInstrumentation);
        continue;
      }

      // Skip referenced-indirectly function as we insert access to shared
      // local memory (SLM) __AsanLaunchInfo and access to SLM in
      // referenced-indirectly function isn't supported yet in
      // intel-graphics-compiler.
      if (F.hasFnAttribute("referenced-indirectly")) {
        F.addFnAttr(Attribute::DisableSanitizerInstrumentation);
        continue;
      }

      if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
        continue;

      SpirFixupKernels.emplace_back(&F);

      auto KernelName = F.getName();
      KernelNamesBytes.append(KernelName.begin(), KernelName.end());
      auto *KernelNameGV = GetOrCreateGlobalString(
          M, "__asan_kernel", KernelName, kSpirOffloadConstantAS);
      SpirKernelsMetadata.emplace_back(ConstantStruct::get(
          StructTy, ConstantExpr::getPointerCast(KernelNameGV, IntptrTy),
          ConstantInt::get(IntptrTy, KernelName.size())));
    }

  // Create global variable to record spirv kernels' information
  ArrayType *ArrayTy = ArrayType::get(StructTy, SpirKernelsMetadata.size());
  Constant *MetadataInitializer =
      ConstantArray::get(ArrayTy, SpirKernelsMetadata);
  GlobalVariable *AsanSpirKernelMetadata = new GlobalVariable(
      M, MetadataInitializer->getType(), false, GlobalValue::AppendingLinkage,
      MetadataInitializer, "__AsanKernelMetadata", nullptr,
      GlobalValue::NotThreadLocal, 1);
  AsanSpirKernelMetadata->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
  // Add device global attributes
  AsanSpirKernelMetadata->addAttribute(
      "sycl-device-global-size", std::to_string(DL.getTypeAllocSize(ArrayTy)));
  AsanSpirKernelMetadata->addAttribute("sycl-device-image-scope");
  AsanSpirKernelMetadata->addAttribute("sycl-host-access", "0"); // read only
  AsanSpirKernelMetadata->addAttribute(
      "sycl-unique-id",
      computeKernelMetadataUniqueId("__AsanKernelMetadata", KernelNamesBytes));
  AsanSpirKernelMetadata->setDSOLocal(true);

  // Handle SpirFixupKernels
  SmallVector<std::pair<Function *, Function *>> SpirFuncs;

  for (auto *F : SpirFixupKernels) {
    SmallVector<Type *, 16> Types;
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I) {
      Types.push_back(I->getType());
    }

    // New argument: uintptr_t as(1)*, which is allocated in shared USM buffer
    Types.push_back(PointerType::get(M.getContext(), kSpirOffloadGlobalAS));

    FunctionType *NewFTy = FunctionType::get(F->getReturnType(), Types, false);

    std::string OrigFuncName = F->getName().str();
    F->setName(OrigFuncName + "_del");

    Function *NewF =
        Function::Create(NewFTy, F->getLinkage(), OrigFuncName, F->getParent());
    NewF->copyAttributesFrom(F);
    NewF->copyMetadata(F, 0);
    NewF->setCallingConv(F->getCallingConv());
    NewF->setDSOLocal(F->isDSOLocal());

    // Set original arguments' names.
    Function::arg_iterator NewI = NewF->arg_begin();
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I, ++NewI) {
      NewI->setName(I->getName());
    }
    // New argument name
    NewI->setName("__asan_launch");
    NewI->addAttr(Attribute::NoUndef);

    NewF->splice(NewF->begin(), F);
    assert(F->isDeclaration() &&
           "splice does not work, original function body is not empty!");

    NewF->setSubprogram(F->getSubprogram());

    NewF->setComdat(F->getComdat());
    F->setComdat(nullptr);

    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
                                NI = NewF->arg_begin();
         I != E; ++I, ++NI) {
      I->replaceAllUsesWith(&*NI);
    }

    // Fixup metadata
    IRBuilder<> Builder(M.getContext());

    auto FixupMetadata = [&NewF](StringRef MDName, Metadata *NewV) {
      auto *Node = NewF->getMetadata(MDName);
      if (!Node)
        return;
      SmallVector<Metadata *, 8> NewMD(Node->operands());
      NewMD.emplace_back(NewV);
      NewF->setMetadata(MDName, llvm::MDNode::get(NewF->getContext(), NewMD));
    };

    FixupMetadata("kernel_arg_buffer_location",
                  ConstantAsMetadata::get(Builder.getInt32(-1)));
    FixupMetadata("kernel_arg_runtime_aligned",
                  ConstantAsMetadata::get(Builder.getFalse()));
    FixupMetadata("kernel_arg_exclusive_ptr",
                  ConstantAsMetadata::get(Builder.getFalse()));

    FixupMetadata(
        "kernel_arg_addr_space",
        ConstantAsMetadata::get(Builder.getInt32(kSpirOffloadGlobalAS)));
    FixupMetadata("kernel_arg_access_qual",
                  MDString::get(M.getContext(), "read_write"));
    FixupMetadata("kernel_arg_type", MDString::get(M.getContext(), "void*"));
    FixupMetadata("kernel_arg_base_type",
                  MDString::get(M.getContext(), "void*"));
    FixupMetadata("kernel_arg_type_qual", MDString::get(M.getContext(), ""));
    FixupMetadata("kernel_arg_accessor_ptr",
                  ConstantAsMetadata::get(Builder.getFalse()));

    SpirFuncs.emplace_back(F, NewF);
  }

  // Fixup all users
  for (auto &[F, NewF] : SpirFuncs) {
    SmallVector<User *, 16> Users(F->users());
    for (User *U : Users) {
      if (auto *CI = dyn_cast<CallInst>(U)) {
        if (CI->getCalledOperand() != F)
          continue;

        // Append "launch_info" into arguments of call instruction
        SmallVector<Value *, 16> Args(CI->args());
        // "launch_info" is the last argument of kernel
        auto *CurF = CI->getFunction();
        Args.push_back(CurF->getArg(CurF->arg_size() - 1));

        CallInst *NewCI =
            CallInst::Create(NewF, Args, CI->getName(), CI->getIterator());
        NewCI->setCallingConv(CI->getCallingConv());
        NewCI->setAttributes(CI->getAttributes());
        NewCI->setDebugLoc(CI->getDebugLoc());
        CI->replaceAllUsesWith(NewCI);
        CI->eraseFromParent();
      }
    }
    // Replace old Func to new Func in metadata
    ValueAsMetadata::handleRAUW(F, NewF);
    F->eraseFromParent();
  }
}

void AddressSanitizerPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<AddressSanitizerPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << '<';
  if (Options.CompileKernel)
    OS << "kernel;";
  if (Options.UseAfterScope)
    OS << "use-after-scope";
  OS << '>';
}

AddressSanitizerPass::AddressSanitizerPass(
    const AddressSanitizerOptions &Options, bool UseGlobalGC,
    bool UseOdrIndicator, AsanDtorKind DestructorKind,
    AsanCtorKind ConstructorKind)
    : Options(Options), UseGlobalGC(UseGlobalGC),
      UseOdrIndicator(UseOdrIndicator), DestructorKind(DestructorKind),
      ConstructorKind(ConstructorKind) {}

PreservedAnalyses AddressSanitizerPass::run(Module &M,
                                            ModuleAnalysisManager &MAM) {
  // Return early if nosanitize_address module flag is present for the module.
  // This implies that asan pass has already run before.
  if (checkIfAlreadyInstrumented(M, "nosanitize_address"))
    return PreservedAnalyses::all();

  ModuleAddressSanitizer ModuleSanitizer(
      M, Options.InsertVersionCheck, Options.CompileKernel, Options.Recover,
      UseGlobalGC, UseOdrIndicator, DestructorKind, ConstructorKind);
  bool Modified = false;
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  const StackSafetyGlobalInfo *const SSGI =
      ClUseStackSafety ? &MAM.getResult<StackSafetyGlobalAnalysis>(M) : nullptr;

  std::optional<AddressSanitizerOnSpirv> AsanSpirv;
  if (Triple(M.getTargetTriple()).isSPIROrSPIRV()) {
    AsanSpirv = AddressSanitizerOnSpirv(&M);
    AsanSpirv->initializeCallbacks();

    // FIXME: W/A skip instrumentation if this module has ESIMD
    bool HasESIMD = false;
    for (auto &F : M) {
      if (F.hasMetadata("sycl_explicit_simd")) {
        HasESIMD = true;
        break;
      }
    }

    // Make sure "__AsanKernelMetadata" always exists
    ExtendSpirKernelArgs(M, FAM, HasESIMD);
    Modified = true;

    if (HasESIMD) {
      GlobalStringMap.clear();
      return PreservedAnalyses::none();
    }
  }

  for (Function &F : M) {
    if (F.empty())
      continue;
    if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage)
      continue;
    if (!ClDebugFunc.empty() && ClDebugFunc == F.getName())
      continue;
    if (F.getName().starts_with("__asan_"))
      continue;
    if (F.isPresplitCoroutine())
      continue;
    AddressSanitizer FunctionSanitizer(
        M, SSGI, Options.InstrumentationWithCallsThreshold,
        Options.MaxInlinePoisoningSize, Options.CompileKernel, Options.Recover,
        Options.UseAfterScope, Options.UseAfterReturn);
    const TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    Modified |= FunctionSanitizer.instrumentFunction(F, &TLI);
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      AsanSpirv->instrumentPrivateBase(F);
      // Instrument __AsanLaunchInfo should always be the last step
      FunctionSanitizer.instrumentInitAsanLaunchInfo(F, &TLI);
    }
  }
  Modified |= ModuleSanitizer.instrumentModule();

  GlobalStringMap.clear();

  if (!Modified)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = PreservedAnalyses::none();
  // GlobalsAA is considered stateless and does not get invalidated unless
  // explicitly invalidated; PreservedAnalyses::none() is not enough. Sanitizers
  // make changes that require GlobalsAA to be invalidated.
  PA.abandon<GlobalsAA>();
  return PA;
}

static size_t TypeStoreSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = llvm::countr_zero(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
}

/// Check if \p G has been created by a trusted compiler pass.
static bool GlobalWasGeneratedByCompiler(GlobalVariable *G) {
  // Do not instrument @llvm.global_ctors, @llvm.used, etc.
  if (G->getName().starts_with("llvm.") ||
      // Do not instrument gcov counter arrays.
      G->getName().starts_with("__llvm_gcov_ctr") ||
      // Do not instrument rtti proxy symbols for function sanitizer.
      G->getName().starts_with("__llvm_rtti_proxy"))
    return true;

  // Do not instrument asan globals.
  if (G->getName().starts_with(kAsanGenPrefix) ||
      G->getName().starts_with(kSanCovGenPrefix) ||
      G->getName().starts_with(kODRGenPrefix))
    return true;

  return false;
}

static bool isUnsupportedAMDGPUAddrspace(Value *Addr) {
  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  unsigned int AddrSpace = PtrTy->getPointerAddressSpace();
  if (AddrSpace == 3 || AddrSpace == 5)
    return true;
  return false;
}

static bool isUnsupportedDeviceGlobal(GlobalVariable *G) {
  // Non image scope device globals are implemented by device USM, and the
  // out-of-bounds check for them will be done by sanitizer USM part. So we
  // exclude them here.
  if (!G->hasAttribute("sycl-device-image-scope"))
    return true;

  // Skip instrumenting on "__AsanKernelMetadata" etc.
  if (G->getName().starts_with("__Asan"))
    return true;

  if (G->getAddressSpace() == kSpirOffloadLocalAS)
    return !ClSpirOffloadLocals;

  Attribute Attr = G->getAttribute("sycl-device-image-scope");
  return (!Attr.isStringAttribute() || Attr.getValueAsString() == "false");
}

static bool isUnsupportedSPIRAccess(Value *Addr, Instruction *Inst) {
  // Skip SPIR-V built-in varibles
  auto *OrigValue = Addr->stripInBoundsOffsets();
  if (OrigValue->getName().starts_with("__spirv_BuiltIn"))
    return true;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(OrigValue);
  if (GV && isUnsupportedDeviceGlobal(GV))
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

  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  switch (PtrTy->getPointerAddressSpace()) {
  case kSpirOffloadPrivateAS: {
    if (!ClSpirOffloadPrivates)
      return true;
    // Skip kernel arguments
    return Inst->getFunction()->getCallingConv() == CallingConv::SPIR_KERNEL &&
           isa<Argument>(Addr);
  }
  case kSpirOffloadGlobalAS: {
    return !ClSpirOffloadGlobals;
  }
  case kSpirOffloadLocalAS: {
    if (!ClSpirOffloadLocals)
      return true;
    return Addr->getName().starts_with("__Asan");
  }
  case kSpirOffloadGenericAS: {
    return !ClSpirOffloadGenerics;
  }
  }
  return true;
}

void AddressSanitizer::AppendDebugInfoToArgs(Instruction *InsertBefore,
                                             Value *Addr,
                                             SmallVectorImpl<Value *> &Args) {
  auto *M = InsertBefore->getModule();
  auto &C = InsertBefore->getContext();
  auto &Loc = InsertBefore->getDebugLoc();

  // SPIR constant address space
  PointerType *ConstASPtrTy = PointerType::get(C, kSpirOffloadConstantAS);

  // File & Line
  if (Loc) {
    llvm::SmallString<128> Source = Loc->getDirectory();
    sys::path::append(Source, Loc->getFilename());
    auto *FileNameGV = GetOrCreateGlobalString(*M, "__asan_file", Source,
                                               kSpirOffloadConstantAS);
    Args.push_back(ConstantExpr::getPointerCast(FileNameGV, ConstASPtrTy));
    Args.push_back(ConstantInt::get(Type::getInt32Ty(C), Loc.getLine()));
  } else {
    Args.push_back(ConstantPointerNull::get(ConstASPtrTy));
    Args.push_back(ConstantInt::get(Type::getInt32Ty(C), 0));
  }

  // Function
  auto FuncName = InsertBefore->getFunction()->getName();
  auto *FuncNameGV = GetOrCreateGlobalString(
      *M, "__asan_func", demangle(FuncName), kSpirOffloadConstantAS);
  Args.push_back(ConstantExpr::getPointerCast(FuncNameGV, ConstASPtrTy));
}

Value *AddressSanitizer::memToShadow(Value *Shadow, IRBuilder<> &IRB,
                                     uint32_t AddressSpace) {
  if (TargetTriple.isSPIROrSPIRV()) {
    return IRB.CreateCall(
        AsanMemToShadow,
        {Shadow, ConstantInt::get(IRB.getInt32Ty(), AddressSpace)},
        "shadow_ptr");
  }
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, Mapping.Scale);
  if (Mapping.Offset == 0) return Shadow;
  // (Shadow >> scale) | offset
  Value *ShadowBase;
  if (LocalDynamicShadow)
    ShadowBase = LocalDynamicShadow;
  else
    ShadowBase = ConstantInt::get(IntptrTy, Mapping.Offset);
  if (Mapping.OrShadowOffset)
    return IRB.CreateOr(Shadow, ShadowBase);
  else
    return IRB.CreateAdd(Shadow, ShadowBase);
}

// Instument dynamic local memory
bool AddressSanitizer::instrumentSyclDynamicLocalMemory(Function &F) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());

  SmallVector<Argument *> LocalArgs;
  for (auto &Arg : F.args()) {
    Type *PtrTy = dyn_cast<PointerType>(Arg.getType()->getScalarType());
    if (PtrTy && PtrTy->getPointerAddressSpace() == kSpirOffloadLocalAS)
      LocalArgs.push_back(&Arg);
  }

  if (LocalArgs.empty())
    return false;

  AllocaInst *ArgsArray = IRB.CreateAlloca(
      IntptrTy, ConstantInt::get(Int32Ty, LocalArgs.size()), "local_args");
  for (size_t i = 0; i < LocalArgs.size(); i++) {
    auto *StoreDest =
        IRB.CreateGEP(IntptrTy, ArgsArray, ConstantInt::get(Int32Ty, i));
    IRB.CreateStore(IRB.CreatePointerCast(LocalArgs[i], IntptrTy), StoreDest);
  }

  auto *ArgsArrayAddr = IRB.CreatePointerCast(ArgsArray, IntptrTy);
  IRB.CreateCall(AsanSetShadowDynamicLocalFunc,
                 {ArgsArrayAddr, ConstantInt::get(Int32Ty, LocalArgs.size())});

  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
        IRBuilder<> IRBRet(Ret);
        IRBRet.CreateCall(
            AsanUnpoisonShadowDynamicLocalFunc,
            {ArgsArrayAddr, ConstantInt::get(Int32Ty, LocalArgs.size())});
      }

  return true;
}

// Initialize the value of local memory "__AsanLaunchInfo",  store
// "__asan_launch" if it's an extended kernel, and store 0 if not
void AddressSanitizer::instrumentInitAsanLaunchInfo(
    Function &F, const TargetLibraryInfo *TLI) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  if (F.arg_size()) {
    auto *LastArg = F.getArg(F.arg_size() - 1);
    if (LastArg->getName() == "__asan_launch") {
      IRB.CreateStore(LastArg, AsanLaunchInfo);
      return;
    }
  }
  // FIXME: if the initial value of "__AsanLaunchInfo" is zero, we'll not need
  // this step
  initializeCallbacks(TLI);
  IRB.CreateStore(
      ConstantPointerNull::get(PointerType::get(*C, kSpirOffloadGlobalAS)),
      AsanLaunchInfo);
}

// Instrument memset/memmove/memcpy
void AddressSanitizer::instrumentMemIntrinsic(MemIntrinsic *MI,
                                              RuntimeCallInserter &RTCI) {
  InstrumentationIRBuilder IRB(MI);
  if (isa<MemTransferInst>(MI)) {
    RTCI.createRuntimeCall(
        IRB, isa<MemMoveInst>(MI) ? AsanMemmove : AsanMemcpy,
        {IRB.CreateAddrSpaceCast(MI->getOperand(0), PtrTy),
         IRB.CreateAddrSpaceCast(MI->getOperand(1), PtrTy),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else if (isa<MemSetInst>(MI)) {
    RTCI.createRuntimeCall(
        IRB, AsanMemset,
        {IRB.CreateAddrSpaceCast(MI->getOperand(0), PtrTy),
         IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  }
  MI->eraseFromParent();
}

/// Check if we want (and can) handle this alloca.
bool AddressSanitizer::isInterestingAlloca(const AllocaInst &AI) {
  auto [It, Inserted] = ProcessedAllocas.try_emplace(&AI);

  if (!Inserted)
    return It->getSecond();

  bool IsInteresting =
      (AI.getAllocatedType()->isSized() &&
       // alloca() may be called with 0 size, ignore it.
       ((!AI.isStaticAlloca()) || !getAllocaSizeInBytes(AI).isZero()) &&
       // We are only interested in allocas not promotable to registers.
       // Promotable allocas are common under -O0.
       (!ClSkipPromotableAllocas || !isAllocaPromotable(&AI)) &&
       // inalloca allocas are not treated as static, and we don't want
       // dynamic alloca instrumentation for them as well.
       !AI.isUsedWithInAlloca() &&
       // swifterror allocas are register promoted by ISel
       !AI.isSwiftError() &&
       // safe allocas are not interesting
       !(SSGI && SSGI->isSafe(AI)) &&
       // ignore alloc contains target ext type since we can't know exactly what
       // size it is.
       !getTargetExtType(AI.getAllocatedType()));

  It->second = IsInteresting;
  return IsInteresting;
}

bool AddressSanitizer::ignoreAccess(Instruction *Inst, Value *Ptr) {
  // SPIR has its own rules to filter the instrument accesses
  if (TargetTriple.isSPIROrSPIRV()) {
    if (isUnsupportedSPIRAccess(Ptr, Inst))
      return true;
  } else {
    // Instrument accesses from different address spaces only for AMDGPU.
    Type *PtrTy = cast<PointerType>(Ptr->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0 &&
        !(TargetTriple.isAMDGPU() && !isUnsupportedAMDGPUAddrspace(Ptr))) {
      return true;
    }
  }

  // Ignore swifterror addresses.
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Ptr->isSwiftError())
    return true;

  // Treat memory accesses to promotable allocas as non-interesting since they
  // will not cause memory violations. This greatly speeds up the instrumented
  // executable at -O0.
  if (auto AI = dyn_cast_or_null<AllocaInst>(Ptr))
    if (ClSkipPromotableAllocas && !isInterestingAlloca(*AI))
      return true;

  if (SSGI != nullptr && SSGI->stackAccessIsSafe(*Inst) &&
      findAllocaForValue(Ptr))
    return true;

  return false;
}

void AddressSanitizer::getInterestingMemoryOperands(
    Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  // Do not instrument the load fetching the dynamic shadow address.
  if (LocalDynamicShadow == I)
    return;

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads || ignoreAccess(I, LI->getPointerOperand()))
      return;
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                             LI->getType(), LI->getAlign());
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites || ignoreAccess(I, SI->getPointerOperand()))
      return;
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                             SI->getValueOperand()->getType(), SI->getAlign());
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics || ignoreAccess(I, RMW->getPointerOperand()))
      return;
    Interesting.emplace_back(I, RMW->getPointerOperandIndex(), true,
                             RMW->getValOperand()->getType(), std::nullopt);
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics || ignoreAccess(I, XCHG->getPointerOperand()))
      return;
    Interesting.emplace_back(I, XCHG->getPointerOperandIndex(), true,
                             XCHG->getCompareOperand()->getType(),
                             std::nullopt);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    switch (CI->getIntrinsicID()) {
    case Intrinsic::masked_load:
    case Intrinsic::masked_store:
    case Intrinsic::masked_gather:
    case Intrinsic::masked_scatter: {
      bool IsWrite = CI->getType()->isVoidTy();
      // Masked store has an initial operand for the value.
      unsigned OpOffset = IsWrite ? 1 : 0;
      if (IsWrite ? !ClInstrumentWrites : !ClInstrumentReads)
        return;

      auto BasePtr = CI->getOperand(OpOffset);
      if (ignoreAccess(I, BasePtr))
        return;
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = Align(1);
      // Otherwise no alignment guarantees. We probably got Undef.
      if (auto *Op = dyn_cast<ConstantInt>(CI->getOperand(1 + OpOffset)))
        Alignment = Op->getMaybeAlignValue();
      Value *Mask = CI->getOperand(2 + OpOffset);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, Mask);
      break;
    }
    case Intrinsic::masked_expandload:
    case Intrinsic::masked_compressstore: {
      bool IsWrite = CI->getIntrinsicID() == Intrinsic::masked_compressstore;
      unsigned OpOffset = IsWrite ? 1 : 0;
      if (IsWrite ? !ClInstrumentWrites : !ClInstrumentReads)
        return;
      auto BasePtr = CI->getOperand(OpOffset);
      if (ignoreAccess(I, BasePtr))
        return;
      MaybeAlign Alignment = BasePtr->getPointerAlignment(*DL);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();

      IRBuilder IB(I);
      Value *Mask = CI->getOperand(1 + OpOffset);
      // Use the popcount of Mask as the effective vector length.
      Type *ExtTy = VectorType::get(IntptrTy, cast<VectorType>(Ty));
      Value *ExtMask = IB.CreateZExt(Mask, ExtTy);
      Value *EVL = IB.CreateAddReduce(ExtMask);
      Value *TrueMask = ConstantInt::get(Mask->getType(), 1);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, TrueMask,
                               EVL);
      break;
    }
    case Intrinsic::vp_load:
    case Intrinsic::vp_store:
    case Intrinsic::experimental_vp_strided_load:
    case Intrinsic::experimental_vp_strided_store: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = CI->getType()->isVoidTy();
      if (IsWrite ? !ClInstrumentWrites : !ClInstrumentReads)
        return;
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getOperand(PtrOpNo)->getPointerAlignment(*DL);
      Value *Stride = nullptr;
      if (IID == Intrinsic::experimental_vp_strided_store ||
          IID == Intrinsic::experimental_vp_strided_load) {
        Stride = VPI->getOperand(PtrOpNo + 1);
        // Use the pointer alignment as the element alignment if the stride is a
        // mutiple of the pointer alignment. Otherwise, the element alignment
        // should be Align(1).
        unsigned PointerAlign = Alignment.valueOrOne().value();
        if (!isa<ConstantInt>(Stride) ||
            cast<ConstantInt>(Stride)->getZExtValue() % PointerAlign != 0)
          Alignment = Align(1);
      }
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(), VPI->getVectorLengthParam(),
                               Stride);
      break;
    }
    case Intrinsic::vp_gather:
    case Intrinsic::vp_scatter: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = IID == Intrinsic::vp_scatter;
      if (IsWrite ? !ClInstrumentWrites : !ClInstrumentReads)
        return;
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getPointerAlignment();
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(),
                               VPI->getVectorLengthParam());
      break;
    }
    default:
      for (unsigned ArgNo = 0; ArgNo < CI->arg_size(); ArgNo++) {
        if (!ClInstrumentByval || !CI->isByValArgument(ArgNo) ||
            ignoreAccess(I, CI->getArgOperand(ArgNo)))
          continue;
        Type *Ty = CI->getParamByValType(ArgNo);
        Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
      }
    }
  }
}

static bool isPointerOperand(Value *V) {
  return V->getType()->isPointerTy() || isa<PtrToIntInst>(V);
}

// This is a rough heuristic; it may cause both false positives and
// false negatives. The proper implementation requires cooperation with
// the frontend.
static bool isInterestingPointerComparison(Instruction *I) {
  if (ICmpInst *Cmp = dyn_cast<ICmpInst>(I)) {
    if (!Cmp->isRelational())
      return false;
  } else {
    return false;
  }
  return isPointerOperand(I->getOperand(0)) &&
         isPointerOperand(I->getOperand(1));
}

// This is a rough heuristic; it may cause both false positives and
// false negatives. The proper implementation requires cooperation with
// the frontend.
static bool isInterestingPointerSubtraction(Instruction *I) {
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
    if (BO->getOpcode() != Instruction::Sub)
      return false;
  } else {
    return false;
  }
  return isPointerOperand(I->getOperand(0)) &&
         isPointerOperand(I->getOperand(1));
}

bool AddressSanitizer::GlobalIsLinkerInitialized(GlobalVariable *G) {
  // If a global variable does not have dynamic initialization we don't
  // have to instrument it.  However, if a global does not have initializer
  // at all, we assume it has dynamic initializer (in other TU).
  if (!G->hasInitializer())
    return false;

  if (G->hasSanitizerMetadata() && G->getSanitizerMetadata().IsDynInit)
    return false;

  return true;
}

void AddressSanitizer::instrumentPointerComparisonOrSubtraction(
    Instruction *I, RuntimeCallInserter &RTCI) {
  IRBuilder<> IRB(I);
  FunctionCallee F = isa<ICmpInst>(I) ? AsanPtrCmpFunction : AsanPtrSubFunction;
  Value *Param[2] = {I->getOperand(0), I->getOperand(1)};
  for (Value *&i : Param) {
    if (i->getType()->isPointerTy())
      i = IRB.CreatePointerCast(i, IntptrTy);
  }
  RTCI.createRuntimeCall(IRB, F, Param);
}

static void doInstrumentAddress(AddressSanitizer *Pass, Instruction *I,
                                Instruction *InsertBefore, Value *Addr,
                                MaybeAlign Alignment, unsigned Granularity,
                                TypeSize TypeStoreSize, bool IsWrite,
                                Value *SizeArgument, bool UseCalls,
                                uint32_t Exp, RuntimeCallInserter &RTCI) {
  // Instrument a 1-, 2-, 4-, 8-, or 16- byte access with one check
  // if the data is properly aligned.
  if (!TypeStoreSize.isScalable()) {
    const auto FixedSize = TypeStoreSize.getFixedValue();
    switch (FixedSize) {
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      if (!Alignment || *Alignment >= Granularity ||
          *Alignment >= FixedSize / 8)
        return Pass->instrumentAddress(I, InsertBefore, Addr, Alignment,
                                       FixedSize, IsWrite, nullptr, UseCalls,
                                       Exp, RTCI);
    }
  }
  Pass->instrumentUnusualSizeOrAlignment(I, InsertBefore, Addr, TypeStoreSize,
                                         IsWrite, nullptr, UseCalls, Exp, RTCI);
}

void AddressSanitizer::instrumentMaskedLoadOrStore(
    AddressSanitizer *Pass, const DataLayout &DL, Type *IntptrTy, Value *Mask,
    Value *EVL, Value *Stride, Instruction *I, Value *Addr,
    MaybeAlign Alignment, unsigned Granularity, Type *OpType, bool IsWrite,
    Value *SizeArgument, bool UseCalls, uint32_t Exp,
    RuntimeCallInserter &RTCI) {
  auto *VTy = cast<VectorType>(OpType);
  TypeSize ElemTypeSize = DL.getTypeStoreSizeInBits(VTy->getScalarType());
  auto Zero = ConstantInt::get(IntptrTy, 0);

  IRBuilder IB(I);
  Instruction *LoopInsertBefore = I;
  if (EVL) {
    // The end argument of SplitBlockAndInsertForLane is assumed bigger
    // than zero, so we should check whether EVL is zero here.
    Type *EVLType = EVL->getType();
    Value *IsEVLZero = IB.CreateICmpNE(EVL, ConstantInt::get(EVLType, 0));
    LoopInsertBefore = SplitBlockAndInsertIfThen(IsEVLZero, I, false);
    IB.SetInsertPoint(LoopInsertBefore);
    // Cast EVL to IntptrTy.
    EVL = IB.CreateZExtOrTrunc(EVL, IntptrTy);
    // To avoid undefined behavior for extracting with out of range index, use
    // the minimum of evl and element count as trip count.
    Value *EC = IB.CreateElementCount(IntptrTy, VTy->getElementCount());
    EVL = IB.CreateBinaryIntrinsic(Intrinsic::umin, EVL, EC);
  } else {
    EVL = IB.CreateElementCount(IntptrTy, VTy->getElementCount());
  }

  // Cast Stride to IntptrTy.
  if (Stride)
    Stride = IB.CreateZExtOrTrunc(Stride, IntptrTy);

  SplitBlockAndInsertForEachLane(EVL, LoopInsertBefore->getIterator(),
                                 [&](IRBuilderBase &IRB, Value *Index) {
    Value *MaskElem = IRB.CreateExtractElement(Mask, Index);
    if (auto *MaskElemC = dyn_cast<ConstantInt>(MaskElem)) {
      if (MaskElemC->isZero())
        // No check
        return;
      // Unconditional check
    } else {
      // Conditional check
      Instruction *ThenTerm = SplitBlockAndInsertIfThen(
          MaskElem, &*IRB.GetInsertPoint(), false);
      IRB.SetInsertPoint(ThenTerm);
    }

    Value *InstrumentedAddress;
    if (isa<VectorType>(Addr->getType())) {
      assert(
          cast<VectorType>(Addr->getType())->getElementType()->isPointerTy() &&
          "Expected vector of pointer.");
      InstrumentedAddress = IRB.CreateExtractElement(Addr, Index);
    } else if (Stride) {
      Index = IRB.CreateMul(Index, Stride);
      InstrumentedAddress = IRB.CreatePtrAdd(Addr, Index);
    } else {
      InstrumentedAddress = IRB.CreateGEP(VTy, Addr, {Zero, Index});
    }
    doInstrumentAddress(Pass, I, &*IRB.GetInsertPoint(), InstrumentedAddress,
                        Alignment, Granularity, ElemTypeSize, IsWrite,
                        SizeArgument, UseCalls, Exp, RTCI);
  });
}

void AddressSanitizer::instrumentMop(ObjectSizeOffsetVisitor &ObjSizeVis,
                                     InterestingMemoryOperand &O, bool UseCalls,
                                     const DataLayout &DL,
                                     RuntimeCallInserter &RTCI) {
  Value *Addr = O.getPtr();

  // Optimization experiments.
  // The experiments can be used to evaluate potential optimizations that remove
  // instrumentation (assess false negatives). Instead of completely removing
  // some instrumentation, you set Exp to a non-zero value (mask of optimization
  // experiments that want to remove instrumentation of this instruction).
  // If Exp is non-zero, this pass will emit special calls into runtime
  // (e.g. __asan_report_exp_load1 instead of __asan_report_load1). These calls
  // make runtime terminate the program in a special way (with a different
  // exit status). Then you run the new compiler on a buggy corpus, collect
  // the special terminations (ideally, you don't see them at all -- no false
  // negatives) and make the decision on the optimization.
  uint32_t Exp = ClForceExperiment;

  if (ClOpt && ClOptGlobals) {
    // If initialization order checking is disabled, a simple access to a
    // dynamically initialized global is always valid.
    GlobalVariable *G = dyn_cast<GlobalVariable>(getUnderlyingObject(Addr));
    if (G && (!ClInitializers || GlobalIsLinkerInitialized(G)) &&
        isSafeAccess(ObjSizeVis, Addr, O.TypeStoreSize)) {
      NumOptimizedAccessesToGlobalVar++;
      return;
    }
  }

  if (ClOpt && ClOptStack) {
    // A direct inbounds access to a stack variable is always valid.
    if (isa<AllocaInst>(getUnderlyingObject(Addr)) &&
        isSafeAccess(ObjSizeVis, Addr, O.TypeStoreSize)) {
      NumOptimizedAccessesToStackVar++;
      return;
    }
  }

  if (O.IsWrite)
    NumInstrumentedWrites++;
  else
    NumInstrumentedReads++;

  unsigned Granularity = 1 << Mapping.Scale;
  if (O.MaybeMask) {
    instrumentMaskedLoadOrStore(this, DL, IntptrTy, O.MaybeMask, O.MaybeEVL,
                                O.MaybeStride, O.getInsn(), Addr, O.Alignment,
                                Granularity, O.OpType, O.IsWrite, nullptr,
                                UseCalls, Exp, RTCI);
  } else {
    doInstrumentAddress(this, O.getInsn(), O.getInsn(), Addr, O.Alignment,
                        Granularity, O.TypeStoreSize, O.IsWrite, nullptr,
                        UseCalls, Exp, RTCI);
  }
}

Instruction *AddressSanitizer::generateCrashCode(Instruction *InsertBefore,
                                                 Value *Addr, bool IsWrite,
                                                 size_t AccessSizeIndex,
                                                 Value *SizeArgument,
                                                 uint32_t Exp,
                                                 RuntimeCallInserter &RTCI) {
  InstrumentationIRBuilder IRB(InsertBefore);
  Value *ExpVal = Exp == 0 ? nullptr : ConstantInt::get(IRB.getInt32Ty(), Exp);
  CallInst *Call = nullptr;
  if (SizeArgument) {
    if (Exp == 0)
      Call = RTCI.createRuntimeCall(IRB, AsanErrorCallbackSized[IsWrite][0],
                                    {Addr, SizeArgument});
    else
      Call = RTCI.createRuntimeCall(IRB, AsanErrorCallbackSized[IsWrite][1],
                                    {Addr, SizeArgument, ExpVal});
  } else {
    if (Exp == 0)
      Call = RTCI.createRuntimeCall(
          IRB, AsanErrorCallback[IsWrite][0][AccessSizeIndex], Addr);
    else
      Call = RTCI.createRuntimeCall(
          IRB, AsanErrorCallback[IsWrite][1][AccessSizeIndex], {Addr, ExpVal});
  }

  Call->setCannotMerge();
  return Call;
}

Value *AddressSanitizer::createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                                           Value *ShadowValue,
                                           uint32_t TypeStoreSize) {
  size_t Granularity = static_cast<size_t>(1) << Mapping.Scale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte =
      IRB.CreateAnd(AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeStoreSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeStoreSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte =
      IRB.CreateIntCast(LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

Instruction *AddressSanitizer::instrumentAMDGPUAddress(
    Instruction *OrigIns, Instruction *InsertBefore, Value *Addr,
    uint32_t TypeStoreSize, bool IsWrite, Value *SizeArgument) {
  // Do not instrument unsupported addrspaces.
  if (isUnsupportedAMDGPUAddrspace(Addr))
    return nullptr;
  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  // Follow host instrumentation for global and constant addresses.
  if (PtrTy->getPointerAddressSpace() != 0)
    return InsertBefore;
  // Instrument generic addresses in supported addressspaces.
  IRBuilder<> IRB(InsertBefore);
  Value *IsShared = IRB.CreateCall(AMDGPUAddressShared, {Addr});
  Value *IsPrivate = IRB.CreateCall(AMDGPUAddressPrivate, {Addr});
  Value *IsSharedOrPrivate = IRB.CreateOr(IsShared, IsPrivate);
  Value *Cmp = IRB.CreateNot(IsSharedOrPrivate);
  Value *AddrSpaceZeroLanding =
      SplitBlockAndInsertIfThen(Cmp, InsertBefore, false);
  InsertBefore = cast<Instruction>(AddrSpaceZeroLanding);
  return InsertBefore;
}

Instruction *AddressSanitizer::genAMDGPUReportBlock(IRBuilder<> &IRB,
                                                    Value *Cond, bool Recover) {
  Module &M = *IRB.GetInsertBlock()->getModule();
  Value *ReportCond = Cond;
  if (!Recover) {
    auto Ballot = M.getOrInsertFunction(kAMDGPUBallotName, IRB.getInt64Ty(),
                                        IRB.getInt1Ty());
    ReportCond = IRB.CreateIsNotNull(IRB.CreateCall(Ballot, {Cond}));
  }

  auto *Trm =
      SplitBlockAndInsertIfThen(ReportCond, &*IRB.GetInsertPoint(), false,
                                MDBuilder(*C).createUnlikelyBranchWeights());
  Trm->getParent()->setName("asan.report");

  if (Recover)
    return Trm;

  Trm = SplitBlockAndInsertIfThen(Cond, Trm, false);
  IRB.SetInsertPoint(Trm);
  return IRB.CreateCall(
      M.getOrInsertFunction(kAMDGPUUnreachableName, IRB.getVoidTy()), {});
}

void AddressSanitizer::instrumentAddress(Instruction *OrigIns,
                                         Instruction *InsertBefore, Value *Addr,
                                         MaybeAlign Alignment,
                                         uint32_t TypeStoreSize, bool IsWrite,
                                         Value *SizeArgument, bool UseCalls,
                                         uint32_t Exp,
                                         RuntimeCallInserter &RTCI) {
  if (TargetTriple.isAMDGPU()) {
    InsertBefore = instrumentAMDGPUAddress(OrigIns, InsertBefore, Addr,
                                           TypeStoreSize, IsWrite, SizeArgument);
    if (!InsertBefore)
      return;
  }

  InstrumentationIRBuilder IRB(InsertBefore);
  size_t AccessSizeIndex = TypeStoreSizeToSizeIndex(TypeStoreSize);

  if (UseCalls && ClOptimizeCallbacks) {
    const ASanAccessInfo AccessInfo(IsWrite, CompileKernel, AccessSizeIndex);
    IRB.CreateIntrinsic(Intrinsic::asan_check_memaccess, {},
                        {IRB.CreatePointerCast(Addr, PtrTy),
                         ConstantInt::get(Int32Ty, AccessInfo.Packed)});
    return;
  }

  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  if (UseCalls) {
    if (Exp == 0) {
      if (TargetTriple.isSPIROrSPIRV()) {
        SmallVector<Value *, 5> Args;
        auto AS = cast<PointerType>(Addr->getType()->getScalarType())
                      ->getPointerAddressSpace();
        Args.push_back(AddrLong);
        AppendDebugInfoToArgs(InsertBefore, Addr, Args);
        RTCI.createRuntimeCall(
            IRB, AsanMemoryAccessCallbackAS[IsWrite][0][AccessSizeIndex][AS],
            Args);
      } else {
        RTCI.createRuntimeCall(
          IRB, AsanMemoryAccessCallback[IsWrite][0][AccessSizeIndex], AddrLong);
      }
    } else {
      RTCI.createRuntimeCall(
          IRB, AsanMemoryAccessCallback[IsWrite][1][AccessSizeIndex],
          {AddrLong, ConstantInt::get(IRB.getInt32Ty(), Exp)});
    }
    return;
  }

  Type *ShadowTy =
      IntegerType::get(*C, std::max(8U, TypeStoreSize >> Mapping.Scale));
  Type *ShadowPtrTy = PointerType::get(*C, 0);
  Value *ShadowPtr = memToShadow(AddrLong, IRB);
  const uint64_t ShadowAlign =
      std::max<uint64_t>(Alignment.valueOrOne().value() >> Mapping.Scale, 1);
  Value *ShadowValue = IRB.CreateAlignedLoad(
      ShadowTy, IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy), Align(ShadowAlign));

  Value *Cmp = IRB.CreateIsNotNull(ShadowValue);
  size_t Granularity = 1ULL << Mapping.Scale;
  Instruction *CrashTerm = nullptr;

  bool GenSlowPath = (ClAlwaysSlowPath || (TypeStoreSize < 8 * Granularity));

  if (TargetTriple.isAMDGCN()) {
    if (GenSlowPath) {
      auto *Cmp2 = createSlowPathCmp(IRB, AddrLong, ShadowValue, TypeStoreSize);
      Cmp = IRB.CreateAnd(Cmp, Cmp2);
    }
    CrashTerm = genAMDGPUReportBlock(IRB, Cmp, Recover);
  } else if (GenSlowPath) {
    // We use branch weights for the slow path check, to indicate that the slow
    // path is rarely taken. This seems to be the case for SPEC benchmarks.
    Instruction *CheckTerm = SplitBlockAndInsertIfThen(
        Cmp, InsertBefore, false, MDBuilder(*C).createUnlikelyBranchWeights());
    assert(cast<BranchInst>(CheckTerm)->isUnconditional());
    BasicBlock *NextBB = CheckTerm->getSuccessor(0);
    IRB.SetInsertPoint(CheckTerm);
    Value *Cmp2 = createSlowPathCmp(IRB, AddrLong, ShadowValue, TypeStoreSize);
    if (Recover) {
      CrashTerm = SplitBlockAndInsertIfThen(Cmp2, CheckTerm, false);
    } else {
      BasicBlock *CrashBlock =
        BasicBlock::Create(*C, "", NextBB->getParent(), NextBB);
      CrashTerm = new UnreachableInst(*C, CrashBlock);
      BranchInst *NewTerm = BranchInst::Create(CrashBlock, NextBB, Cmp2);
      ReplaceInstWithInst(CheckTerm, NewTerm);
    }
  } else {
    CrashTerm = SplitBlockAndInsertIfThen(Cmp, InsertBefore, !Recover);
  }

  Instruction *Crash = generateCrashCode(
      CrashTerm, AddrLong, IsWrite, AccessSizeIndex, SizeArgument, Exp, RTCI);
  if (OrigIns->getDebugLoc())
    Crash->setDebugLoc(OrigIns->getDebugLoc());
}

// Instrument unusual size or unusual alignment.
// We can not do it with a single check, so we do 1-byte check for the first
// and the last bytes. We call __asan_report_*_n(addr, real_size) to be able
// to report the actual access size.
void AddressSanitizer::instrumentUnusualSizeOrAlignment(
    Instruction *I, Instruction *InsertBefore, Value *Addr,
    TypeSize TypeStoreSize, bool IsWrite, Value *SizeArgument, bool UseCalls,
    uint32_t Exp, RuntimeCallInserter &RTCI) {
  InstrumentationIRBuilder IRB(InsertBefore);
  Value *NumBits = IRB.CreateTypeSize(IntptrTy, TypeStoreSize);
  Value *Size = IRB.CreateLShr(NumBits, ConstantInt::get(IntptrTy, 3));

  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  if (UseCalls) {
    if (Exp == 0) {
      if (TargetTriple.isSPIROrSPIRV()) {
        SmallVector<Value *, 6> Args;
        auto AS = cast<PointerType>(Addr->getType()->getScalarType())
                      ->getPointerAddressSpace();
        Args.push_back(AddrLong);
        Args.push_back(Size);
        AppendDebugInfoToArgs(InsertBefore, Addr, Args);
        RTCI.createRuntimeCall(
            IRB, AsanMemoryAccessCallbackSizedAS[IsWrite][0][AS], Args);
      } else {
        RTCI.createRuntimeCall(IRB, AsanMemoryAccessCallbackSized[IsWrite][0],
                               {AddrLong, Size});
      }
    } else
      RTCI.createRuntimeCall(
          IRB, AsanMemoryAccessCallbackSized[IsWrite][1],
          {AddrLong, Size, ConstantInt::get(IRB.getInt32Ty(), Exp)});
  } else {
    Value *SizeMinusOne = IRB.CreateSub(Size, ConstantInt::get(IntptrTy, 1));
    Value *LastByte = IRB.CreateIntToPtr(
        IRB.CreateAdd(AddrLong, SizeMinusOne),
        Addr->getType());
    instrumentAddress(I, InsertBefore, Addr, {}, 8, IsWrite, Size, false, Exp,
                      RTCI);
    instrumentAddress(I, InsertBefore, LastByte, {}, 8, IsWrite, Size, false,
                      Exp, RTCI);
  }
}

void ModuleAddressSanitizer::poisonOneInitializer(Function &GlobalInit) {
  // Set up the arguments to our poison/unpoison functions.
  IRBuilder<> IRB(&GlobalInit.front(),
                  GlobalInit.front().getFirstInsertionPt());

  // Add a call to poison all external globals before the given function starts.
  Value *ModuleNameAddr =
      ConstantExpr::getPointerCast(getOrCreateModuleName(), IntptrTy);
  IRB.CreateCall(AsanPoisonGlobals, ModuleNameAddr);

  // Add calls to unpoison all globals before each return instruction.
  for (auto &BB : GlobalInit)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      CallInst::Create(AsanUnpoisonGlobals, "", RI->getIterator());
}

void ModuleAddressSanitizer::createInitializerPoisonCalls() {
  GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
  if (!GV)
    return;

  ConstantArray *CA = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!CA)
    return;

  for (Use &OP : CA->operands()) {
    if (isa<ConstantAggregateZero>(OP)) continue;
    ConstantStruct *CS = cast<ConstantStruct>(OP);

    // Must have a function or null ptr.
    if (Function *F = dyn_cast<Function>(CS->getOperand(1))) {
      if (F->getName() == kAsanModuleCtorName) continue;
      auto *Priority = cast<ConstantInt>(CS->getOperand(0));
      // Don't instrument CTORs that will run before asan.module_ctor.
      if (Priority->getLimitedValue() <= GetCtorAndDtorPriority(TargetTriple))
        continue;
      poisonOneInitializer(*F);
    }
  }
}

const GlobalVariable *
ModuleAddressSanitizer::getExcludedAliasedGlobal(const GlobalAlias &GA) const {
  // In case this function should be expanded to include rules that do not just
  // apply when CompileKernel is true, either guard all existing rules with an
  // 'if (CompileKernel) { ... }' or be absolutely sure that all these rules
  // should also apply to user space.
  assert(CompileKernel && "Only expecting to be called when compiling kernel");

  const Constant *C = GA.getAliasee();

  // When compiling the kernel, globals that are aliased by symbols prefixed
  // by "__" are special and cannot be padded with a redzone.
  if (GA.getName().starts_with("__"))
    return dyn_cast<GlobalVariable>(C->stripPointerCastsAndAliases());

  return nullptr;
}

bool ModuleAddressSanitizer::shouldInstrumentGlobal(GlobalVariable *G) const {
  Type *Ty = G->getValueType();
  LLVM_DEBUG(dbgs() << "GLOBAL: " << *G << "\n");

  if (G->hasSanitizerMetadata() && G->getSanitizerMetadata().NoAddress)
    return false;
  if (!Ty->isSized()) return false;
  if (!G->hasInitializer()) return false;
  // Globals in address space 1 and 4 are supported for AMDGPU.
  if (G->getAddressSpace() &&
      !(TargetTriple.isAMDGPU() && !isUnsupportedAMDGPUAddrspace(G)))
    return false;
  if (GlobalWasGeneratedByCompiler(G)) return false; // Our own globals.
  // Two problems with thread-locals:
  //   - The address of the main thread's copy can't be computed at link-time.
  //   - Need to poison all copies, not just the main thread's one.
  if (G->isThreadLocal()) return false;
  // For now, just ignore this Global if the alignment is large.
  if (G->getAlign() && *G->getAlign() > getMinRedzoneSizeForGlobal()) return false;

  // For non-COFF targets, only instrument globals known to be defined by this
  // TU.
  // FIXME: We can instrument comdat globals on ELF if we are using the
  // GC-friendly metadata scheme.
  if (!TargetTriple.isOSBinFormatCOFF()) {
    if (!G->hasExactDefinition() || G->hasComdat())
      return false;
  } else {
    // On COFF, don't instrument non-ODR linkages.
    if (G->isInterposable())
      return false;
    // If the global has AvailableExternally linkage, then it is not in this
    // module, which means it does not need to be instrumented.
    if (G->hasAvailableExternallyLinkage())
      return false;
  }

  // If a comdat is present, it must have a selection kind that implies ODR
  // semantics: no duplicates, any, or exact match.
  if (Comdat *C = G->getComdat()) {
    switch (C->getSelectionKind()) {
    case Comdat::Any:
    case Comdat::ExactMatch:
    case Comdat::NoDeduplicate:
      break;
    case Comdat::Largest:
    case Comdat::SameSize:
      return false;
    }
  }

  if (G->hasSection()) {
    // The kernel uses explicit sections for mostly special global variables
    // that we should not instrument. E.g. the kernel may rely on their layout
    // without redzones, or remove them at link time ("discard.*"), etc.
    if (CompileKernel)
      return false;

    StringRef Section = G->getSection();

    // Globals from llvm.metadata aren't emitted, do not instrument them.
    if (Section == "llvm.metadata") return false;
    // Do not instrument globals from special LLVM sections.
    if (Section.contains("__llvm") || Section.contains("__LLVM"))
      return false;

    // Do not instrument function pointers to initialization and termination
    // routines: dynamic linker will not properly handle redzones.
    if (Section.starts_with(".preinit_array") ||
        Section.starts_with(".init_array") ||
        Section.starts_with(".fini_array")) {
      return false;
    }

    // Do not instrument user-defined sections (with names resembling
    // valid C identifiers)
    if (TargetTriple.isOSBinFormatELF()) {
      if (llvm::all_of(Section,
                       [](char c) { return llvm::isAlnum(c) || c == '_'; }))
        return false;
    }

    // On COFF, if the section name contains '$', it is highly likely that the
    // user is using section sorting to create an array of globals similar to
    // the way initialization callbacks are registered in .init_array and
    // .CRT$XCU. The ATL also registers things in .ATL$__[azm]. Adding redzones
    // to such globals is counterproductive, because the intent is that they
    // will form an array, and out-of-bounds accesses are expected.
    // See https://github.com/google/sanitizers/issues/305
    // and http://msdn.microsoft.com/en-US/en-en/library/bb918180(v=vs.120).aspx
    if (TargetTriple.isOSBinFormatCOFF() && Section.contains('$')) {
      LLVM_DEBUG(dbgs() << "Ignoring global in sorted section (contains '$'): "
                        << *G << "\n");
      return false;
    }

    if (TargetTriple.isOSBinFormatMachO()) {
      StringRef ParsedSegment, ParsedSection;
      unsigned TAA = 0, StubSize = 0;
      bool TAAParsed;
      cantFail(MCSectionMachO::ParseSectionSpecifier(
          Section, ParsedSegment, ParsedSection, TAA, TAAParsed, StubSize));

      // Ignore the globals from the __OBJC section. The ObjC runtime assumes
      // those conform to /usr/lib/objc/runtime.h, so we can't add redzones to
      // them.
      if (ParsedSegment == "__OBJC" ||
          (ParsedSegment == "__DATA" && ParsedSection.starts_with("__objc_"))) {
        LLVM_DEBUG(dbgs() << "Ignoring ObjC runtime global: " << *G << "\n");
        return false;
      }
      // See https://github.com/google/sanitizers/issues/32
      // Constant CFString instances are compiled in the following way:
      //  -- the string buffer is emitted into
      //     __TEXT,__cstring,cstring_literals
      //  -- the constant NSConstantString structure referencing that buffer
      //     is placed into __DATA,__cfstring
      // Therefore there's no point in placing redzones into __DATA,__cfstring.
      // Moreover, it causes the linker to crash on OS X 10.7
      if (ParsedSegment == "__DATA" && ParsedSection == "__cfstring") {
        LLVM_DEBUG(dbgs() << "Ignoring CFString: " << *G << "\n");
        return false;
      }
      // The linker merges the contents of cstring_literals and removes the
      // trailing zeroes.
      if (ParsedSegment == "__TEXT" && (TAA & MachO::S_CSTRING_LITERALS)) {
        LLVM_DEBUG(dbgs() << "Ignoring a cstring literal: " << *G << "\n");
        return false;
      }
    }
  }

  if (CompileKernel) {
    // Globals that prefixed by "__" are special and cannot be padded with a
    // redzone.
    if (G->getName().starts_with("__"))
      return false;
  }

  return true;
}

// On Mach-O platforms, we emit global metadata in a separate section of the
// binary in order to allow the linker to properly dead strip. This is only
// supported on recent versions of ld64.
bool ModuleAddressSanitizer::ShouldUseMachOGlobalsSection() const {
  if (!TargetTriple.isOSBinFormatMachO())
    return false;

  if (TargetTriple.isMacOSX() && !TargetTriple.isMacOSXVersionLT(10, 11))
    return true;
  if (TargetTriple.isiOS() /* or tvOS */ && !TargetTriple.isOSVersionLT(9))
    return true;
  if (TargetTriple.isWatchOS() && !TargetTriple.isOSVersionLT(2))
    return true;
  if (TargetTriple.isDriverKit())
    return true;
  if (TargetTriple.isXROS())
    return true;

  return false;
}

StringRef ModuleAddressSanitizer::getGlobalMetadataSection() const {
  switch (TargetTriple.getObjectFormat()) {
  case Triple::COFF:  return ".ASAN$GL";
  case Triple::ELF:   return "asan_globals";
  case Triple::MachO: return "__DATA,__asan_globals,regular";
  case Triple::Wasm:
  case Triple::GOFF:
  case Triple::SPIRV:
  case Triple::XCOFF:
  case Triple::DXContainer:
    report_fatal_error(
        "ModuleAddressSanitizer not implemented for object file format");
  case Triple::UnknownObjectFormat:
    break;
  }
  llvm_unreachable("unsupported object format");
}

void ModuleAddressSanitizer::initializeCallbacks() {
  IRBuilder<> IRB(*C);

  // Declare our poisoning and unpoisoning functions.
  AsanPoisonGlobals =
      M.getOrInsertFunction(kAsanPoisonGlobalsName, IRB.getVoidTy(), IntptrTy);
  AsanUnpoisonGlobals =
      M.getOrInsertFunction(kAsanUnpoisonGlobalsName, IRB.getVoidTy());

  // Declare functions that register/unregister globals.
  AsanRegisterGlobals = M.getOrInsertFunction(
      kAsanRegisterGlobalsName, IRB.getVoidTy(), IntptrTy, IntptrTy);
  AsanUnregisterGlobals = M.getOrInsertFunction(
      kAsanUnregisterGlobalsName, IRB.getVoidTy(), IntptrTy, IntptrTy);

  // Declare the functions that find globals in a shared object and then invoke
  // the (un)register function on them.
  AsanRegisterImageGlobals = M.getOrInsertFunction(
      kAsanRegisterImageGlobalsName, IRB.getVoidTy(), IntptrTy);
  AsanUnregisterImageGlobals = M.getOrInsertFunction(
      kAsanUnregisterImageGlobalsName, IRB.getVoidTy(), IntptrTy);

  AsanRegisterElfGlobals =
      M.getOrInsertFunction(kAsanRegisterElfGlobalsName, IRB.getVoidTy(),
                            IntptrTy, IntptrTy, IntptrTy);
  AsanUnregisterElfGlobals =
      M.getOrInsertFunction(kAsanUnregisterElfGlobalsName, IRB.getVoidTy(),
                            IntptrTy, IntptrTy, IntptrTy);

  // __asan_set_shadow_static_local(
  //   uptr ptr,
  //   size_t size,
  //   size_t size_with_redzone
  // )
  AsanSetShadowStaticLocalFunc =
      M.getOrInsertFunction("__asan_set_shadow_static_local", IRB.getVoidTy(),
                            IntptrTy, IntptrTy, IntptrTy);

  // __asan_unpoison_shadow_static_local(
  //   uptr ptr,
  //   size_t size,
  //   size_t size_with_redzone
  // )
  AsanUnpoisonShadowStaticLocalFunc =
      M.getOrInsertFunction("__asan_unpoison_shadow_static_local",
                            IRB.getVoidTy(), IntptrTy, IntptrTy, IntptrTy);
}

// Put the metadata and the instrumented global in the same group. This ensures
// that the metadata is discarded if the instrumented global is discarded.
void ModuleAddressSanitizer::SetComdatForGlobalMetadata(
    GlobalVariable *G, GlobalVariable *Metadata, StringRef InternalSuffix) {
  Module &M = *G->getParent();
  Comdat *C = G->getComdat();
  if (!C) {
    if (!G->hasName()) {
      // If G is unnamed, it must be internal. Give it an artificial name
      // so we can put it in a comdat.
      assert(G->hasLocalLinkage());
      G->setName(genName("anon_global"));
    }

    if (!InternalSuffix.empty() && G->hasLocalLinkage()) {
      std::string Name = std::string(G->getName());
      Name += InternalSuffix;
      C = M.getOrInsertComdat(Name);
    } else {
      C = M.getOrInsertComdat(G->getName());
    }

    // Make this IMAGE_COMDAT_SELECT_NODUPLICATES on COFF. Also upgrade private
    // linkage to internal linkage so that a symbol table entry is emitted. This
    // is necessary in order to create the comdat group.
    if (TargetTriple.isOSBinFormatCOFF()) {
      C->setSelectionKind(Comdat::NoDeduplicate);
      if (G->hasPrivateLinkage())
        G->setLinkage(GlobalValue::InternalLinkage);
    }
    G->setComdat(C);
  }

  assert(G->hasComdat());
  Metadata->setComdat(G->getComdat());
}

// Create a separate metadata global and put it in the appropriate ASan
// global registration section.
GlobalVariable *
ModuleAddressSanitizer::CreateMetadataGlobal(Constant *Initializer,
                                             StringRef OriginalName) {
  auto Linkage = TargetTriple.isOSBinFormatMachO()
                     ? GlobalVariable::InternalLinkage
                     : GlobalVariable::PrivateLinkage;
  GlobalVariable *Metadata = new GlobalVariable(
      M, Initializer->getType(), false, Linkage, Initializer,
      Twine("__asan_global_") + GlobalValue::dropLLVMManglingEscape(OriginalName));
  Metadata->setSection(getGlobalMetadataSection());
  // Place metadata in a large section for x86-64 ELF binaries to mitigate
  // relocation pressure.
  setGlobalVariableLargeSection(TargetTriple, *Metadata);
  return Metadata;
}

Instruction *ModuleAddressSanitizer::CreateAsanModuleDtor() {
  AsanDtorFunction = Function::createWithDefaultAttr(
      FunctionType::get(Type::getVoidTy(*C), false),
      GlobalValue::InternalLinkage, 0, kAsanModuleDtorName, &M);
  AsanDtorFunction->addFnAttr(Attribute::NoUnwind);
  // Ensure Dtor cannot be discarded, even if in a comdat.
  appendToUsed(M, {AsanDtorFunction});
  BasicBlock *AsanDtorBB = BasicBlock::Create(*C, "", AsanDtorFunction);

  return ReturnInst::Create(*C, AsanDtorBB);
}

void ModuleAddressSanitizer::instrumentDeviceGlobal(IRBuilder<> &IRB) {
  auto &DL = M.getDataLayout();
  SmallVector<GlobalVariable *, 8> GlobalsToRemove;
  SmallVector<Constant *, 8> DeviceGlobalMetadata;

  Type *IntptrTy = M.getDataLayout().getIntPtrType(*C, kSpirOffloadGlobalAS);

  // Device global meta data is described by a structure
  //  size_t device_global_size
  //  size_t device_global_size_with_red_zone
  //  size_t beginning address of the device global
  StructType *StructTy = StructType::get(IntptrTy, IntptrTy, IntptrTy);

  for (auto &G : M.globals()) {
    // DeviceSanitizers cannot handle nameless globals, therefore we set a name
    // for them so that we can handle them like regular globals.
    if (G.getName().empty() && G.hasInternalLinkage())
      G.setName("nameless_global");

    if (isUnsupportedDeviceGlobal(&G))
      continue;

    // This case is handled by instrumentSyclStaticLocalMemory
    if (G.getAddressSpace() == kSpirOffloadLocalAS)
      continue;

    Type *Ty = G.getValueType();
    const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
    const uint64_t RightRedzoneSize = getRedzoneSizeForGlobal(SizeInBytes);
    Type *RightRedZoneTy = ArrayType::get(IRB.getInt8Ty(), RightRedzoneSize);
    StructType *NewTy = StructType::get(Ty, RightRedZoneTy);
    Constant *NewInitializer = ConstantStruct::get(
        NewTy, G.getInitializer(), Constant::getNullValue(RightRedZoneTy));

    // Create a new global variable with enough space for a redzone.
    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G.isConstant(), G.getLinkage(), NewInitializer, "", &G,
        G.getThreadLocalMode(), G.getAddressSpace());
    NewGlobal->copyAttributesFrom(&G);
    NewGlobal->setComdat(G.getComdat());
    NewGlobal->setAlignment(Align(getMinRedzoneSizeForGlobal()));
    NewGlobal->copyMetadata(&G, 0);

    Value *Indices2[2];
    Indices2[0] = IRB.getInt32(0);
    Indices2[1] = IRB.getInt32(0);

    G.replaceAllUsesWith(
        ConstantExpr::getGetElementPtr(NewTy, NewGlobal, Indices2, true));
    NewGlobal->takeName(&G);
    GlobalsToRemove.push_back(&G);
    DeviceGlobalMetadata.push_back(ConstantStruct::get(
        StructTy, ConstantInt::get(IntptrTy, SizeInBytes),
        ConstantInt::get(IntptrTy, SizeInBytes + RightRedzoneSize),
        ConstantExpr::getPointerCast(NewGlobal, IntptrTy)));
  }

  if (GlobalsToRemove.empty())
    return;

  // Create meta data global to record device globals' information
  ArrayType *ArrayTy = ArrayType::get(StructTy, DeviceGlobalMetadata.size());
  Constant *MetadataInitializer =
      ConstantArray::get(ArrayTy, DeviceGlobalMetadata);
  GlobalVariable *AsanDeviceGlobalMetadata = new GlobalVariable(
      M, MetadataInitializer->getType(), false, GlobalValue::AppendingLinkage,
      MetadataInitializer, "__AsanDeviceGlobalMetadata", nullptr,
      GlobalValue::NotThreadLocal, 1);
  AsanDeviceGlobalMetadata->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  for (auto *G : GlobalsToRemove)
    G->eraseFromParent();
}

void ModuleAddressSanitizer::initializeRetVecMap(Function *F) {
  if (KernelToRetVecMap.find(F) != KernelToRetVecMap.end())
    return;

  SmallVector<Instruction *, 8> RetVec;
  for (auto &BB : *F) {
    for (auto &Inst : BB) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(&Inst)) {
        if (CallInst *CI = RI->getParent()->getTerminatingMustTailCall())
          RetVec.push_back(CI);
        else
          RetVec.push_back(RI);
      } else if (ResumeInst *RI = dyn_cast<ResumeInst>(&Inst)) {
        RetVec.push_back(RI);
      } else if (CleanupReturnInst *CRI = dyn_cast<CleanupReturnInst>(&Inst)) {
        RetVec.push_back(CRI);
      }
    }
  }

  KernelToRetVecMap[F] = std::move(RetVec);
}

void ModuleAddressSanitizer::initializeKernelCallerMap(Function *F) {
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

// Instument static local memory
void ModuleAddressSanitizer::instrumentSyclStaticLocalMemory(IRBuilder<> &IRB) {
  auto &DL = M.getDataLayout();
  SmallVector<GlobalVariable *, 8> GlobalsToRemove;
  SmallVector<GlobalVariable *, 8> LocalGlobals;

  Type *IntptrTy = M.getDataLayout().getIntPtrType(*C, kSpirOffloadGlobalAS);

  // Step1. Create a new global variable with enough space for a redzone.
  for (auto &G : M.globals()) {
    if (G.getAddressSpace() != kSpirOffloadLocalAS)
      continue;
    if (G.getName().starts_with("__Asan"))
      continue;

    Type *Ty = G.getValueType();
    const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
    const uint64_t RightRedzoneSize = getRedzoneSizeForGlobal(SizeInBytes);
    Type *RightRedZoneTy = ArrayType::get(IRB.getInt8Ty(), RightRedzoneSize);
    StructType *NewTy = StructType::get(Ty, RightRedZoneTy);
    Constant *NewInitializer =
        G.hasInitializer()
            ? ConstantStruct::get(NewTy, G.getInitializer(),
                                  Constant::getNullValue(RightRedZoneTy))
            : nullptr;

    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G.isConstant(), G.getLinkage(), NewInitializer, "", &G,
        G.getThreadLocalMode(), G.getAddressSpace());
    NewGlobal->copyAttributesFrom(&G);
    NewGlobal->setComdat(G.getComdat());
    NewGlobal->setAlignment(Align(getMinRedzoneSizeForGlobal()));
    NewGlobal->copyMetadata(&G, 0);

    Value *Indices2[2];
    Indices2[0] = IRB.getInt32(0);
    Indices2[1] = IRB.getInt32(0);

    G.replaceAllUsesWith(
        ConstantExpr::getGetElementPtr(NewTy, NewGlobal, Indices2, true));
    NewGlobal->takeName(&G);
    GlobalsToRemove.push_back(&G);
    LocalGlobals.push_back(NewGlobal);
  }

  if (GlobalsToRemove.empty())
    return;

  for (auto *G : GlobalsToRemove)
    G->eraseFromParent();

  // Step2. Instrument initialization functions on kernel
  DenseMap<Function *, Instruction *> FuncToLaunchInfoMap;
  auto Instrument = [&](GlobalVariable *G, Function *F) {
    StructType *Type = cast<StructType>(G->getValueType());
    const uint64_t Size = DL.getTypeAllocSize(Type->getElementType(0));
    const uint64_t SizeWithRedZone = DL.getTypeAllocSize(Type);

    // Poison shadow of static local memory
    if (FuncToLaunchInfoMap.find(F) == FuncToLaunchInfoMap.end()) {
      for (auto &Inst : F->getEntryBlock()) {
        auto *SI = dyn_cast<StoreInst>(&Inst);
        if (SI && (SI->getPointerOperand()->getName() == "__AsanLaunchInfo")) {
          FuncToLaunchInfoMap[F] = &Inst;
          break;
        }
      }
    }
    assert(FuncToLaunchInfoMap.find(F) != FuncToLaunchInfoMap.end() &&
           "All spir kernels should be instrumented.");

    IRBuilder<> Builder(FuncToLaunchInfoMap[F]->getNextNode());
    Builder.CreateCall(AsanSetShadowStaticLocalFunc,
                       {Builder.CreatePointerCast(G, IntptrTy),
                        ConstantInt::get(IntptrTy, Size),
                        ConstantInt::get(IntptrTy, SizeWithRedZone)});

    // Unpoison shadow of static local memory, required by CPU device
    initializeRetVecMap(F);
    for (auto *RI : KernelToRetVecMap[F]) {
      IRBuilder<> Builder(RI);
      Builder.CreateCall(AsanUnpoisonShadowStaticLocalFunc,
                         {Builder.CreatePointerCast(G, IntptrTy),
                          ConstantInt::get(IntptrTy, Size),
                          ConstantInt::get(IntptrTy, SizeWithRedZone)});
    }
  };

  // We only instrument on spir_kernel, because local variables are
  // kind of global variable
  for (auto *G : LocalGlobals) {
    SmallVector<Function *> WorkList;
    DenseSet<Function *> InstrumentedKernel;
    for (auto *User : G->users())
      getFunctionsOfUser(User, WorkList);
    while (!WorkList.empty()) {
      Function *F = WorkList.pop_back_val();
      if (F->getCallingConv() == CallingConv::SPIR_KERNEL) {
        if (!InstrumentedKernel.contains(F)) {
          Instrument(G, F);
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

void ModuleAddressSanitizer::InstrumentGlobalsCOFF(
    IRBuilder<> &IRB, ArrayRef<GlobalVariable *> ExtendedGlobals,
    ArrayRef<Constant *> MetadataInitializers) {
  assert(ExtendedGlobals.size() == MetadataInitializers.size());
  auto &DL = M.getDataLayout();

  SmallVector<GlobalValue *, 16> MetadataGlobals(ExtendedGlobals.size());
  for (size_t i = 0; i < ExtendedGlobals.size(); i++) {
    Constant *Initializer = MetadataInitializers[i];
    GlobalVariable *G = ExtendedGlobals[i];
    GlobalVariable *Metadata = CreateMetadataGlobal(Initializer, G->getName());
    MDNode *MD = MDNode::get(M.getContext(), ValueAsMetadata::get(G));
    Metadata->setMetadata(LLVMContext::MD_associated, MD);
    MetadataGlobals[i] = Metadata;

    // The MSVC linker always inserts padding when linking incrementally. We
    // cope with that by aligning each struct to its size, which must be a power
    // of two.
    unsigned SizeOfGlobalStruct = DL.getTypeAllocSize(Initializer->getType());
    assert(isPowerOf2_32(SizeOfGlobalStruct) &&
           "global metadata will not be padded appropriately");
    Metadata->setAlignment(assumeAligned(SizeOfGlobalStruct));

    SetComdatForGlobalMetadata(G, Metadata, "");
  }

  // Update llvm.compiler.used, adding the new metadata globals. This is
  // needed so that during LTO these variables stay alive.
  if (!MetadataGlobals.empty())
    appendToCompilerUsed(M, MetadataGlobals);
}

void ModuleAddressSanitizer::instrumentGlobalsELF(
    IRBuilder<> &IRB, ArrayRef<GlobalVariable *> ExtendedGlobals,
    ArrayRef<Constant *> MetadataInitializers,
    const std::string &UniqueModuleId) {
  assert(ExtendedGlobals.size() == MetadataInitializers.size());

  // Putting globals in a comdat changes the semantic and potentially cause
  // false negative odr violations at link time. If odr indicators are used, we
  // keep the comdat sections, as link time odr violations will be dectected on
  // the odr indicator symbols.
  bool UseComdatForGlobalsGC = UseOdrIndicator && !UniqueModuleId.empty();

  SmallVector<GlobalValue *, 16> MetadataGlobals(ExtendedGlobals.size());
  for (size_t i = 0; i < ExtendedGlobals.size(); i++) {
    GlobalVariable *G = ExtendedGlobals[i];
    GlobalVariable *Metadata =
        CreateMetadataGlobal(MetadataInitializers[i], G->getName());
    MDNode *MD = MDNode::get(M.getContext(), ValueAsMetadata::get(G));
    Metadata->setMetadata(LLVMContext::MD_associated, MD);
    MetadataGlobals[i] = Metadata;

    if (UseComdatForGlobalsGC)
      SetComdatForGlobalMetadata(G, Metadata, UniqueModuleId);
  }

  // Update llvm.compiler.used, adding the new metadata globals. This is
  // needed so that during LTO these variables stay alive.
  if (!MetadataGlobals.empty())
    appendToCompilerUsed(M, MetadataGlobals);

  // RegisteredFlag serves two purposes. First, we can pass it to dladdr()
  // to look up the loaded image that contains it. Second, we can store in it
  // whether registration has already occurred, to prevent duplicate
  // registration.
  //
  // Common linkage ensures that there is only one global per shared library.
  GlobalVariable *RegisteredFlag = new GlobalVariable(
      M, IntptrTy, false, GlobalVariable::CommonLinkage,
      ConstantInt::get(IntptrTy, 0), kAsanGlobalsRegisteredFlagName);
  RegisteredFlag->setVisibility(GlobalVariable::HiddenVisibility);

  // Create start and stop symbols.
  GlobalVariable *StartELFMetadata = new GlobalVariable(
      M, IntptrTy, false, GlobalVariable::ExternalWeakLinkage, nullptr,
      "__start_" + getGlobalMetadataSection());
  StartELFMetadata->setVisibility(GlobalVariable::HiddenVisibility);
  GlobalVariable *StopELFMetadata = new GlobalVariable(
      M, IntptrTy, false, GlobalVariable::ExternalWeakLinkage, nullptr,
      "__stop_" + getGlobalMetadataSection());
  StopELFMetadata->setVisibility(GlobalVariable::HiddenVisibility);

  // Create a call to register the globals with the runtime.
  if (ConstructorKind == AsanCtorKind::Global)
    IRB.CreateCall(AsanRegisterElfGlobals,
                 {IRB.CreatePointerCast(RegisteredFlag, IntptrTy),
                  IRB.CreatePointerCast(StartELFMetadata, IntptrTy),
                  IRB.CreatePointerCast(StopELFMetadata, IntptrTy)});

  // We also need to unregister globals at the end, e.g., when a shared library
  // gets closed.
  if (DestructorKind != AsanDtorKind::None && !MetadataGlobals.empty()) {
    IRBuilder<> IrbDtor(CreateAsanModuleDtor());
    IrbDtor.CreateCall(AsanUnregisterElfGlobals,
                       {IRB.CreatePointerCast(RegisteredFlag, IntptrTy),
                        IRB.CreatePointerCast(StartELFMetadata, IntptrTy),
                        IRB.CreatePointerCast(StopELFMetadata, IntptrTy)});
  }
}

void ModuleAddressSanitizer::InstrumentGlobalsMachO(
    IRBuilder<> &IRB, ArrayRef<GlobalVariable *> ExtendedGlobals,
    ArrayRef<Constant *> MetadataInitializers) {
  assert(ExtendedGlobals.size() == MetadataInitializers.size());

  // On recent Mach-O platforms, use a structure which binds the liveness of
  // the global variable to the metadata struct. Keep the list of "Liveness" GV
  // created to be added to llvm.compiler.used
  StructType *LivenessTy = StructType::get(IntptrTy, IntptrTy);
  SmallVector<GlobalValue *, 16> LivenessGlobals(ExtendedGlobals.size());

  for (size_t i = 0; i < ExtendedGlobals.size(); i++) {
    Constant *Initializer = MetadataInitializers[i];
    GlobalVariable *G = ExtendedGlobals[i];
    GlobalVariable *Metadata = CreateMetadataGlobal(Initializer, G->getName());

    // On recent Mach-O platforms, we emit the global metadata in a way that
    // allows the linker to properly strip dead globals.
    auto LivenessBinder =
        ConstantStruct::get(LivenessTy, Initializer->getAggregateElement(0u),
                            ConstantExpr::getPointerCast(Metadata, IntptrTy));
    GlobalVariable *Liveness = new GlobalVariable(
        M, LivenessTy, false, GlobalVariable::InternalLinkage, LivenessBinder,
        Twine("__asan_binder_") + G->getName());
    Liveness->setSection("__DATA,__asan_liveness,regular,live_support");
    LivenessGlobals[i] = Liveness;
  }

  // Update llvm.compiler.used, adding the new liveness globals. This is
  // needed so that during LTO these variables stay alive. The alternative
  // would be to have the linker handling the LTO symbols, but libLTO
  // current API does not expose access to the section for each symbol.
  if (!LivenessGlobals.empty())
    appendToCompilerUsed(M, LivenessGlobals);

  // RegisteredFlag serves two purposes. First, we can pass it to dladdr()
  // to look up the loaded image that contains it. Second, we can store in it
  // whether registration has already occurred, to prevent duplicate
  // registration.
  //
  // common linkage ensures that there is only one global per shared library.
  GlobalVariable *RegisteredFlag = new GlobalVariable(
      M, IntptrTy, false, GlobalVariable::CommonLinkage,
      ConstantInt::get(IntptrTy, 0), kAsanGlobalsRegisteredFlagName);
  RegisteredFlag->setVisibility(GlobalVariable::HiddenVisibility);

  if (ConstructorKind == AsanCtorKind::Global)
    IRB.CreateCall(AsanRegisterImageGlobals,
                 {IRB.CreatePointerCast(RegisteredFlag, IntptrTy)});

  // We also need to unregister globals at the end, e.g., when a shared library
  // gets closed.
  if (DestructorKind != AsanDtorKind::None) {
    IRBuilder<> IrbDtor(CreateAsanModuleDtor());
    IrbDtor.CreateCall(AsanUnregisterImageGlobals,
                       {IRB.CreatePointerCast(RegisteredFlag, IntptrTy)});
  }
}

void ModuleAddressSanitizer::InstrumentGlobalsWithMetadataArray(
    IRBuilder<> &IRB, ArrayRef<GlobalVariable *> ExtendedGlobals,
    ArrayRef<Constant *> MetadataInitializers) {
  assert(ExtendedGlobals.size() == MetadataInitializers.size());
  unsigned N = ExtendedGlobals.size();
  assert(N > 0);

  // On platforms that don't have a custom metadata section, we emit an array
  // of global metadata structures.
  ArrayType *ArrayOfGlobalStructTy =
      ArrayType::get(MetadataInitializers[0]->getType(), N);
  auto AllGlobals = new GlobalVariable(
      M, ArrayOfGlobalStructTy, false, GlobalVariable::InternalLinkage,
      ConstantArray::get(ArrayOfGlobalStructTy, MetadataInitializers), "");
  if (Mapping.Scale > 3)
    AllGlobals->setAlignment(Align(1ULL << Mapping.Scale));

  if (ConstructorKind == AsanCtorKind::Global)
    IRB.CreateCall(AsanRegisterGlobals,
                 {IRB.CreatePointerCast(AllGlobals, IntptrTy),
                  ConstantInt::get(IntptrTy, N)});

  // We also need to unregister globals at the end, e.g., when a shared library
  // gets closed.
  if (DestructorKind != AsanDtorKind::None) {
    IRBuilder<> IrbDtor(CreateAsanModuleDtor());
    IrbDtor.CreateCall(AsanUnregisterGlobals,
                       {IRB.CreatePointerCast(AllGlobals, IntptrTy),
                        ConstantInt::get(IntptrTy, N)});
  }
}

// This function replaces all global variables with new variables that have
// trailing redzones. It also creates a function that poisons
// redzones and inserts this function into llvm.global_ctors.
// Sets *CtorComdat to true if the global registration code emitted into the
// asan constructor is comdat-compatible.
void ModuleAddressSanitizer::instrumentGlobals(IRBuilder<> &IRB,
                                               bool *CtorComdat) {
  // Build set of globals that are aliased by some GA, where
  // getExcludedAliasedGlobal(GA) returns the relevant GlobalVariable.
  SmallPtrSet<const GlobalVariable *, 16> AliasedGlobalExclusions;
  if (CompileKernel) {
    for (auto &GA : M.aliases()) {
      if (const GlobalVariable *GV = getExcludedAliasedGlobal(GA))
        AliasedGlobalExclusions.insert(GV);
    }
  }

  SmallVector<GlobalVariable *, 16> GlobalsToChange;
  for (auto &G : M.globals()) {
    if (!AliasedGlobalExclusions.count(&G) && shouldInstrumentGlobal(&G))
      GlobalsToChange.push_back(&G);
  }

  size_t n = GlobalsToChange.size();
  auto &DL = M.getDataLayout();

  // A global is described by a structure
  //   size_t beg;
  //   size_t size;
  //   size_t size_with_redzone;
  //   const char *name;
  //   const char *module_name;
  //   size_t has_dynamic_init;
  //   size_t padding_for_windows_msvc_incremental_link;
  //   size_t odr_indicator;
  // We initialize an array of such structures and pass it to a run-time call.
  StructType *GlobalStructTy =
      StructType::get(IntptrTy, IntptrTy, IntptrTy, IntptrTy, IntptrTy,
                      IntptrTy, IntptrTy, IntptrTy);
  SmallVector<GlobalVariable *, 16> NewGlobals(n);
  SmallVector<Constant *, 16> Initializers(n);

  for (size_t i = 0; i < n; i++) {
    GlobalVariable *G = GlobalsToChange[i];

    GlobalValue::SanitizerMetadata MD;
    if (G->hasSanitizerMetadata())
      MD = G->getSanitizerMetadata();

    // The runtime library tries demangling symbol names in the descriptor but
    // functionality like __cxa_demangle may be unavailable (e.g.
    // -static-libstdc++). So we demangle the symbol names here.
    std::string NameForGlobal = G->getName().str();
    GlobalVariable *Name =
        createPrivateGlobalForString(M, llvm::demangle(NameForGlobal),
                                     /*AllowMerging*/ true, genName("global"));

    Type *Ty = G->getValueType();
    const uint64_t SizeInBytes = DL.getTypeAllocSize(Ty);
    const uint64_t RightRedzoneSize = getRedzoneSizeForGlobal(SizeInBytes);
    Type *RightRedZoneTy = ArrayType::get(IRB.getInt8Ty(), RightRedzoneSize);

    StructType *NewTy = StructType::get(Ty, RightRedZoneTy);
    Constant *NewInitializer = ConstantStruct::get(
        NewTy, G->getInitializer(), Constant::getNullValue(RightRedZoneTy));

    // Create a new global variable with enough space for a redzone.
    GlobalValue::LinkageTypes Linkage = G->getLinkage();
    if (G->isConstant() && Linkage == GlobalValue::PrivateLinkage)
      Linkage = GlobalValue::InternalLinkage;
    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G->isConstant(), Linkage, NewInitializer, "", G,
        G->getThreadLocalMode(), G->getAddressSpace());
    NewGlobal->copyAttributesFrom(G);
    NewGlobal->setComdat(G->getComdat());
    NewGlobal->setAlignment(Align(getMinRedzoneSizeForGlobal()));
    // Don't fold globals with redzones. ODR violation detector and redzone
    // poisoning implicitly creates a dependence on the global's address, so it
    // is no longer valid for it to be marked unnamed_addr.
    NewGlobal->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

    // Move null-terminated C strings to "__asan_cstring" section on Darwin.
    if (TargetTriple.isOSBinFormatMachO() && !G->hasSection() &&
        G->isConstant()) {
      auto Seq = dyn_cast<ConstantDataSequential>(G->getInitializer());
      if (Seq && Seq->isCString())
        NewGlobal->setSection("__TEXT,__asan_cstring,regular");
    }

    // Transfer the debug info and type metadata.  The payload starts at offset
    // zero so we can copy the metadata over as is.
    NewGlobal->copyMetadata(G, 0);

    Value *Indices2[2];
    Indices2[0] = IRB.getInt32(0);
    Indices2[1] = IRB.getInt32(0);

    G->replaceAllUsesWith(
        ConstantExpr::getGetElementPtr(NewTy, NewGlobal, Indices2, true));
    NewGlobal->takeName(G);
    G->eraseFromParent();
    NewGlobals[i] = NewGlobal;

    Constant *ODRIndicator = ConstantPointerNull::get(PtrTy);
    GlobalValue *InstrumentedGlobal = NewGlobal;

    bool CanUsePrivateAliases =
        TargetTriple.isOSBinFormatELF() || TargetTriple.isOSBinFormatMachO() ||
        TargetTriple.isOSBinFormatWasm();
    if (CanUsePrivateAliases && UsePrivateAlias) {
      // Create local alias for NewGlobal to avoid crash on ODR between
      // instrumented and non-instrumented libraries.
      InstrumentedGlobal =
          GlobalAlias::create(GlobalValue::PrivateLinkage, "", NewGlobal);
    }

    // ODR should not happen for local linkage.
    if (NewGlobal->hasLocalLinkage()) {
      ODRIndicator =
          ConstantExpr::getIntToPtr(ConstantInt::get(IntptrTy, -1), PtrTy);
    } else if (UseOdrIndicator) {
      // With local aliases, we need to provide another externally visible
      // symbol __odr_asan_XXX to detect ODR violation.
      auto *ODRIndicatorSym =
          new GlobalVariable(M, IRB.getInt8Ty(), false, Linkage,
                             Constant::getNullValue(IRB.getInt8Ty()),
                             kODRGenPrefix + NameForGlobal, nullptr,
                             NewGlobal->getThreadLocalMode());

      // Set meaningful attributes for indicator symbol.
      ODRIndicatorSym->setVisibility(NewGlobal->getVisibility());
      ODRIndicatorSym->setDLLStorageClass(NewGlobal->getDLLStorageClass());
      ODRIndicatorSym->setAlignment(Align(1));
      ODRIndicator = ODRIndicatorSym;
    }

    Constant *Initializer = ConstantStruct::get(
        GlobalStructTy,
        ConstantExpr::getPointerCast(InstrumentedGlobal, IntptrTy),
        ConstantInt::get(IntptrTy, SizeInBytes),
        ConstantInt::get(IntptrTy, SizeInBytes + RightRedzoneSize),
        ConstantExpr::getPointerCast(Name, IntptrTy),
        ConstantExpr::getPointerCast(getOrCreateModuleName(), IntptrTy),
        ConstantInt::get(IntptrTy, MD.IsDynInit),
        Constant::getNullValue(IntptrTy),
        ConstantExpr::getPointerCast(ODRIndicator, IntptrTy));

    LLVM_DEBUG(dbgs() << "NEW GLOBAL: " << *NewGlobal << "\n");

    Initializers[i] = Initializer;
  }

  // Add instrumented globals to llvm.compiler.used list to avoid LTO from
  // ConstantMerge'ing them.
  SmallVector<GlobalValue *, 16> GlobalsToAddToUsedList;
  for (size_t i = 0; i < n; i++) {
    GlobalVariable *G = NewGlobals[i];
    if (G->getName().empty()) continue;
    GlobalsToAddToUsedList.push_back(G);
  }
  appendToCompilerUsed(M, ArrayRef<GlobalValue *>(GlobalsToAddToUsedList));

  if (UseGlobalsGC && TargetTriple.isOSBinFormatELF()) {
    // Use COMDAT and register globals even if n == 0 to ensure that (a) the
    // linkage unit will only have one module constructor, and (b) the register
    // function will be called. The module destructor is not created when n ==
    // 0.
    *CtorComdat = true;
    instrumentGlobalsELF(IRB, NewGlobals, Initializers, getUniqueModuleId(&M));
  } else if (n == 0) {
    // When UseGlobalsGC is false, COMDAT can still be used if n == 0, because
    // all compile units will have identical module constructor/destructor.
    *CtorComdat = TargetTriple.isOSBinFormatELF();
  } else {
    *CtorComdat = false;
    if (UseGlobalsGC && TargetTriple.isOSBinFormatCOFF()) {
      InstrumentGlobalsCOFF(IRB, NewGlobals, Initializers);
    } else if (UseGlobalsGC && ShouldUseMachOGlobalsSection()) {
      InstrumentGlobalsMachO(IRB, NewGlobals, Initializers);
    } else {
      InstrumentGlobalsWithMetadataArray(IRB, NewGlobals, Initializers);
    }
  }

  // Create calls for poisoning before initializers run and unpoisoning after.
  if (ClInitializers)
    createInitializerPoisonCalls();

  LLVM_DEBUG(dbgs() << M);
}

uint64_t
ModuleAddressSanitizer::getRedzoneSizeForGlobal(uint64_t SizeInBytes) const {
  constexpr uint64_t kMaxRZ = 1 << 18;
  const uint64_t MinRZ = getMinRedzoneSizeForGlobal();

  uint64_t RZ = 0;
  if (SizeInBytes <= MinRZ / 2) {
    // Reduce redzone size for small size objects, e.g. int, char[1]. MinRZ is
    // at least 32 bytes, optimize when SizeInBytes is less than or equal to
    // half of MinRZ.
    RZ = MinRZ - SizeInBytes;
  } else {
    // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 * SizeInBytes.
    RZ = std::clamp((SizeInBytes / MinRZ / 4) * MinRZ, MinRZ, kMaxRZ);

    // Round up to multiple of MinRZ.
    if (SizeInBytes % MinRZ)
      RZ += MinRZ - (SizeInBytes % MinRZ);
  }

  assert((RZ + SizeInBytes) % MinRZ == 0);

  return RZ;
}

int ModuleAddressSanitizer::GetAsanVersion() const {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  bool isAndroid = M.getTargetTriple().isAndroid();
  int Version = 8;
  // 32-bit Android is one version ahead because of the switch to dynamic
  // shadow.
  Version += (LongSize == 32 && isAndroid);
  return Version;
}

GlobalVariable *ModuleAddressSanitizer::getOrCreateModuleName() {
  if (!ModuleName) {
    // We shouldn't merge same module names, as this string serves as unique
    // module ID in runtime.
    ModuleName =
        createPrivateGlobalForString(M, M.getModuleIdentifier(),
                                     /*AllowMerging*/ false, genName("module"));
  }
  return ModuleName;
}

bool ModuleAddressSanitizer::instrumentModule() {
  initializeCallbacks();

  for (Function &F : M)
    removeASanIncompatibleFnAttributes(F, /*ReadsArgMem=*/false);

  // Create a module constructor. A destructor is created lazily because not all
  // platforms, and not all modules need it.
  if (ConstructorKind == AsanCtorKind::Global) {
    if (CompileKernel) {
      // The kernel always builds with its own runtime, and therefore does not
      // need the init and version check calls.
      AsanCtorFunction = createSanitizerCtor(M, kAsanModuleCtorName);
    } else {
      std::string AsanVersion = std::to_string(GetAsanVersion());
      std::string VersionCheckName =
          InsertVersionCheck ? (kAsanVersionCheckNamePrefix + AsanVersion) : "";
      std::tie(AsanCtorFunction, std::ignore) =
          createSanitizerCtorAndInitFunctions(
              M, kAsanModuleCtorName, kAsanInitName, /*InitArgTypes=*/{},
              /*InitArgs=*/{}, VersionCheckName);
    }
  }

  bool CtorComdat = true;
  if (ClGlobals) {
    assert(AsanCtorFunction || ConstructorKind == AsanCtorKind::None);
    if (AsanCtorFunction) {
      IRBuilder<> IRB(AsanCtorFunction->getEntryBlock().getTerminator());
      instrumentGlobals(IRB, &CtorComdat);
    } else {
      IRBuilder<> IRB(*C);
      instrumentGlobals(IRB, &CtorComdat);
    }
  }

  if (TargetTriple.isSPIROrSPIRV()) {
    if (ClSpirOffloadLocals) {
      IRBuilder<> IRB(*C);
      instrumentSyclStaticLocalMemory(IRB);
    }
    if (ClDeviceGlobals) {
      IRBuilder<> IRB(*C);
      instrumentDeviceGlobal(IRB);
    }
  }

  const uint64_t Priority = GetCtorAndDtorPriority(TargetTriple);

  // Put the constructor and destructor in comdat if both
  // (1) global instrumentation is not TU-specific
  // (2) target is ELF.
  if (UseCtorComdat && TargetTriple.isOSBinFormatELF() && CtorComdat) {
    if (AsanCtorFunction) {
      AsanCtorFunction->setComdat(M.getOrInsertComdat(kAsanModuleCtorName));
      appendToGlobalCtors(M, AsanCtorFunction, Priority, AsanCtorFunction);
    }
    if (AsanDtorFunction) {
      AsanDtorFunction->setComdat(M.getOrInsertComdat(kAsanModuleDtorName));
      appendToGlobalDtors(M, AsanDtorFunction, Priority, AsanDtorFunction);
    }
  } else {
    if (AsanCtorFunction)
      appendToGlobalCtors(M, AsanCtorFunction, Priority);
    if (AsanDtorFunction)
      appendToGlobalDtors(M, AsanDtorFunction, Priority);
  }

  return true;
}

void AddressSanitizer::initializeCallbacks(const TargetLibraryInfo *TLI) {
  IRBuilder<> IRB(*C);
  // Create __asan_report* callbacks.
  // IsWrite, TypeSize and Exp are encoded in the function name.
  for (int Exp = 0; Exp < 2; Exp++) {
    for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
      const std::string TypeStr = AccessIsWrite ? "store" : "load";
      const std::string ExpStr = Exp ? "exp_" : "";
      const std::string EndingStr = Recover ? "_noabort" : "";

      SmallVector<Type *, 3> Args2 = {IntptrTy, IntptrTy};
      SmallVector<Type *, 2> Args1{1, IntptrTy};
      AttributeList AL2;
      AttributeList AL1;
      if (Exp) {
        Type *ExpType = Type::getInt32Ty(*C);
        Args2.push_back(ExpType);
        Args1.push_back(ExpType);
        if (auto AK = TLI->getExtAttrForI32Param(false)) {
          AL2 = AL2.addParamAttribute(*C, 2, AK);
          AL1 = AL1.addParamAttribute(*C, 1, AK);
        }
      }

      // __asan_loadX/__asan_storeX(
      //   ...
      //   char* file,
      //   unsigned int line,
      //   char* func
      // )
      if (TargetTriple.isSPIROrSPIRV()) {
        auto *Int8PtrTy = PointerType::get(*C, kSpirOffloadConstantAS);

        Args1.push_back(Int8PtrTy);            // file
        Args1.push_back(Type::getInt32Ty(*C)); // line
        Args1.push_back(Int8PtrTy);            // func

        Args2.push_back(Int8PtrTy);            // file
        Args2.push_back(Type::getInt32Ty(*C)); // line
        Args2.push_back(Int8PtrTy);            // func

        for (size_t AddressSpaceIndex = 0;
             AddressSpaceIndex < kNumberOfAddressSpace; AddressSpaceIndex++) {
          AsanMemoryAccessCallbackSizedAS
              [AccessIsWrite][Exp][AddressSpaceIndex] = M.getOrInsertFunction(
                  ClMemoryAccessCallbackPrefix + ExpStr + TypeStr + "N" +
                      "_as" + itostr(AddressSpaceIndex) + EndingStr,
                  FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);

          for (size_t AccessSizeIndex = 0;
               AccessSizeIndex < kNumberOfAccessSizes; AccessSizeIndex++) {
            const std::string Suffix = TypeStr +
                                       itostr(1ULL << AccessSizeIndex) + "_as" +
                                       itostr(AddressSpaceIndex);
            AsanMemoryAccessCallbackAS[AccessIsWrite][Exp][AccessSizeIndex]
                                      [AddressSpaceIndex] =
                                          M.getOrInsertFunction(
                                              ClMemoryAccessCallbackPrefix +
                                                  ExpStr + Suffix + EndingStr,
                                              FunctionType::get(IRB.getVoidTy(),
                                                                Args1, false),
                                              AL1);
          }
        }

        continue;
      }
      AsanErrorCallbackSized[AccessIsWrite][Exp] = M.getOrInsertFunction(
          kAsanReportErrorTemplate + ExpStr + TypeStr + "_n" + EndingStr,
          FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);

      AsanMemoryAccessCallbackSized[AccessIsWrite][Exp] = M.getOrInsertFunction(
          ClMemoryAccessCallbackPrefix + ExpStr + TypeStr + "N" + EndingStr,
          FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);

      for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
           AccessSizeIndex++) {
        const std::string Suffix = TypeStr + itostr(1ULL << AccessSizeIndex);
        AsanErrorCallback[AccessIsWrite][Exp][AccessSizeIndex] =
            M.getOrInsertFunction(
                kAsanReportErrorTemplate + ExpStr + Suffix + EndingStr,
                FunctionType::get(IRB.getVoidTy(), Args1, false), AL1);

        AsanMemoryAccessCallback[AccessIsWrite][Exp][AccessSizeIndex] =
            M.getOrInsertFunction(
                ClMemoryAccessCallbackPrefix + ExpStr + Suffix + EndingStr,
                FunctionType::get(IRB.getVoidTy(), Args1, false), AL1);
      }
    }
  }

  const std::string MemIntrinCallbackPrefix =
      (CompileKernel && !ClKasanMemIntrinCallbackPrefix)
          ? std::string("")
          : ClMemoryAccessCallbackPrefix;
  AsanMemmove = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memmove",
                                      PtrTy, PtrTy, PtrTy, IntptrTy);
  AsanMemcpy = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memcpy", PtrTy,
                                     PtrTy, PtrTy, IntptrTy);
  AsanMemset = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memset",
                                     TLI->getAttrList(C, {1}, /*Signed=*/false),
                                     PtrTy, PtrTy, IRB.getInt32Ty(), IntptrTy);

  AsanHandleNoReturnFunc =
      M.getOrInsertFunction(kAsanHandleNoReturnName, IRB.getVoidTy());

  AsanPtrCmpFunction =
      M.getOrInsertFunction(kAsanPtrCmp, IRB.getVoidTy(), IntptrTy, IntptrTy);
  AsanPtrSubFunction =
      M.getOrInsertFunction(kAsanPtrSub, IRB.getVoidTy(), IntptrTy, IntptrTy);
  if (Mapping.InGlobal)
    AsanShadowGlobal = M.getOrInsertGlobal("__asan_shadow",
                                           ArrayType::get(IRB.getInt8Ty(), 0));

  if (TargetTriple.isSPIROrSPIRV()) {
    // __asan_set_shadow_dynamic_local(
    //   uptr ptr,
    //   uint32_t num_args
    // )
    AsanSetShadowDynamicLocalFunc = M.getOrInsertFunction(
        "__asan_set_shadow_dynamic_local", IRB.getVoidTy(), IntptrTy, Int32Ty);

    // __asan_unpoison_shadow_dynamic_local(
    //   uptr ptr,
    //   uint32_t num_args
    // )
    AsanUnpoisonShadowDynamicLocalFunc =
        M.getOrInsertFunction("__asan_unpoison_shadow_dynamic_local",
                              IRB.getVoidTy(), IntptrTy, Int32Ty);

    AsanLaunchInfo = M.getOrInsertGlobal(
        "__AsanLaunchInfo", PointerType::get(*C, kSpirOffloadGlobalAS), [&] {
          return new GlobalVariable(
              M, PointerType::get(*C, kSpirOffloadGlobalAS), false,
              GlobalVariable::ExternalLinkage, nullptr, "__AsanLaunchInfo",
              nullptr, GlobalVariable::NotThreadLocal, kSpirOffloadLocalAS);
        });

    AsanMemToShadow = M.getOrInsertFunction(kAsanMemToShadow, IntptrTy,
                                            IntptrTy, Type::getInt32Ty(*C));
  }

  AMDGPUAddressShared =
      M.getOrInsertFunction(kAMDGPUAddressSharedName, IRB.getInt1Ty(), PtrTy);
  AMDGPUAddressPrivate =
      M.getOrInsertFunction(kAMDGPUAddressPrivateName, IRB.getInt1Ty(), PtrTy);
}

bool AddressSanitizer::maybeInsertAsanInitAtFunctionEntry(Function &F) {
  // For each NSObject descendant having a +load method, this method is invoked
  // by the ObjC runtime before any of the static constructors is called.
  // Therefore we need to instrument such methods with a call to __asan_init
  // at the beginning in order to initialize our runtime before any access to
  // the shadow memory.
  // We cannot just ignore these methods, because they may call other
  // instrumented functions.
  if (F.getName().contains(" load]")) {
    FunctionCallee AsanInitFunction =
        declareSanitizerInitFunction(*F.getParent(), kAsanInitName, {});
    IRBuilder<> IRB(&F.front(), F.front().begin());
    IRB.CreateCall(AsanInitFunction, {});
    return true;
  }
  return false;
}

bool AddressSanitizer::maybeInsertDynamicShadowAtFunctionEntry(Function &F) {
  // Generate code only when dynamic addressing is needed.
  if (Mapping.Offset != kDynamicShadowSentinel)
    return false;

  IRBuilder<> IRB(&F.front().front());
  if (Mapping.InGlobal) {
    if (ClWithIfuncSuppressRemat) {
      // An empty inline asm with input reg == output reg.
      // An opaque pointer-to-int cast, basically.
      InlineAsm *Asm = InlineAsm::get(
          FunctionType::get(IntptrTy, {AsanShadowGlobal->getType()}, false),
          StringRef(""), StringRef("=r,0"),
          /*hasSideEffects=*/false);
      LocalDynamicShadow =
          IRB.CreateCall(Asm, {AsanShadowGlobal}, ".asan.shadow");
    } else {
      LocalDynamicShadow =
          IRB.CreatePointerCast(AsanShadowGlobal, IntptrTy, ".asan.shadow");
    }
  } else {
    Value *GlobalDynamicAddress = F.getParent()->getOrInsertGlobal(
        kAsanShadowMemoryDynamicAddress, IntptrTy);
    LocalDynamicShadow = IRB.CreateLoad(IntptrTy, GlobalDynamicAddress);
  }
  return true;
}

void AddressSanitizer::markEscapedLocalAllocas(Function &F) {
  // Find the one possible call to llvm.localescape and pre-mark allocas passed
  // to it as uninteresting. This assumes we haven't started processing allocas
  // yet. This check is done up front because iterating the use list in
  // isInterestingAlloca would be algorithmically slower.
  assert(ProcessedAllocas.empty() && "must process localescape before allocas");

  // Try to get the declaration of llvm.localescape. If it's not in the module,
  // we can exit early.
  if (!F.getParent()->getFunction("llvm.localescape")) return;

  // Look for a call to llvm.localescape call in the entry block. It can't be in
  // any other block.
  for (Instruction &I : F.getEntryBlock()) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
    if (II && II->getIntrinsicID() == Intrinsic::localescape) {
      // We found a call. Mark all the allocas passed in as uninteresting.
      for (Value *Arg : II->args()) {
        AllocaInst *AI = dyn_cast<AllocaInst>(Arg->stripPointerCasts());
        assert(AI && AI->isStaticAlloca() &&
               "non-static alloca arg to localescape");
        ProcessedAllocas[AI] = false;
      }
      break;
    }
  }
}

bool AddressSanitizer::suppressInstrumentationSiteForDebug(int &Instrumented) {
  bool ShouldInstrument =
      ClDebugMin < 0 || ClDebugMax < 0 ||
      (Instrumented >= ClDebugMin && Instrumented <= ClDebugMax);
  Instrumented++;
  return !ShouldInstrument;
}

bool AddressSanitizer::instrumentFunction(Function &F,
                                          const TargetLibraryInfo *TLI) {
  bool FunctionModified = false;

  // Do not apply any instrumentation for naked functions.
  if (F.hasFnAttribute(Attribute::Naked))
    return FunctionModified;

  // If needed, insert __asan_init before checking for SanitizeAddress attr.
  // This function needs to be called even if the function body is not
  // instrumented.
  if (maybeInsertAsanInitAtFunctionEntry(F))
    FunctionModified = true;

  // Leave if the function doesn't need instrumentation.
  if (!F.hasFnAttribute(Attribute::SanitizeAddress)) return FunctionModified;

  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return FunctionModified;

  LLVM_DEBUG(dbgs() << "ASAN instrumenting:\n" << F << "\n");

  initializeCallbacks(TLI);

  FunctionStateRAII CleanupObj(this);

  RuntimeCallInserter RTCI(F);

  FunctionModified |= maybeInsertDynamicShadowAtFunctionEntry(F);

  // We can't instrument allocas used with llvm.localescape. Only static allocas
  // can be passed to that intrinsic.
  markEscapedLocalAllocas(F);

  // We want to instrument every address only once per basic block (unless there
  // are calls between uses).
  SmallPtrSet<Value *, 16> TempsToInstrument;
  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
  SmallVector<MemIntrinsic *, 16> IntrinToInstrument;
  SmallVector<Instruction *, 8> NoReturnCalls;
  SmallVector<BasicBlock *, 16> AllBlocks;
  SmallVector<Instruction *, 16> PointerComparisonsOrSubtracts;

  // Fill the set of memory operations to instrument.
  for (auto &BB : F) {
    AllBlocks.push_back(&BB);
    TempsToInstrument.clear();
    int NumInsnsPerBB = 0;
    for (auto &Inst : BB) {
      if (LooksLikeCodeInBug11395(&Inst)) return false;
      // Skip instructions inserted by another instrumentation.
      if (Inst.hasMetadata(LLVMContext::MD_nosanitize))
        continue;
      SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
      getInterestingMemoryOperands(&Inst, InterestingOperands);

      if (!InterestingOperands.empty()) {
        for (auto &Operand : InterestingOperands) {
          if (ClOpt && ClOptSameTemp) {
            Value *Ptr = Operand.getPtr();
            // If we have a mask, skip instrumentation if we've already
            // instrumented the full object. But don't add to TempsToInstrument
            // because we might get another load/store with a different mask.
            if (Operand.MaybeMask) {
              if (TempsToInstrument.count(Ptr))
                continue; // We've seen this (whole) temp in the current BB.
            } else {
              if (!TempsToInstrument.insert(Ptr).second)
                continue; // We've seen this temp in the current BB.
            }
          }
          OperandsToInstrument.push_back(Operand);
          NumInsnsPerBB++;
        }
      } else if (((ClInvalidPointerPairs || ClInvalidPointerCmp) &&
                  isInterestingPointerComparison(&Inst)) ||
                 ((ClInvalidPointerPairs || ClInvalidPointerSub) &&
                  isInterestingPointerSubtraction(&Inst))) {
        PointerComparisonsOrSubtracts.push_back(&Inst);
      } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst)) {
        // ok, take it.
        IntrinToInstrument.push_back(MI);
        NumInsnsPerBB++;
      } else {
        if (auto *CB = dyn_cast<CallBase>(&Inst)) {
          // On device side, the only non return cases should be *.trap or
          // assert, and none of these cases need to be handles.
          if (!TargetTriple.isSPIROrSPIRV()) {
            // A call inside BB.
            TempsToInstrument.clear();
            if (CB->doesNotReturn())
              NoReturnCalls.push_back(CB);
          }
        }
        if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);
      }
      if (NumInsnsPerBB >= ClMaxInsnsToInstrumentPerBB) break;
    }
  }

  bool UseCalls = (InstrumentationWithCallsThreshold >= 0 &&
                   OperandsToInstrument.size() + IntrinToInstrument.size() >
                       (unsigned)InstrumentationWithCallsThreshold);
  const DataLayout &DL = F.getDataLayout();
  ObjectSizeOffsetVisitor ObjSizeVis(DL, TLI, F.getContext());

  // Instrument.
  int NumInstrumented = 0;
  for (auto &Operand : OperandsToInstrument) {
    if (!suppressInstrumentationSiteForDebug(NumInstrumented))
      instrumentMop(ObjSizeVis, Operand, UseCalls,
                    F.getDataLayout(), RTCI);
    FunctionModified = true;
  }
  if (!TargetTriple.isSPIROrSPIRV()) {
    for (auto *Inst : IntrinToInstrument) {
      if (!suppressInstrumentationSiteForDebug(NumInstrumented))
        instrumentMemIntrinsic(Inst, RTCI);
      FunctionModified = true;
    }
  }

  FunctionStackPoisoner FSP(F, *this, RTCI);
  bool ChangedStack = FSP.runOnFunction();

  // We must unpoison the stack before NoReturn calls (throw, _exit, etc).
  // See e.g. https://github.com/google/sanitizers/issues/37
  for (auto *CI : NoReturnCalls) {
    IRBuilder<> IRB(CI);
    RTCI.createRuntimeCall(IRB, AsanHandleNoReturnFunc, {});
  }

  for (auto *Inst : PointerComparisonsOrSubtracts) {
    instrumentPointerComparisonOrSubtraction(Inst, RTCI);
    FunctionModified = true;
  }

  if (ChangedStack || !NoReturnCalls.empty())
    FunctionModified = true;

  // We need to instrument dynamic local arguments after stack poisoner
  if (TargetTriple.isSPIROrSPIRV()) {
    if (ClSpirOffloadLocals && F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      FunctionModified |= instrumentSyclDynamicLocalMemory(F);
    }
  }

  LLVM_DEBUG(dbgs() << "ASAN done instrumenting: " << FunctionModified << " "
                    << F << "\n");

  return FunctionModified;
}

// Workaround for bug 11395: we don't want to instrument stack in functions
// with large assembly blobs (32-bit only), otherwise reg alloc may crash.
// FIXME: remove once the bug 11395 is fixed.
bool AddressSanitizer::LooksLikeCodeInBug11395(Instruction *I) {
  if (LongSize != 32) return false;
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || !CI->isInlineAsm()) return false;
  if (CI->arg_size() <= 5)
    return false;
  // We have inline assembly with quite a few arguments.
  return true;
}

void FunctionStackPoisoner::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  if (ASan.UseAfterReturn == AsanDetectStackUseAfterReturnMode::Always ||
      ASan.UseAfterReturn == AsanDetectStackUseAfterReturnMode::Runtime) {
    const char *MallocNameTemplate =
        ASan.UseAfterReturn == AsanDetectStackUseAfterReturnMode::Always
            ? kAsanStackMallocAlwaysNameTemplate
            : kAsanStackMallocNameTemplate;
    for (int Index = 0; Index <= kMaxAsanStackMallocSizeClass; Index++) {
      std::string Suffix = itostr(Index);
      AsanStackMallocFunc[Index] = M.getOrInsertFunction(
          MallocNameTemplate + Suffix, IntptrTy, IntptrTy);
      AsanStackFreeFunc[Index] =
          M.getOrInsertFunction(kAsanStackFreeNameTemplate + Suffix,
                                IRB.getVoidTy(), IntptrTy, IntptrTy);
    }
  }
  if (ASan.UseAfterScope) {
    AsanPoisonStackMemoryFunc = M.getOrInsertFunction(
        kAsanPoisonStackMemoryName, IRB.getVoidTy(), IntptrTy, IntptrTy);
    AsanUnpoisonStackMemoryFunc = M.getOrInsertFunction(
        kAsanUnpoisonStackMemoryName, IRB.getVoidTy(), IntptrTy, IntptrTy);
  }

  for (size_t Val : {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0xf1, 0xf2,
                     0xf3, 0xf5, 0xf8}) {
    std::ostringstream Name;
    Name << kAsanSetShadowPrefix;
    Name << std::setw(2) << std::setfill('0') << std::hex << Val;
    AsanSetShadowFunc[Val] =
        M.getOrInsertFunction(Name.str(), IRB.getVoidTy(), IntptrTy, IntptrTy);
  }
  AsanSetShadowPrivateFunc =
      M.getOrInsertFunction("__asan_set_shadow_private", IRB.getVoidTy(),
                            IntptrTy, IntptrTy, IRB.getInt8Ty());

  AsanAllocaPoisonFunc = M.getOrInsertFunction(
      kAsanAllocaPoison, IRB.getVoidTy(), IntptrTy, IntptrTy);
  AsanAllocasUnpoisonFunc = M.getOrInsertFunction(
      kAsanAllocasUnpoison, IRB.getVoidTy(), IntptrTy, IntptrTy);
}

void FunctionStackPoisoner::copyToShadowInline(ArrayRef<uint8_t> ShadowMask,
                                               ArrayRef<uint8_t> ShadowBytes,
                                               size_t Begin, size_t End,
                                               IRBuilder<> &IRB,
                                               Value *ShadowBase) {
  if (Begin >= End)
    return;

  const size_t LargestStoreSizeInBytes =
      std::min<size_t>(sizeof(uint64_t), ASan.LongSize / 8);

  const bool IsLittleEndian = F.getDataLayout().isLittleEndian();

  // Poison given range in shadow using larges store size with out leading and
  // trailing zeros in ShadowMask. Zeros never change, so they need neither
  // poisoning nor up-poisoning. Still we don't mind if some of them get into a
  // middle of a store.
  for (size_t i = Begin; i < End;) {
    if (!ShadowMask[i]) {
      assert(!ShadowBytes[i]);
      ++i;
      continue;
    }

    size_t StoreSizeInBytes = LargestStoreSizeInBytes;
    // Fit store size into the range.
    while (StoreSizeInBytes > End - i)
      StoreSizeInBytes /= 2;

    // Minimize store size by trimming trailing zeros.
    for (size_t j = StoreSizeInBytes - 1; j && !ShadowMask[i + j]; --j) {
      while (j <= StoreSizeInBytes / 2)
        StoreSizeInBytes /= 2;
    }

    uint64_t Val = 0;
    for (size_t j = 0; j < StoreSizeInBytes; j++) {
      if (IsLittleEndian)
        Val |= (uint64_t)ShadowBytes[i + j] << (8 * j);
      else
        Val = (Val << 8) | ShadowBytes[i + j];
    }

    Value *Ptr = IRB.CreateAdd(ShadowBase, ConstantInt::get(IntptrTy, i));
    Value *Poison = IRB.getIntN(StoreSizeInBytes * 8, Val);
    IRB.CreateAlignedStore(
        Poison, IRB.CreateIntToPtr(Ptr, PointerType::getUnqual(Poison->getContext())),
        Align(1));

    i += StoreSizeInBytes;
  }
}

void FunctionStackPoisoner::copyToShadow(ArrayRef<uint8_t> ShadowMask,
                                         ArrayRef<uint8_t> ShadowBytes,
                                         IRBuilder<> &IRB, Value *ShadowBase,
                                         bool ForceOutline) {
  copyToShadow(ShadowMask, ShadowBytes, 0, ShadowMask.size(), IRB, ShadowBase,
               ForceOutline);
}

void FunctionStackPoisoner::copyToShadow(ArrayRef<uint8_t> ShadowMask,
                                         ArrayRef<uint8_t> ShadowBytes,
                                         size_t Begin, size_t End,
                                         IRBuilder<> &IRB, Value *ShadowBase,
                                         bool ForceOutline) {
  assert(ShadowMask.size() == ShadowBytes.size());
  size_t Done = Begin;
  for (size_t i = Begin, j = Begin + 1; i < End; i = j++) {
    if (!ShadowMask[i]) {
      assert(!ShadowBytes[i]);
      continue;
    }
    uint8_t Val = ShadowBytes[i];
    if (!AsanSetShadowFunc[Val] && !ForceOutline)
      continue;

    // Skip same values.
    for (; j < End && ShadowMask[j] && Val == ShadowBytes[j]; ++j) {
    }

    if (ForceOutline) {
      RTCI.createRuntimeCall(
          IRB, AsanSetShadowPrivateFunc,
          {IRB.CreateAdd(ShadowBase, ConstantInt::get(IntptrTy, i)),
           ConstantInt::get(IntptrTy, j - i),
           ConstantInt::get(IRB.getInt8Ty(), Val)});
    } else if (j - i >= ASan.MaxInlinePoisoningSize) {
      copyToShadowInline(ShadowMask, ShadowBytes, Done, i, IRB, ShadowBase);
      RTCI.createRuntimeCall(
          IRB, AsanSetShadowFunc[Val],
          {IRB.CreateAdd(ShadowBase, ConstantInt::get(IntptrTy, i)),
           ConstantInt::get(IntptrTy, j - i)});
      Done = j;
    }
  }

  if (!ForceOutline)
    copyToShadowInline(ShadowMask, ShadowBytes, Done, End, IRB, ShadowBase);
}

// Fake stack allocator (asan_fake_stack.h) has 11 size classes
// for every power of 2 from kMinStackMallocSize to kMaxAsanStackMallocSizeClass
static int StackMallocSizeClass(uint64_t LocalStackSize) {
  assert(LocalStackSize <= kMaxStackMallocSize);
  uint64_t MaxSize = kMinStackMallocSize;
  for (int i = 0;; i++, MaxSize *= 2)
    if (LocalStackSize <= MaxSize) return i;
  llvm_unreachable("impossible LocalStackSize");
}

void FunctionStackPoisoner::copyArgsPassedByValToAllocas() {
  Instruction *CopyInsertPoint = &F.front().front();
  if (CopyInsertPoint == ASan.LocalDynamicShadow) {
    // Insert after the dynamic shadow location is determined
    CopyInsertPoint = CopyInsertPoint->getNextNode();
    assert(CopyInsertPoint);
  }
  IRBuilder<> IRB(CopyInsertPoint);
  const DataLayout &DL = F.getDataLayout();
  for (Argument &Arg : F.args()) {
    if (Arg.hasByValAttr()) {
      Type *Ty = Arg.getParamByValType();
      const Align Alignment =
          DL.getValueOrABITypeAlignment(Arg.getParamAlign(), Ty);

      unsigned int AS = Triple(F.getParent()->getTargetTriple()).isSPIROrSPIRV()
                            ? Arg.getType()->getPointerAddressSpace()
                            : DL.getAllocaAddrSpace();

      AllocaInst *AI = IRB.CreateAlloca(
          Ty, AS, nullptr,
          (Arg.hasName() ? Arg.getName() : "Arg" + Twine(Arg.getArgNo())) +
              ".byval");
      AI->setAlignment(Alignment);
      Arg.replaceAllUsesWith(AI);

      uint64_t AllocSize = DL.getTypeAllocSize(Ty);
      IRB.CreateMemCpy(AI, Alignment, &Arg, Alignment, AllocSize);
    }
  }
}

PHINode *FunctionStackPoisoner::createPHI(IRBuilder<> &IRB, Value *Cond,
                                          Value *ValueIfTrue,
                                          Instruction *ThenTerm,
                                          Value *ValueIfFalse) {
  PHINode *PHI = IRB.CreatePHI(IntptrTy, 2);
  BasicBlock *CondBlock = cast<Instruction>(Cond)->getParent();
  PHI->addIncoming(ValueIfFalse, CondBlock);
  BasicBlock *ThenBlock = ThenTerm->getParent();
  PHI->addIncoming(ValueIfTrue, ThenBlock);
  return PHI;
}

Value *FunctionStackPoisoner::createAllocaForLayout(
    IRBuilder<> &IRB, const ASanStackFrameLayout &L, bool Dynamic) {
  AllocaInst *Alloca;
  if (Dynamic) {
    Alloca = IRB.CreateAlloca(IRB.getInt8Ty(),
                              ConstantInt::get(IRB.getInt64Ty(), L.FrameSize),
                              "MyAlloca");
  } else {
    Alloca = IRB.CreateAlloca(ArrayType::get(IRB.getInt8Ty(), L.FrameSize),
                              nullptr, "MyAlloca");
    assert(Alloca->isStaticAlloca());
  }
  assert((ClRealignStack & (ClRealignStack - 1)) == 0);
  uint64_t FrameAlignment = std::max(L.FrameAlignment, uint64_t(ClRealignStack));
  Alloca->setAlignment(Align(FrameAlignment));
  return IRB.CreatePointerCast(Alloca, IntptrTy);
}

void FunctionStackPoisoner::createDynamicAllocasInitStorage() {
  BasicBlock &FirstBB = *F.begin();
  IRBuilder<> IRB(dyn_cast<Instruction>(FirstBB.begin()));
  DynamicAllocaLayout = IRB.CreateAlloca(IntptrTy, nullptr);
  IRB.CreateStore(Constant::getNullValue(IntptrTy), DynamicAllocaLayout);
  DynamicAllocaLayout->setAlignment(Align(32));
}

void FunctionStackPoisoner::processDynamicAllocas() {
  if (!ClInstrumentDynamicAllocas || DynamicAllocaVec.empty()) {
    assert(DynamicAllocaPoisonCallVec.empty());
    return;
  }

  // Insert poison calls for lifetime intrinsics for dynamic allocas.
  for (const auto &APC : DynamicAllocaPoisonCallVec) {
    assert(APC.InsBefore);
    assert(APC.AI);
    assert(ASan.isInterestingAlloca(*APC.AI));
    assert(!APC.AI->isStaticAlloca());

    IRBuilder<> IRB(APC.InsBefore);
    poisonAlloca(APC.AI, APC.Size, IRB, APC.DoPoison);
    // Dynamic allocas will be unpoisoned unconditionally below in
    // unpoisonDynamicAllocas.
    // Flag that we need unpoison static allocas.
  }

  // Handle dynamic allocas.
  createDynamicAllocasInitStorage();
  for (auto &AI : DynamicAllocaVec)
    handleDynamicAllocaCall(AI);
  unpoisonDynamicAllocas();
}

/// Collect instructions in the entry block after \p InsBefore which initialize
/// permanent storage for a function argument. These instructions must remain in
/// the entry block so that uninitialized values do not appear in backtraces. An
/// added benefit is that this conserves spill slots. This does not move stores
/// before instrumented / "interesting" allocas.
static void findStoresToUninstrumentedArgAllocas(
    AddressSanitizer &ASan, Instruction &InsBefore,
    SmallVectorImpl<Instruction *> &InitInsts) {
  Instruction *Start = InsBefore.getNextNonDebugInstruction();
  for (Instruction *It = Start; It; It = It->getNextNonDebugInstruction()) {
    // Argument initialization looks like:
    // 1) store <Argument>, <Alloca> OR
    // 2) <CastArgument> = cast <Argument> to ...
    //    store <CastArgument> to <Alloca>
    // Do not consider any other kind of instruction.
    //
    // Note: This covers all known cases, but may not be exhaustive. An
    // alternative to pattern-matching stores is to DFS over all Argument uses:
    // this might be more general, but is probably much more complicated.
    if (isa<AllocaInst>(It) || isa<CastInst>(It))
      continue;
    if (auto *Store = dyn_cast<StoreInst>(It)) {
      // The store destination must be an alloca that isn't interesting for
      // ASan to instrument. These are moved up before InsBefore, and they're
      // not interesting because allocas for arguments can be mem2reg'd.
      auto *Alloca = dyn_cast<AllocaInst>(Store->getPointerOperand());
      if (!Alloca || ASan.isInterestingAlloca(*Alloca))
        continue;

      Value *Val = Store->getValueOperand();
      bool IsDirectArgInit = isa<Argument>(Val);
      bool IsArgInitViaCast =
          isa<CastInst>(Val) &&
          isa<Argument>(cast<CastInst>(Val)->getOperand(0)) &&
          // Check that the cast appears directly before the store. Otherwise
          // moving the cast before InsBefore may break the IR.
          Val == It->getPrevNonDebugInstruction();
      bool IsArgInit = IsDirectArgInit || IsArgInitViaCast;
      if (!IsArgInit)
        continue;

      if (IsArgInitViaCast)
        InitInsts.push_back(cast<Instruction>(Val));
      InitInsts.push_back(Store);
      continue;
    }

    // Do not reorder past unknown instructions: argument initialization should
    // only involve casts and stores.
    return;
  }
}

static StringRef getAllocaName(AllocaInst *AI) {
  // Alloca could have been renamed for uniqueness. Its true name will have been
  // recorded as an annotation.
  if (AI->hasMetadata(LLVMContext::MD_annotation)) {
    MDTuple *AllocaAnnotations =
        cast<MDTuple>(AI->getMetadata(LLVMContext::MD_annotation));
    for (auto &Annotation : AllocaAnnotations->operands()) {
      if (!isa<MDTuple>(Annotation))
        continue;
      auto AnnotationTuple = cast<MDTuple>(Annotation);
      for (unsigned Index = 0; Index < AnnotationTuple->getNumOperands();
           Index++) {
        // All annotations are strings
        auto MetadataString =
            cast<MDString>(AnnotationTuple->getOperand(Index));
        if (MetadataString->getString() == "alloca_name_altered")
          return cast<MDString>(AnnotationTuple->getOperand(Index + 1))
              ->getString();
      }
    }
  }
  return AI->getName();
}

void FunctionStackPoisoner::processStaticAllocas() {
  if (AllocaVec.empty()) {
    assert(StaticAllocaPoisonCallVec.empty());
    return;
  }

  int StackMallocIdx = -1;
  DebugLoc EntryDebugLocation;
  if (auto SP = F.getSubprogram())
    EntryDebugLocation =
        DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP);

  Instruction *InsBefore = AllocaVec[0];
  IRBuilder<> IRB(InsBefore);

  // Make sure non-instrumented allocas stay in the entry block. Otherwise,
  // debug info is broken, because only entry-block allocas are treated as
  // regular stack slots.
  auto InsBeforeB = InsBefore->getParent();
  assert(InsBeforeB == &F.getEntryBlock());
  for (auto *AI : StaticAllocasToMoveUp)
    if (AI->getParent() == InsBeforeB)
      AI->moveBefore(InsBefore->getIterator());

  // Move stores of arguments into entry-block allocas as well. This prevents
  // extra stack slots from being generated (to house the argument values until
  // they can be stored into the allocas). This also prevents uninitialized
  // values from being shown in backtraces.
  SmallVector<Instruction *, 8> ArgInitInsts;
  findStoresToUninstrumentedArgAllocas(ASan, *InsBefore, ArgInitInsts);
  for (Instruction *ArgInitInst : ArgInitInsts)
    ArgInitInst->moveBefore(InsBefore->getIterator());

  // If we have a call to llvm.localescape, keep it in the entry block.
  if (LocalEscapeCall)
    LocalEscapeCall->moveBefore(InsBefore->getIterator());

  SmallVector<ASanStackVariableDescription, 16> SVD;
  SVD.reserve(AllocaVec.size());
  for (AllocaInst *AI : AllocaVec) {
    StringRef Name = getAllocaName(AI);
    ASanStackVariableDescription D = {Name.data(),
                                      ASan.getAllocaSizeInBytes(*AI),
                                      0,
                                      AI->getAlign().value(),
                                      AI,
                                      0,
                                      0};
    SVD.push_back(D);
  }

  // Minimal header size (left redzone) is 4 pointers,
  // i.e. 32 bytes on 64-bit platforms and 16 bytes in 32-bit platforms.
  uint64_t Granularity = 1ULL << Mapping.Scale;
  uint64_t MinHeaderSize = std::max((uint64_t)ASan.LongSize / 2, Granularity);
  const ASanStackFrameLayout &L =
      ComputeASanStackFrameLayout(SVD, Granularity, MinHeaderSize);

  // Build AllocaToSVDMap for ASanStackVariableDescription lookup.
  DenseMap<const AllocaInst *, ASanStackVariableDescription *> AllocaToSVDMap;
  for (auto &Desc : SVD)
    AllocaToSVDMap[Desc.AI] = &Desc;

  // Update SVD with information from lifetime intrinsics.
  for (const auto &APC : StaticAllocaPoisonCallVec) {
    assert(APC.InsBefore);
    assert(APC.AI);
    assert(ASan.isInterestingAlloca(*APC.AI));
    assert(APC.AI->isStaticAlloca());

    ASanStackVariableDescription &Desc = *AllocaToSVDMap[APC.AI];
    Desc.LifetimeSize = Desc.Size;
    if (const DILocation *FnLoc = EntryDebugLocation.get()) {
      if (const DILocation *LifetimeLoc = APC.InsBefore->getDebugLoc().get()) {
        if (LifetimeLoc->getFile() == FnLoc->getFile())
          if (unsigned Line = LifetimeLoc->getLine())
            Desc.Line = std::min(Desc.Line ? Desc.Line : Line, Line);
      }
    }
  }

  auto DescriptionString = ComputeASanStackFrameDescription(SVD);
  LLVM_DEBUG(dbgs() << DescriptionString << " --- " << L.FrameSize << "\n");
  uint64_t LocalStackSize = L.FrameSize;
  bool DoStackMalloc =
      ASan.UseAfterReturn != AsanDetectStackUseAfterReturnMode::Never &&
      !ASan.CompileKernel && LocalStackSize <= kMaxStackMallocSize;
  bool DoDynamicAlloca = ClDynamicAllocaStack;
  // Don't do dynamic alloca or stack malloc if:
  // 1) There is inline asm: too often it makes assumptions on which registers
  //    are available.
  // 2) There is a returns_twice call (typically setjmp), which is
  //    optimization-hostile, and doesn't play well with introduced indirect
  //    register-relative calculation of local variable addresses.
  DoDynamicAlloca &= !HasInlineAsm && !HasReturnsTwiceCall;
  DoStackMalloc &= !HasInlineAsm && !HasReturnsTwiceCall;

  Value *StaticAlloca =
      DoDynamicAlloca ? nullptr : createAllocaForLayout(IRB, L, false);

  Value *FakeStack;
  Value *LocalStackBase;
  Value *LocalStackBaseAlloca;
  uint8_t DIExprFlags = DIExpression::ApplyOffset;

  if (DoStackMalloc) {
    LocalStackBaseAlloca =
        IRB.CreateAlloca(IntptrTy, nullptr, "asan_local_stack_base");
    if (ASan.UseAfterReturn == AsanDetectStackUseAfterReturnMode::Runtime) {
      // void *FakeStack = __asan_option_detect_stack_use_after_return
      //     ? __asan_stack_malloc_N(LocalStackSize)
      //     : nullptr;
      // void *LocalStackBase = (FakeStack) ? FakeStack :
      //                        alloca(LocalStackSize);
      Constant *OptionDetectUseAfterReturn = F.getParent()->getOrInsertGlobal(
          kAsanOptionDetectUseAfterReturn, IRB.getInt32Ty());
      Value *UseAfterReturnIsEnabled = IRB.CreateICmpNE(
          IRB.CreateLoad(IRB.getInt32Ty(), OptionDetectUseAfterReturn),
          Constant::getNullValue(IRB.getInt32Ty()));
      Instruction *Term =
          SplitBlockAndInsertIfThen(UseAfterReturnIsEnabled, InsBefore, false);
      IRBuilder<> IRBIf(Term);
      StackMallocIdx = StackMallocSizeClass(LocalStackSize);
      assert(StackMallocIdx <= kMaxAsanStackMallocSizeClass);
      Value *FakeStackValue =
          RTCI.createRuntimeCall(IRBIf, AsanStackMallocFunc[StackMallocIdx],
                                 ConstantInt::get(IntptrTy, LocalStackSize));
      IRB.SetInsertPoint(InsBefore);
      FakeStack = createPHI(IRB, UseAfterReturnIsEnabled, FakeStackValue, Term,
                            ConstantInt::get(IntptrTy, 0));
    } else {
      // assert(ASan.UseAfterReturn == AsanDetectStackUseAfterReturnMode:Always)
      // void *FakeStack = __asan_stack_malloc_N(LocalStackSize);
      // void *LocalStackBase = (FakeStack) ? FakeStack :
      //                        alloca(LocalStackSize);
      StackMallocIdx = StackMallocSizeClass(LocalStackSize);
      FakeStack =
          RTCI.createRuntimeCall(IRB, AsanStackMallocFunc[StackMallocIdx],
                                 ConstantInt::get(IntptrTy, LocalStackSize));
    }
    Value *NoFakeStack =
        IRB.CreateICmpEQ(FakeStack, Constant::getNullValue(IntptrTy));
    Instruction *Term =
        SplitBlockAndInsertIfThen(NoFakeStack, InsBefore, false);
    IRBuilder<> IRBIf(Term);
    Value *AllocaValue =
        DoDynamicAlloca ? createAllocaForLayout(IRBIf, L, true) : StaticAlloca;

    IRB.SetInsertPoint(InsBefore);
    LocalStackBase = createPHI(IRB, NoFakeStack, AllocaValue, Term, FakeStack);
    IRB.CreateStore(LocalStackBase, LocalStackBaseAlloca);
    DIExprFlags |= DIExpression::DerefBefore;
  } else {
    // void *FakeStack = nullptr;
    // void *LocalStackBase = alloca(LocalStackSize);
    FakeStack = ConstantInt::get(IntptrTy, 0);
    LocalStackBase =
        DoDynamicAlloca ? createAllocaForLayout(IRB, L, true) : StaticAlloca;
    LocalStackBaseAlloca = LocalStackBase;
  }

  // It shouldn't matter whether we pass an `alloca` or a `ptrtoint` as the
  // dbg.declare address opereand, but passing a `ptrtoint` seems to confuse
  // later passes and can result in dropped variable coverage in debug info.
  Value *LocalStackBaseAllocaPtr =
      isa<PtrToIntInst>(LocalStackBaseAlloca)
          ? cast<PtrToIntInst>(LocalStackBaseAlloca)->getPointerOperand()
          : LocalStackBaseAlloca;
  assert(isa<AllocaInst>(LocalStackBaseAllocaPtr) &&
         "Variable descriptions relative to ASan stack base will be dropped");

  // Replace Alloca instructions with base+offset.
  for (const auto &Desc : SVD) {
    AllocaInst *AI = Desc.AI;
    replaceDbgDeclare(AI, LocalStackBaseAllocaPtr, DIB, DIExprFlags,
                      Desc.Offset);
    Value *NewAllocaPtr = IRB.CreateIntToPtr(
        IRB.CreateAdd(LocalStackBase, ConstantInt::get(IntptrTy, Desc.Offset)),
        AI->getType());
    AI->replaceAllUsesWith(NewAllocaPtr);
  }

  const auto &TargetTriple = Triple(F.getParent()->getTargetTriple());

  // The left-most redzone has enough space for at least 4 pointers.
  Value *BasePlus0 = IRB.CreateIntToPtr(LocalStackBase, IntptrPtrTy);
  // SPIRV doesn't use the following metadata
  if (!TargetTriple.isSPIROrSPIRV()) {
    // Write the Magic value to redzone[0].
    IRB.CreateStore(ConstantInt::get(IntptrTy, kCurrentStackFrameMagic),
                    BasePlus0);
    // Write the frame description constant to redzone[1].
    Value *BasePlus1 = IRB.CreateIntToPtr(
        IRB.CreateAdd(LocalStackBase,
                      ConstantInt::get(IntptrTy, ASan.LongSize / 8)),
        IntptrPtrTy);
    GlobalVariable *StackDescriptionGlobal =
        createPrivateGlobalForString(*F.getParent(), DescriptionString,
                                     /*AllowMerging*/ true, genName("stack"));
    Value *Description =
        IRB.CreatePointerCast(StackDescriptionGlobal, IntptrTy);
    IRB.CreateStore(Description, BasePlus1);
    // Write the PC to redzone[2].
    Value *BasePlus2 = IRB.CreateIntToPtr(
        IRB.CreateAdd(LocalStackBase,
                      ConstantInt::get(IntptrTy, 2 * ASan.LongSize / 8)),
        IntptrPtrTy);
    IRB.CreateStore(IRB.CreatePointerCast(&F, IntptrTy), BasePlus2);
  }

  const auto &ShadowAfterScope = GetShadowBytesAfterScope(SVD, L);

  // Poison the stack red zones at the entry.
  Value *ShadowBase =
      ASan.memToShadow(LocalStackBase, IRB, kSpirOffloadPrivateAS);
  // As mask we must use most poisoned case: red zones and after scope.
  // As bytes we can use either the same or just red zones only.
  copyToShadow(ShadowAfterScope, ShadowAfterScope, IRB, ShadowBase,
               TargetTriple.isSPIROrSPIRV());

  if (!StaticAllocaPoisonCallVec.empty()) {
    const auto &ShadowInScope = GetShadowBytes(SVD, L);

    // Poison static allocas near lifetime intrinsics.
    for (const auto &APC : StaticAllocaPoisonCallVec) {
      const ASanStackVariableDescription &Desc = *AllocaToSVDMap[APC.AI];
      assert(Desc.Offset % L.Granularity == 0);
      size_t Begin = Desc.Offset / L.Granularity;
      size_t End = Begin + (APC.Size + L.Granularity - 1) / L.Granularity;

      IRBuilder<> IRB(APC.InsBefore);
      copyToShadow(ShadowAfterScope,
                   APC.DoPoison ? ShadowAfterScope : ShadowInScope, Begin, End,
                   IRB, ShadowBase);
    }
  }

  SmallVector<uint8_t, 64> ShadowClean(ShadowAfterScope.size(), 0);
  SmallVector<uint8_t, 64> ShadowAfterReturn;

  // (Un)poison the stack before all ret instructions.
  for (Instruction *Ret : RetVec) {
    IRBuilder<> IRBRet(Ret);
    // Mark the current frame as retired.
    IRBRet.CreateStore(ConstantInt::get(IntptrTy, kRetiredStackFrameMagic),
                       BasePlus0);
    if (DoStackMalloc) {
      assert(StackMallocIdx >= 0);
      // if FakeStack != 0  // LocalStackBase == FakeStack
      //     // In use-after-return mode, poison the whole stack frame.
      //     if StackMallocIdx <= 4
      //         // For small sizes inline the whole thing:
      //         memset(ShadowBase, kAsanStackAfterReturnMagic, ShadowSize);
      //         **SavedFlagPtr(FakeStack) = 0
      //     else
      //         __asan_stack_free_N(FakeStack, LocalStackSize)
      // else
      //     <This is not a fake stack; unpoison the redzones>
      Value *Cmp =
          IRBRet.CreateICmpNE(FakeStack, Constant::getNullValue(IntptrTy));
      Instruction *ThenTerm, *ElseTerm;
      SplitBlockAndInsertIfThenElse(Cmp, Ret, &ThenTerm, &ElseTerm);

      IRBuilder<> IRBPoison(ThenTerm);
      if (ASan.MaxInlinePoisoningSize != 0 && StackMallocIdx <= 4) {
        int ClassSize = kMinStackMallocSize << StackMallocIdx;
        ShadowAfterReturn.resize(ClassSize / L.Granularity,
                                 kAsanStackUseAfterReturnMagic);
        copyToShadow(ShadowAfterReturn, ShadowAfterReturn, IRBPoison,
                     ShadowBase);
        Value *SavedFlagPtrPtr = IRBPoison.CreateAdd(
            FakeStack,
            ConstantInt::get(IntptrTy, ClassSize - ASan.LongSize / 8));
        Value *SavedFlagPtr = IRBPoison.CreateLoad(
            IntptrTy, IRBPoison.CreateIntToPtr(SavedFlagPtrPtr, IntptrPtrTy));
        IRBPoison.CreateStore(
            Constant::getNullValue(IRBPoison.getInt8Ty()),
            IRBPoison.CreateIntToPtr(SavedFlagPtr, IRBPoison.getPtrTy()));
      } else {
        // For larger frames call __asan_stack_free_*.
        RTCI.createRuntimeCall(
            IRBPoison, AsanStackFreeFunc[StackMallocIdx],
            {FakeStack, ConstantInt::get(IntptrTy, LocalStackSize)});
      }

      IRBuilder<> IRBElse(ElseTerm);
      copyToShadow(ShadowAfterScope, ShadowClean, IRBElse, ShadowBase);
    } else {
      copyToShadow(ShadowAfterScope, ShadowClean, IRBRet, ShadowBase,
                   TargetTriple.isSPIROrSPIRV());
    }
  }

  // We are done. Remove the old unused alloca instructions.
  for (auto *AI : AllocaVec)
    AI->eraseFromParent();
}

void FunctionStackPoisoner::poisonAlloca(Value *V, uint64_t Size,
                                         IRBuilder<> &IRB, bool DoPoison) {
  // For now just insert the call to ASan runtime.
  Value *AddrArg = IRB.CreatePointerCast(V, IntptrTy);
  Value *SizeArg = ConstantInt::get(IntptrTy, Size);
  RTCI.createRuntimeCall(
      IRB, DoPoison ? AsanPoisonStackMemoryFunc : AsanUnpoisonStackMemoryFunc,
      {AddrArg, SizeArg});
}

// Handling llvm.lifetime intrinsics for a given %alloca:
// (1) collect all llvm.lifetime.xxx(%size, %value) describing the alloca.
// (2) if %size is constant, poison memory for llvm.lifetime.end (to detect
//     invalid accesses) and unpoison it for llvm.lifetime.start (the memory
//     could be poisoned by previous llvm.lifetime.end instruction, as the
//     variable may go in and out of scope several times, e.g. in loops).
// (3) if we poisoned at least one %alloca in a function,
//     unpoison the whole stack frame at function exit.
void FunctionStackPoisoner::handleDynamicAllocaCall(AllocaInst *AI) {
  IRBuilder<> IRB(AI);

  const Align Alignment = std::max(Align(kAllocaRzSize), AI->getAlign());
  const uint64_t AllocaRedzoneMask = kAllocaRzSize - 1;

  Value *Zero = Constant::getNullValue(IntptrTy);
  Value *AllocaRzSize = ConstantInt::get(IntptrTy, kAllocaRzSize);
  Value *AllocaRzMask = ConstantInt::get(IntptrTy, AllocaRedzoneMask);

  // Since we need to extend alloca with additional memory to locate
  // redzones, and OldSize is number of allocated blocks with
  // ElementSize size, get allocated memory size in bytes by
  // OldSize * ElementSize.
  const unsigned ElementSize =
      F.getDataLayout().getTypeAllocSize(AI->getAllocatedType());
  Value *OldSize =
      IRB.CreateMul(IRB.CreateIntCast(AI->getArraySize(), IntptrTy, false),
                    ConstantInt::get(IntptrTy, ElementSize));

  // PartialSize = OldSize % 32
  Value *PartialSize = IRB.CreateAnd(OldSize, AllocaRzMask);

  // Misalign = kAllocaRzSize - PartialSize;
  Value *Misalign = IRB.CreateSub(AllocaRzSize, PartialSize);

  // PartialPadding = Misalign != kAllocaRzSize ? Misalign : 0;
  Value *Cond = IRB.CreateICmpNE(Misalign, AllocaRzSize);
  Value *PartialPadding = IRB.CreateSelect(Cond, Misalign, Zero);

  // AdditionalChunkSize = Alignment + PartialPadding + kAllocaRzSize
  // Alignment is added to locate left redzone, PartialPadding for possible
  // partial redzone and kAllocaRzSize for right redzone respectively.
  Value *AdditionalChunkSize = IRB.CreateAdd(
      ConstantInt::get(IntptrTy, Alignment.value() + kAllocaRzSize),
      PartialPadding);

  Value *NewSize = IRB.CreateAdd(OldSize, AdditionalChunkSize);

  // Insert new alloca with new NewSize and Alignment params.
  AllocaInst *NewAlloca = IRB.CreateAlloca(IRB.getInt8Ty(), NewSize);
  NewAlloca->setAlignment(Alignment);

  // NewAddress = Address + Alignment
  Value *NewAddress =
      IRB.CreateAdd(IRB.CreatePtrToInt(NewAlloca, IntptrTy),
                    ConstantInt::get(IntptrTy, Alignment.value()));

  // Insert __asan_alloca_poison call for new created alloca.
  RTCI.createRuntimeCall(IRB, AsanAllocaPoisonFunc, {NewAddress, OldSize});

  // Store the last alloca's address to DynamicAllocaLayout. We'll need this
  // for unpoisoning stuff.
  IRB.CreateStore(IRB.CreatePtrToInt(NewAlloca, IntptrTy), DynamicAllocaLayout);

  Value *NewAddressPtr = IRB.CreateIntToPtr(NewAddress, AI->getType());

  // Replace all uses of AddessReturnedByAlloca with NewAddressPtr.
  AI->replaceAllUsesWith(NewAddressPtr);

  // We are done. Erase old alloca from parent.
  AI->eraseFromParent();
}

// isSafeAccess returns true if Addr is always inbounds with respect to its
// base object. For example, it is a field access or an array access with
// constant inbounds index.
bool AddressSanitizer::isSafeAccess(ObjectSizeOffsetVisitor &ObjSizeVis,
                                    Value *Addr, TypeSize TypeStoreSize) const {
  if (TypeStoreSize.isScalable())
    // TODO: We can use vscale_range to convert a scalable value to an
    // upper bound on the access size.
    return false;

  SizeOffsetAPInt SizeOffset = ObjSizeVis.compute(Addr);
  if (!SizeOffset.bothKnown())
    return false;

  uint64_t Size = SizeOffset.Size.getZExtValue();
  int64_t Offset = SizeOffset.Offset.getSExtValue();

  // Three checks are required to ensure safety:
  // . Offset >= 0  (since the offset is given from the base ptr)
  // . Size >= Offset  (unsigned)
  // . Size - Offset >= NeededSize  (unsigned)
  return Offset >= 0 && Size >= uint64_t(Offset) &&
         Size - uint64_t(Offset) >= TypeStoreSize / 8;
}
