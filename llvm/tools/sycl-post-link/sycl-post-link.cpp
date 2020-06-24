//===- sycl-post-link.cpp - SYCL post-link device code processing tool ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This source is a collection of utilities run on device code's LLVM IR before
// handing off to back-end for further compilation or emitting SPIRV. The
// utilities are:
// - module splitter to split a big input module into smaller ones
// - specialization constant intrinsic transformation
//===----------------------------------------------------------------------===//

#include "SpecConstants.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SYCLRTShared.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory>

using namespace llvm;
using llvm::util::sycl::cl_intel_devicelib_assert;
using llvm::util::sycl::cl_intel_devicelib_complex;
using llvm::util::sycl::cl_intel_devicelib_complex_fp64;
using llvm::util::sycl::cl_intel_devicelib_math;
using llvm::util::sycl::cl_intel_devicelib_math_fp64;

using string_vector = std::vector<std::string>;
using SpecIDMapTy = std::map<StringRef, unsigned>;

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// Column names in the output file table. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
static constexpr char COL_CODE[] = "Code";
static constexpr char COL_SYM[] = "Symbols";
static constexpr char COL_PROPS[] = "Properties";
static constexpr char DEVICELIB_FUNC_PREFIX[] = "__devicelib_";
// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename{
    cl::Positional, cl::desc("<input bitcode file>"), cl::init("-"),
    cl::value_desc("filename")};

static cl::opt<std::string> OutputDir{
    "out-dir",
    cl::desc(
        "Directory where files listed in the result file table will be output"),
    cl::value_desc("dirname"), cl::cat(PostLinkCat)};

static cl::opt<std::string> OutputFilename{"o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"), cl::cat(PostLinkCat)};

static cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                           cl::cat(PostLinkCat)};

static cl::opt<bool> IROutputOnly{
    "ir-output-only", cl::desc("Output single IR file"), cl::cat(PostLinkCat)};

static cl::opt<bool> OutputAssembly{"S",
                                    cl::desc("Write output as LLVM assembly"),
                                    cl::Hidden, cl::cat(PostLinkCat)};

enum IRSplitMode {
  SPLIT_PER_TU,    // one module per translation unit
  SPLIT_PER_KERNEL // one module per kernel
};

static cl::opt<IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional,
    cl::init(SPLIT_PER_TU),
    cl::values(clEnumValN(SPLIT_PER_TU, "source",
                          "1 output module per source (translation unit)"),
               clEnumValN(SPLIT_PER_KERNEL, "kernel",
                          "1 output module per kernel")),
    cl::cat(PostLinkCat));

static cl::opt<bool> DoSymGen{"symbols",
                              cl::desc("generate exported symbol files"),
                              cl::cat(PostLinkCat)};

enum SpecConstMode { SC_USE_RT_VAL, SC_USE_DEFAULT_VAL };

static cl::opt<SpecConstMode> SpecConstLower{
    "spec-const",
    cl::desc("lower and generate specialization constants information"),
    cl::Optional,
    cl::init(SC_USE_RT_VAL),
    cl::values(
        clEnumValN(SC_USE_RT_VAL, "rt", "spec constants are set at runtime"),
        clEnumValN(SC_USE_DEFAULT_VAL, "default",
                   "set spec constants to C++ defaults")),
    cl::cat(PostLinkCat)};

struct ImagePropSaveInfo {
  bool NeedDeviceLibReqMask;
  bool DoSpecConst;
  bool SetSpecConstAtRT;
  bool SpecConstsMet;
};
// Please update DeviceLibFuncMap if any item is added to or removed from
// fallback device libraries in libdevice.
static std::map<std::string, uint32_t> DeviceLibFuncMap = {
    {"__devicelib_acosf", cl_intel_devicelib_math},
    {"__devicelib_acoshf", cl_intel_devicelib_math},
    {"__devicelib_asinf", cl_intel_devicelib_math},
    {"__devicelib_asinhf", cl_intel_devicelib_math},
    {"__devicelib_atan2f", cl_intel_devicelib_math},
    {"__devicelib_atanf", cl_intel_devicelib_math},
    {"__devicelib_atanhf", cl_intel_devicelib_math},
    {"__devicelib_cbrtf", cl_intel_devicelib_math},
    {"__devicelib_cosf", cl_intel_devicelib_math},
    {"__devicelib_coshf", cl_intel_devicelib_math},
    {"__devicelib_erfcf", cl_intel_devicelib_math},
    {"__devicelib_erff", cl_intel_devicelib_math},
    {"__devicelib_exp2f", cl_intel_devicelib_math},
    {"__devicelib_expf", cl_intel_devicelib_math},
    {"__devicelib_expm1f", cl_intel_devicelib_math},
    {"__devicelib_fdimf", cl_intel_devicelib_math},
    {"__devicelib_fmaf", cl_intel_devicelib_math},
    {"__devicelib_fmodf", cl_intel_devicelib_math},
    {"__devicelib_frexpf", cl_intel_devicelib_math},
    {"__devicelib_hypotf", cl_intel_devicelib_math},
    {"__devicelib_ilogbf", cl_intel_devicelib_math},
    {"__devicelib_ldexpf", cl_intel_devicelib_math},
    {"__devicelib_lgammaf", cl_intel_devicelib_math},
    {"__devicelib_log10f", cl_intel_devicelib_math},
    {"__devicelib_log1pf", cl_intel_devicelib_math},
    {"__devicelib_log2f", cl_intel_devicelib_math},
    {"__devicelib_logbf", cl_intel_devicelib_math},
    {"__devicelib_logf", cl_intel_devicelib_math},
    {"__devicelib_modff", cl_intel_devicelib_math},
    {"__devicelib_nextafterf", cl_intel_devicelib_math},
    {"__devicelib_powf", cl_intel_devicelib_math},
    {"__devicelib_remainderf", cl_intel_devicelib_math},
    {"__devicelib_remquof", cl_intel_devicelib_math},
    {"__devicelib_sinf", cl_intel_devicelib_math},
    {"__devicelib_sinhf", cl_intel_devicelib_math},
    {"__devicelib_sqrtf", cl_intel_devicelib_math},
    {"__devicelib_tanf", cl_intel_devicelib_math},
    {"__devicelib_tanhf", cl_intel_devicelib_math},
    {"__devicelib_tgammaf", cl_intel_devicelib_math},
    {"__devicelib_acos", cl_intel_devicelib_math_fp64},
    {"__devicelib_acosh", cl_intel_devicelib_math_fp64},
    {"__devicelib_asin", cl_intel_devicelib_math_fp64},
    {"__devicelib_asinh", cl_intel_devicelib_math_fp64},
    {"__devicelib_atan", cl_intel_devicelib_math_fp64},
    {"__devicelib_atan2", cl_intel_devicelib_math_fp64},
    {"__devicelib_atanh", cl_intel_devicelib_math_fp64},
    {"__devicelib_cbrt", cl_intel_devicelib_math_fp64},
    {"__devicelib_cos", cl_intel_devicelib_math_fp64},
    {"__devicelib_cosh", cl_intel_devicelib_math_fp64},
    {"__devicelib_erf", cl_intel_devicelib_math_fp64},
    {"__devicelib_erfc", cl_intel_devicelib_math_fp64},
    {"__devicelib_exp", cl_intel_devicelib_math_fp64},
    {"__devicelib_exp2", cl_intel_devicelib_math_fp64},
    {"__devicelib_expm1", cl_intel_devicelib_math_fp64},
    {"__devicelib_fdim", cl_intel_devicelib_math_fp64},
    {"__devicelib_fma", cl_intel_devicelib_math_fp64},
    {"__devicelib_fmod", cl_intel_devicelib_math_fp64},
    {"__devicelib_frexp", cl_intel_devicelib_math_fp64},
    {"__devicelib_hypot", cl_intel_devicelib_math_fp64},
    {"__devicelib_ilogb", cl_intel_devicelib_math_fp64},
    {"__devicelib_ldexp", cl_intel_devicelib_math_fp64},
    {"__devicelib_lgamma", cl_intel_devicelib_math_fp64},
    {"__devicelib_log", cl_intel_devicelib_math_fp64},
    {"__devicelib_log10", cl_intel_devicelib_math_fp64},
    {"__devicelib_log1p", cl_intel_devicelib_math_fp64},
    {"__devicelib_log2", cl_intel_devicelib_math_fp64},
    {"__devicelib_logb", cl_intel_devicelib_math_fp64},
    {"__devicelib_modf", cl_intel_devicelib_math_fp64},
    {"__devicelib_nextafter", cl_intel_devicelib_math_fp64},
    {"__devicelib_pow", cl_intel_devicelib_math_fp64},
    {"__devicelib_remainder", cl_intel_devicelib_math_fp64},
    {"__devicelib_remquo", cl_intel_devicelib_math_fp64},
    {"__devicelib_sin", cl_intel_devicelib_math_fp64},
    {"__devicelib_sinh", cl_intel_devicelib_math_fp64},
    {"__devicelib_sqrt", cl_intel_devicelib_math_fp64},
    {"__devicelib_tan", cl_intel_devicelib_math_fp64},
    {"__devicelib_tanh", cl_intel_devicelib_math_fp64},
    {"__devicelib_tgamma", cl_intel_devicelib_math_fp64},
    {"__devicelib___divsc3", cl_intel_devicelib_complex},
    {"__devicelib___mulsc3", cl_intel_devicelib_complex},
    {"__devicelib_cabsf", cl_intel_devicelib_complex},
    {"__devicelib_cacosf", cl_intel_devicelib_complex},
    {"__devicelib_cacoshf", cl_intel_devicelib_complex},
    {"__devicelib_cargf", cl_intel_devicelib_complex},
    {"__devicelib_casinf", cl_intel_devicelib_complex},
    {"__devicelib_casinhf", cl_intel_devicelib_complex},
    {"__devicelib_catanf", cl_intel_devicelib_complex},
    {"__devicelib_catanhf", cl_intel_devicelib_complex},
    {"__devicelib_ccosf", cl_intel_devicelib_complex},
    {"__devicelib_ccoshf", cl_intel_devicelib_complex},
    {"__devicelib_cexpf", cl_intel_devicelib_complex},
    {"__devicelib_cimagf", cl_intel_devicelib_complex},
    {"__devicelib_clogf", cl_intel_devicelib_complex},
    {"__devicelib_cpolarf", cl_intel_devicelib_complex},
    {"__devicelib_cpowf", cl_intel_devicelib_complex},
    {"__devicelib_cprojf", cl_intel_devicelib_complex},
    {"__devicelib_crealf", cl_intel_devicelib_complex},
    {"__devicelib_csinf", cl_intel_devicelib_complex},
    {"__devicelib_csinhf", cl_intel_devicelib_complex},
    {"__devicelib_csqrtf", cl_intel_devicelib_complex},
    {"__devicelib_ctanf", cl_intel_devicelib_complex},
    {"__devicelib_ctanhf", cl_intel_devicelib_complex},
    {"__devicelib___divdc3", cl_intel_devicelib_complex_fp64},
    {"__devicelib___muldc3", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cabs", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cacos", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cacosh", cl_intel_devicelib_complex_fp64},
    {"__devicelib_carg", cl_intel_devicelib_complex_fp64},
    {"__devicelib_casin", cl_intel_devicelib_complex_fp64},
    {"__devicelib_casinh", cl_intel_devicelib_complex_fp64},
    {"__devicelib_catan", cl_intel_devicelib_complex_fp64},
    {"__devicelib_catanh", cl_intel_devicelib_complex_fp64},
    {"__devicelib_ccos", cl_intel_devicelib_complex_fp64},
    {"__devicelib_ccosh", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cexp", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cimag", cl_intel_devicelib_complex_fp64},
    {"__devicelib_clog", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cpolar", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cpow", cl_intel_devicelib_complex_fp64},
    {"__devicelib_cproj", cl_intel_devicelib_complex_fp64},
    {"__devicelib_creal", cl_intel_devicelib_complex_fp64},
    {"__devicelib_csin", cl_intel_devicelib_complex_fp64},
    {"__devicelib_csinh", cl_intel_devicelib_complex_fp64},
    {"__devicelib_csqrt", cl_intel_devicelib_complex_fp64},
    {"__devicelib_ctan", cl_intel_devicelib_complex_fp64},
    {"__devicelib_ctanh", cl_intel_devicelib_complex_fp64},
};

static void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

static void checkError(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

static void writeToFile(std::string Filename, std::string Content) {
  std::error_code EC;
  raw_fd_ostream OS{Filename, EC, sys::fs::OpenFlags::OF_None};
  checkError(EC, "error opening the file '" + Filename + "'");
  OS.write(Content.data(), Content.size());
  OS.close();
}

// Describes scope covered by each entry in the module-kernel map populated by
// the collectKernelModuleMap function.
enum KernelMapEntryScope {
  Scope_PerKernel, // one entry per kernel
  Scope_PerModule, // one entry per module
  Scope_Global     // single entry in the map for all kernels
};

// This function decides how kernels of the input module M will be distributed
// ("split") into multiple modules based on the command options and IR
// attributes. The decision is recorded in the output map parameter
// ResKernelModuleMap which maps some key to a group of kernels. Each such group
// along with IR it depends on (globals, functions from its call graph,...) will
// constitute a separate module.
static void collectKernelModuleMap(
    Module &M, std::map<StringRef, std::vector<Function *>> &ResKernelModuleMap,
    KernelMapEntryScope EntryScope) {

  for (auto &F : M.functions()) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      switch (EntryScope) {
      case Scope_PerKernel:
        ResKernelModuleMap[F.getName()].push_back(&F);
        break;
      case Scope_PerModule: {
        constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";

        // TODO It may make sense to group all kernels w/o the attribute into
        // a separate module rather than issuing an error. Should probably be
        // controlled by an option.
        if (!F.hasFnAttribute(ATTR_SYCL_MODULE_ID))
          error("no '" + Twine(ATTR_SYCL_MODULE_ID) +
                "' attribute in kernel '" + F.getName() +
                "', per-module split not possible");
        Attribute Id = F.getFnAttribute(ATTR_SYCL_MODULE_ID);
        StringRef Val = Id.getValueAsString();
        ResKernelModuleMap[Val].push_back(&F);
        break;
      }
      case Scope_Global:
        // the map key is not significant here
        ResKernelModuleMap["<GLOBAL>"].push_back(&F);
        break;
      default:
        llvm_unreachable("unknown scope");
      }
    }
  }
}

// Input parameter KernelModuleMap is a map containing groups of kernels with
// same values of the sycl-module-id attribute. ResSymbolsLists is a vector of
// kernel name lists. Each vector element is a string with kernel names from the
// same module separated by \n.
// The function saves names of kernels from one group to a single std::string
// and stores this string to the ResSymbolsLists vector.
static void collectSymbolsLists(
    std::map<StringRef, std::vector<Function *>> &KernelModuleMap,
    string_vector &ResSymbolsLists) {
  for (auto &It : KernelModuleMap) {
    std::string SymbolsList;
    for (auto &F : It.second) {
      SymbolsList =
          (Twine(SymbolsList) + Twine(F->getName()) + Twine("\n")).str();
    }
    ResSymbolsLists.push_back(std::move(SymbolsList));
  }
}

// Input parameter KernelModuleMap is a map containing groups of kernels with
// same values of the sycl-module-id attribute. For each group of kernels a
// separate IR module will be produced.
// ResModules is a vector of produced modules.
// The function splits input LLVM IR module M into smaller ones and stores them
// to the ResModules vector.
static void
splitModule(Module &M,
            std::map<StringRef, std::vector<Function *>> &KernelModuleMap,
            std::vector<std::unique_ptr<Module>> &ResModules) {
  for (auto &It : KernelModuleMap) {
    // For each group of kernels collect all dependencies.
    SetVector<const GlobalValue *> GVs;
    std::vector<llvm::Function *> Workqueue;

    for (auto &F : It.second) {
      GVs.insert(F);
      Workqueue.push_back(F);
    }

    while (!Workqueue.empty()) {
      Function *F = &*Workqueue.back();
      Workqueue.pop_back();
      for (auto &I : instructions(F)) {
        if (CallBase *CB = dyn_cast<CallBase>(&I))
          if (Function *CF = CB->getCalledFunction())
            if (!CF->isDeclaration() && !GVs.count(CF)) {
              GVs.insert(CF);
              Workqueue.push_back(CF);
            }
      }
    }

    // It's not easy to trace global variable's uses inside needed functions
    // because global variable can be used inside a combination of operators, so
    // mark all global variables as needed and remove dead ones after
    // cloning.
    for (auto &G : M.globals()) {
      GVs.insert(&G);
    }

    ValueToValueMapTy VMap;
    // Clone definitions only for needed globals. Others will be added as
    // declarations and removed later.
    std::unique_ptr<Module> MClone = CloneModule(
        M, VMap, [&](const GlobalValue *GV) { return GVs.count(GV); });

    // TODO: Use the new PassManager instead?
    legacy::PassManager Passes;
    // Do cleanup.
    Passes.add(createGlobalDCEPass());           // Delete unreachable globals.
    Passes.add(createStripDeadDebugInfoPass());  // Remove dead debug info.
    Passes.add(createStripDeadPrototypesPass()); // Remove dead func decls.
    Passes.run(*MClone.get());

    // Save results.
    ResModules.push_back(std::move(MClone));
  }
}

static std::string makeResultFileName(Twine Ext, int I) {
  const StringRef Dir0 = OutputDir.getNumOccurrences() > 0
                             ? OutputDir
                             : sys::path::parent_path(OutputFilename);
  const StringRef Sep = sys::path::get_separator();
  std::string Dir = Dir0.str();
  if (!Dir0.empty() && !Dir0.endswith(Sep))
    Dir += Sep.str();
  return (Dir + Twine(sys::path::stem(OutputFilename)) + "_" +
          std::to_string(I) + Ext)
      .str();
}

static void saveModule(Module &M, StringRef OutFilename) {
  std::error_code EC;
  raw_fd_ostream Out{OutFilename, EC, sys::fs::OF_None};
  checkError(EC, "error opening the file '" + OutFilename + "'");

  // TODO: Use the new PassManager instead?
  legacy::PassManager PrintModule;

  if (OutputAssembly)
    PrintModule.add(createPrintModulePass(Out, ""));
  else if (Force || !CheckBitcodeOutputToConsole(Out, true))
    PrintModule.add(createBitcodeWriterPass(Out));
  PrintModule.run(M);
}

// Saves specified collection of llvm IR modules to files.
// Saves file list if user specified corresponding filename.
static string_vector
saveResultModules(std::vector<std::unique_ptr<Module>> &ResModules) {
  string_vector Res;

  for (size_t I = 0; I < ResModules.size(); ++I) {
    std::error_code EC;
    StringRef FileExt = (OutputAssembly) ? ".ll" : ".bc";
    std::string CurOutFileName = makeResultFileName(FileExt, I);
    saveModule(*ResModules[I].get(), CurOutFileName);
    Res.emplace_back(std::move(CurOutFileName));
  }
  return Res;
}

// Each fallback device library corresponds to one bit in "require mask" which
// is an unsigned int32. getDeviceLibBit checks which fallback device library
// is required for FuncName and returns the corresponding bit. The corresponding
// mask for each fallback device library is:
// fallback-cassert:      0x1
// fallback-cmath:        0x2
// fallback-cmath-fp64:   0x4
// fallback-complex:      0x8
// fallback-complex-fp64: 0x10
static uint32_t getDeviceLibBits(const std::string &FuncName) {

  if (DeviceLibFuncMap.count(FuncName) == 0)
    return 0;
  else
    return 0x1 << (DeviceLibFuncMap[FuncName] - cl_intel_devicelib_assert);
}

// For each device image module, we go through all functions which meets
// 1. The function name has prefix "__devicelib_"
// 2. The function has SPIR_FUNC calling convention
// 3. The function is declaration which means it doesn't have function body
static uint32_t getModuleReqMask(const Module &M) {
  // 0x1 means sycl runtime will link and load libsycl-fallback-assert.spv as
  // default. In fact, default link assert spv is not necessary but dramatic
  // perf regression is observed if we don't link any device library. The perf
  // regression is caused by a clang issue.
  uint32_t ReqMask = 0x1;
  uint32_t DeviceLibBits = 0;
  for (const Function &SF : M) {
    if (SF.getName().startswith(DEVICELIB_FUNC_PREFIX) &&
        (SF.getCallingConv() == CallingConv::SPIR_FUNC) && SF.isDeclaration()) {
      DeviceLibBits = getDeviceLibBits(SF.getName().str());
      ReqMask |= DeviceLibBits;
    }
  }
  return ReqMask;
}

static string_vector saveDeviceImageProperty(
    const std::vector<std::unique_ptr<Module>> &ResultModules,
    const ImagePropSaveInfo &ImgPSInfo) {
  string_vector Res;
  for (size_t I = 0; I < ResultModules.size(); ++I) {
    std::string SCFile = makeResultFileName(".prop", I);
    llvm::util::PropertySetRegistry PropSet;
    if (ImgPSInfo.NeedDeviceLibReqMask) {
      uint32_t MRMask = getModuleReqMask(*ResultModules[I]);
      std::map<StringRef, uint32_t> RMEntry = {{"DeviceLibReqMask", MRMask}};
      PropSet.add(llvm::util::PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK,
                  RMEntry);
    }
    if (ImgPSInfo.DoSpecConst && ImgPSInfo.SetSpecConstAtRT) {
      // extract spec constant maps per each module
      SpecIDMapTy TmpSpecIDMap;
      if (ImgPSInfo.SpecConstsMet)
        SpecConstantsPass::collectSpecConstantMetadata(*ResultModules[I].get(),
                                                       TmpSpecIDMap);
      PropSet.add(
          llvm::util::PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS,
          TmpSpecIDMap);
    }
    std::error_code EC;
    raw_fd_ostream SCOut(SCFile, EC);
    PropSet.write(SCOut);
    Res.emplace_back(std::move(SCFile));
  }

  return Res;
}

// Saves specified collection of symbols lists to files.
// Saves file list if user specified corresponding filename.
static string_vector saveResultSymbolsLists(string_vector &ResSymbolsLists) {
  string_vector Res;

  std::string TxtFilesList;
  for (size_t I = 0; I < ResSymbolsLists.size(); ++I) {
    std::string CurOutFileName = makeResultFileName(".sym", I);
    writeToFile(CurOutFileName, ResSymbolsLists[I]);
    Res.emplace_back(std::move(CurOutFileName));
  }
  return std::move(Res);
}

#define CHECK_AND_EXIT(E)                                                      \
  {                                                                            \
    Error LocE = std::move(E);                                                 \
    if (LocE) {                                                                \
      logAllUnhandledErrors(std::move(LocE), WithColor::error(errs()));        \
      return 1;                                                                \
    }                                                                          \
  }

int main(int argc, char **argv) {
  InitLLVM X{argc, argv};

  LLVMContext Context;
  cl::HideUnrelatedOptions(PostLinkCat);
  cl::ParseCommandLineOptions(
      argc, argv,
      "SYCL post-link device code processing tool.\n"
      "This is a collection of utilities run on device code's LLVM IR before\n"
      "handing off to back-end for further compilation or emitting SPIRV.\n"
      "The utilities are:\n"
      "- Module splitter to split a big input module into smaller ones.\n"
      "  Groups kernels using function attribute 'sycl-module-id', i.e.\n"
      "  kernels with the same values of the 'sycl-module-id' attribute will\n"
      "  be put into the same module. If -split=kernel option is specified,\n"
      "  one module per kernel will be emitted.\n"
      "- If -symbols options is also specified, then for each produced module\n"
      "  a text file containing names of all spir kernels in it is generated.\n"
      "- Specialization constant intrinsic transformer. Replaces symbolic\n"
      "  ID-based intrinsics to integer ID-based ones to make them friendly\n"
      "  for the SPIRV translator\n"
      "Normally, the tool generates a number of files and \"file table\"\n"
      "file listing all generated files in a table manner. For example, if\n"
      "the input file 'example.bc' contains two kernels, then the command\n"
      "  $ sycl-post-link --split=kernel --symbols --spec-const=rt \\\n"
      "    -o example.table example.bc\n"
      "will produce 'example.table' file with the following content:\n"
      "  [Code|Properties|Symbols]\n"
      "  example_0.bc|example_0.prop|example_0.sym\n"
      "  example_1.bc|example_1.prop|example_1.sym\n"
      "When only specialization constant processing is needed, the tool can\n"
      "output a single transformed IR file if --ir-output-only is specified:\n"
      "  $ sycl-post-link --ir-output-only --spec-const=default \\\n"
      "    -o example_p.bc example.bc\n"
      "will produce single output file example_p.bc suitable for SPIRV\n"
      "translation.\n");

  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }
  if (IROutputOnly && DoSplit) {
    errs() << "error: -" << SplitMode.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoSymGen) {
    errs() << "error: -" << DoSymGen.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  // It is OK to use raw pointer here as we control that it does not outlive M
  // or objects it is moved to
  Module *MPtr = M.get();

  if (!MPtr) {
    Err.print(argv[0], errs());
    return 1;
  }
  if (OutputFilename.getNumOccurrences() == 0)
    OutputFilename = (Twine(sys::path::stem(InputFilename)) + ".files").str();

  std::map<StringRef, std::vector<Function *>> GlobalsSet;

  if (DoSplit || DoSymGen) {
    KernelMapEntryScope Scope = Scope_Global;
    if (DoSplit)
      Scope = SplitMode == SPLIT_PER_KERNEL ? Scope_PerKernel : Scope_PerModule;
    collectKernelModuleMap(*MPtr, GlobalsSet, Scope);
  }

  std::vector<std::unique_ptr<Module>> ResultModules;
  string_vector ResultSymbolsLists;

  util::SimpleTable Table;
  bool SpecConstsMet = false;
  bool SetSpecConstAtRT = DoSpecConst && (SpecConstLower == SC_USE_RT_VAL);

  if (DoSpecConst) {
    // perform the spec constant intrinsics transformation and enumeration on
    // the whole module
    ModulePassManager RunSpecConst;
    ModuleAnalysisManager MAM;
    SpecConstantsPass SCP(SetSpecConstAtRT);
    // Register required analysis
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    RunSpecConst.addPass(SCP);
    if (!DoSplit)
      // This pass deletes unreachable globals. Code splitter runs it later.
      RunSpecConst.addPass(GlobalDCEPass());
    PreservedAnalyses Res = RunSpecConst.run(*MPtr, MAM);
    SpecConstsMet = !Res.areAllPreserved();
  }
  if (IROutputOnly) {
    // the result is the transformed input LLVMIR file rather than a file table
    saveModule(*MPtr, OutputFilename);
    return 0;
  }
  if (DoSplit) {
    splitModule(*MPtr, GlobalsSet, ResultModules);
    // post-link always produces a code result, even if it is unmodified input
    if (ResultModules.size() == 0)
      ResultModules.push_back(std::move(M));
  } else
    ResultModules.push_back(std::move(M));

  {
    // reuse input module if there were no spec constants and no splitting
    string_vector Files = SpecConstsMet || (ResultModules.size() > 1)
                              ? saveResultModules(ResultModules)
                              : string_vector{InputFilename};
    // "Code" column is always output
    Error Err = Table.addColumn(COL_CODE, Files);
    CHECK_AND_EXIT(Err);
  }

  {
    ImagePropSaveInfo ImgPSInfo = {true, DoSpecConst, SetSpecConstAtRT,
                                   SpecConstsMet};
    string_vector Files = saveDeviceImageProperty(ResultModules, ImgPSInfo);
    Error Err = Table.addColumn(COL_PROPS, Files);
    CHECK_AND_EXIT(Err);
  }

  if (DoSymGen) {
    // extract symbols per each module
    collectSymbolsLists(GlobalsSet, ResultSymbolsLists);
    if (ResultSymbolsLists.empty()) {
      // push empty symbols list for consistency
      assert(ResultModules.size() == 1);
      ResultSymbolsLists.push_back("");
    }
    string_vector Files = saveResultSymbolsLists(ResultSymbolsLists);
    Error Err = Table.addColumn(COL_SYM, Files);
    CHECK_AND_EXIT(Err);
  }
  {
    std::error_code EC;
    raw_fd_ostream Out{OutputFilename, EC, sys::fs::OF_None};
    checkError(EC, "error opening file '" + OutputFilename + "'");
    Table.write(Out);
  }
  return 0;
}
