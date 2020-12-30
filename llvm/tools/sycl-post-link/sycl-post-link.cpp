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

#include "SPIRKernelParamOptInfo.h"
#include "SpecConstants.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Triple.h"
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
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>

using namespace llvm;

using string_vector = std::vector<std::string>;

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// Column names in the output file table. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
static constexpr char COL_CODE[] = "Code";
static constexpr char COL_SYM[] = "Symbols";
static constexpr char COL_PROPS[] = "Properties";
static constexpr char DEVICELIB_FUNC_PREFIX[] = "__devicelib_";

// DeviceLibExt is shared between sycl-post-link tool and sycl runtime.
// If any change is made here, need to sync with DeviceLibExt definition
// in sycl/source/detail/program_manager/program_manager.hpp
enum class DeviceLibExt : std::uint32_t {
  cl_intel_devicelib_assert,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64
};

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
  SPLIT_PER_TU,     // one module per translation unit
  SPLIT_PER_KERNEL, // one module per kernel
  SPLIT_AUTO        // automatically select split mode
};

static cl::opt<IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional, cl::init(SPLIT_AUTO),
    cl::values(
        clEnumValN(SPLIT_PER_TU, "source",
                   "1 output module per source (translation unit)"),
        clEnumValN(SPLIT_PER_KERNEL, "kernel", "1 output module per kernel"),
        clEnumValN(SPLIT_AUTO, "auto", "Choose split mode automatically")),
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

static cl::opt<bool> EmitKernelParamInfo{
    "emit-param-info", cl::desc("emit kernel parameter optimization info"),
    cl::cat(PostLinkCat)};

struct ImagePropSaveInfo {
  bool NeedDeviceLibReqMask;
  bool DoSpecConst;
  bool SetSpecConstAtRT;
  bool SpecConstsMet;
  bool EmitKernelParamInfo;
};
// Please update DeviceLibFuncMap if any item is added to or removed from
// fallback device libraries in libdevice.
static std::unordered_map<std::string, DeviceLibExt> DeviceLibFuncMap = {
    {"__devicelib_acosf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_acoshf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_asinf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_asinhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_atan2f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_atanf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_atanhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_cbrtf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_cosf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_coshf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_erfcf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_erff", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_exp2f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_expf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_expm1f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_fdimf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_fmaf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_fmodf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_frexpf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_hypotf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_ilogbf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_ldexpf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_lgammaf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_log10f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_log1pf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_log2f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_logbf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_logf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_modff", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_nextafterf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_powf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_remainderf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_remquof", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_scalbnf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_sinf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_sinhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_sqrtf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_tanf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_tanhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_tgammaf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_acos", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_acosh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_asin", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_asinh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_atan", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_atan2", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_atanh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_cbrt", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_cos", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_cosh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_erf", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_erfc", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_exp", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_exp2", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_expm1", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_fdim", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_fma", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_fmod", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_frexp", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_hypot", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_ilogb", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_ldexp", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_lgamma", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log10", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log1p", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log2", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_logb", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_modf", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_nextafter", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_pow", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_remainder", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_remquo", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_scalbn", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_sin", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_sinh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_sqrt", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_tan", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_tanh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_tgamma", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib___divsc3", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib___mulsc3", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cabsf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cacosf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cacoshf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cargf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_casinf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_casinhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_catanf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_catanhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ccosf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ccoshf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cexpf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cimagf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_clogf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cpolarf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cpowf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cprojf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_crealf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_csinf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_csinhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_csqrtf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ctanf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ctanhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib___divdc3", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib___muldc3", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cabs", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cacos", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cacosh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_carg", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_casin", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_casinh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_catan", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_catanh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ccos", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ccosh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cexp", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cimag", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_clog", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cpolar", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cpow", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cproj", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_creal", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_csin", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_csinh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_csqrt", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ctan", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ctanh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
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

static KernelMapEntryScope selectDeviceCodeSplitScopeAutomatically(Module &M) {
  if (IROutputOnly) {
    // We allow enabling auto split mode even in presence of -ir-output-only
    // flag, but in this case we are limited by it so we can't do any split at
    // all.
    return Scope_Global;
  }

  for (const auto &F : M.functions()) {
    // There are functions marked with [[intel::device_indirectly_callable]]
    // attribute, because it instructs us to make this function available to the
    // whole program as it was compiled as a single module.
    if (F.hasFnAttribute("referenced-indirectly"))
      return Scope_Global;
    if (F.isDeclaration())
      continue;
    // There are indirect calls in the module, which means that we don't know
    // how to group functions so both caller and callee of indirect call are in
    // the same module.
    for (const auto &BB : F) {
      for (const auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (!CI->getCalledFunction())
            return Scope_Global;
        }
      }
    }
  }

  // At the moment, we assume that per-source split is the best way of splitting
  // device code and can always be used execpt for cases handled above.
  return Scope_PerModule;
}

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
  else if (Force || !CheckBitcodeOutputToConsole(Out))
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
  auto DeviceLibFuncIter = DeviceLibFuncMap.find(FuncName);
  return ((DeviceLibFuncIter == DeviceLibFuncMap.end())
              ? 0
              : 0x1 << (static_cast<uint32_t>(DeviceLibFuncIter->second) -
                        static_cast<uint32_t>(
                            DeviceLibExt::cl_intel_devicelib_assert)));
}

// For each device image module, we go through all functions which meets
// 1. The function name has prefix "__devicelib_"
// 2. The function is declaration which means it doesn't have function body
// And we don't expect non-spirv functions with "__devicelib_" prefix.
static uint32_t getModuleReqMask(const Module &M) {
  // Device libraries will be enabled only for spir-v module.
  if (!llvm::Triple(M.getTargetTriple()).isSPIR())
    return 0;
  // 0x1 means sycl runtime will link and load libsycl-fallback-assert.spv as
  // default. In fact, default link assert spv is not necessary but dramatic
  // perf regression is observed if we don't link any device library. The perf
  // regression is caused by a clang issue.
  uint32_t ReqMask = 0x1;
  for (const Function &SF : M) {
    if (SF.getName().startswith(DEVICELIB_FUNC_PREFIX) && SF.isDeclaration()) {
      assert(SF.getCallingConv() == CallingConv::SPIR_FUNC);
      uint32_t DeviceLibBits = getDeviceLibBits(SF.getName().str());
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
    llvm::util::PropertySetRegistry PropSet;
    if (ImgPSInfo.NeedDeviceLibReqMask) {
      uint32_t MRMask = getModuleReqMask(*ResultModules[I]);
      std::map<StringRef, uint32_t> RMEntry = {{"DeviceLibReqMask", MRMask}};
      PropSet.add(llvm::util::PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK,
                  RMEntry);
    }
    if (ImgPSInfo.DoSpecConst && ImgPSInfo.SetSpecConstAtRT) {
      if (ImgPSInfo.SpecConstsMet) {
        // extract spec constant maps per each module
        ScalarSpecIDMapTy TmpScalarSpecIDMap;
        CompositeSpecIDMapTy TmpCompositeSpecIDMap;
        SpecConstantsPass::collectSpecConstantMetadata(
            *ResultModules[I].get(), TmpScalarSpecIDMap, TmpCompositeSpecIDMap);
        PropSet.add(
            llvm::util::PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS,
            TmpScalarSpecIDMap);
        PropSet.add(llvm::util::PropertySetRegistry::
                        SYCL_COMPOSITE_SPECIALIZATION_CONSTANTS,
                    TmpCompositeSpecIDMap);
      }
    }
    if (ImgPSInfo.EmitKernelParamInfo) {
      // extract kernel parameter optimization info per module
      ModuleAnalysisManager MAM;
      // Register required analysis
      MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
      // Register the payload analysis
      MAM.registerPass([&] { return SPIRKernelParamOptInfoAnalysis(); });
      SPIRKernelParamOptInfo PInfo =
          MAM.getResult<SPIRKernelParamOptInfoAnalysis>(*ResultModules[I]);

      // convert analysis results into properties and record them
      llvm::util::PropertySet &Props =
          PropSet[llvm::util::PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO];

      for (const auto &NameInfoPair : PInfo) {
        const llvm::BitVector &Bits = NameInfoPair.second;
        const llvm::ArrayRef<uintptr_t> Arr = NameInfoPair.second.getData();
        const unsigned char *Data =
            reinterpret_cast<const unsigned char *>(Arr.begin());
        llvm::util::PropertyValue::SizeTy DataBitSize = Bits.size();
        Props.insert(std::make_pair(
            NameInfoPair.first, llvm::util::PropertyValue(Data, DataBitSize)));
      }
    }
    std::error_code EC;
    std::string SCFile = makeResultFileName(".prop", I);
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
      "  '-split=auto' mode automatically selects the best way of splitting\n"
      "  kernels into modules based on some heuristic.\n"
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
      "translation.\n"
      "--ir-output-only option is not not compatible with split modes other\n"
      "than 'auto'.\n");

  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;
  bool DoParamInfo = EmitKernelParamInfo.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen && !DoParamInfo) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }
  if (IROutputOnly && (DoSplit && SplitMode != SPLIT_AUTO)) {
    errs() << "error: -" << SplitMode.ArgStr << "=" << SplitMode.ValueStr
           << " can't be used with -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoSymGen) {
    errs() << "error: -" << DoSymGen.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoParamInfo) {
    errs() << "error: -" << EmitKernelParamInfo.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
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
    if (DoSplit) {
      if (SplitMode == SPLIT_AUTO)
        Scope = selectDeviceCodeSplitScopeAutomatically(*MPtr);
      else
        Scope =
            SplitMode == SPLIT_PER_KERNEL ? Scope_PerKernel : Scope_PerModule;
    }
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
                                   SpecConstsMet, EmitKernelParamInfo};
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
