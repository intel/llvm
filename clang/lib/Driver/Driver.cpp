//===--- Driver.cpp - Clang GCC Compatible Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Driver/Driver.h"
#include "ToolChains/AIX.h"
#include "ToolChains/AMDGPU.h"
#include "ToolChains/AMDGPUOpenMP.h"
#include "ToolChains/AVR.h"
#include "ToolChains/Arch/RISCV.h"
#include "ToolChains/BareMetal.h"
#include "ToolChains/CSKYToolChain.h"
#include "ToolChains/Clang.h"
#include "ToolChains/CrossWindows.h"
#include "ToolChains/Cuda.h"
#include "ToolChains/Darwin.h"
#include "ToolChains/DragonFly.h"
#include "ToolChains/FreeBSD.h"
#include "ToolChains/Fuchsia.h"
#include "ToolChains/Gnu.h"
#include "ToolChains/HIPAMD.h"
#include "ToolChains/HIPSPV.h"
#include "ToolChains/HLSL.h"
#include "ToolChains/Haiku.h"
#include "ToolChains/Hexagon.h"
#include "ToolChains/Hurd.h"
#include "ToolChains/Lanai.h"
#include "ToolChains/Linux.h"
#include "ToolChains/MSP430.h"
#include "ToolChains/MSVC.h"
#include "ToolChains/MinGW.h"
#include "ToolChains/MipsLinux.h"
#include "ToolChains/NaCl.h"
#include "ToolChains/NetBSD.h"
#include "ToolChains/OHOS.h"
#include "ToolChains/OpenBSD.h"
#include "ToolChains/PPCFreeBSD.h"
#include "ToolChains/PPCLinux.h"
#include "ToolChains/PS4CPU.h"
#include "ToolChains/RISCVToolchain.h"
#include "ToolChains/SPIRV.h"
#include "ToolChains/SYCL.h"
#include "ToolChains/Solaris.h"
#include "ToolChains/TCE.h"
#include "ToolChains/VEToolchain.h"
#include "ToolChains/WebAssembly.h"
#include "ToolChains/XCore.h"
#include "ToolChains/ZOS.h"
#include "clang/Basic/TargetID.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Phases.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/SYCLLowerIR/DeviceConfigFile.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ExitCodes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <cstdlib> // ::getenv
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <set>
#include <utility>
#if LLVM_ON_UNIX
#include <unistd.h> // getpid
#endif

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

static std::optional<llvm::Triple> getOffloadTargetTriple(const Driver &D,
                                                          const ArgList &Args) {
  auto OffloadTargets = Args.getAllArgValues(options::OPT_offload_EQ);
  // Offload compilation flow does not support multiple targets for now. We
  // need the HIPActionBuilder (and possibly the CudaActionBuilder{,Base}too)
  // to support multiple tool chains first.
  switch (OffloadTargets.size()) {
  default:
    D.Diag(diag::err_drv_only_one_offload_target_supported);
    return std::nullopt;
  case 0:
    D.Diag(diag::err_drv_invalid_or_unsupported_offload_target) << "";
    return std::nullopt;
  case 1:
    break;
  }
  return llvm::Triple(OffloadTargets[0]);
}

static std::optional<llvm::Triple>
getNVIDIAOffloadTargetTriple(const Driver &D, const ArgList &Args,
                             const llvm::Triple &HostTriple) {
  if (!Args.hasArg(options::OPT_offload_EQ)) {
    return llvm::Triple(HostTriple.isArch64Bit() ? "nvptx64-nvidia-cuda"
                                                 : "nvptx-nvidia-cuda");
  }
  auto TT = getOffloadTargetTriple(D, Args);
  if (TT && (TT->getArch() == llvm::Triple::spirv32 ||
             TT->getArch() == llvm::Triple::spirv64)) {
    if (Args.hasArg(options::OPT_emit_llvm))
      return TT;
    D.Diag(diag::err_drv_cuda_offload_only_emit_bc);
    return std::nullopt;
  }
  D.Diag(diag::err_drv_invalid_or_unsupported_offload_target) << TT->str();
  return std::nullopt;
}
static std::optional<llvm::Triple>
getHIPOffloadTargetTriple(const Driver &D, const ArgList &Args) {
  if (!Args.hasArg(options::OPT_offload_EQ)) {
    return llvm::Triple("amdgcn-amd-amdhsa"); // Default HIP triple.
  }
  auto TT = getOffloadTargetTriple(D, Args);
  if (!TT)
    return std::nullopt;
  if (TT->getArch() == llvm::Triple::amdgcn &&
      TT->getVendor() == llvm::Triple::AMD &&
      TT->getOS() == llvm::Triple::AMDHSA)
    return TT;
  if (TT->getArch() == llvm::Triple::spirv64)
    return TT;
  D.Diag(diag::err_drv_invalid_or_unsupported_offload_target) << TT->str();
  return std::nullopt;
}

// static
std::string Driver::GetResourcesPath(StringRef BinaryPath,
                                     StringRef CustomResourceDir) {
  // Since the resource directory is embedded in the module hash, it's important
  // that all places that need it call this function, so that they get the
  // exact same string ("a/../b/" and "b/" get different hashes, for example).

  // Dir is bin/ or lib/, depending on where BinaryPath is.
  std::string Dir = std::string(llvm::sys::path::parent_path(BinaryPath));

  SmallString<128> P(Dir);
  if (CustomResourceDir != "") {
    llvm::sys::path::append(P, CustomResourceDir);
  } else {
    // On Windows, libclang.dll is in bin/.
    // On non-Windows, libclang.so/.dylib is in lib/.
    // With a static-library build of libclang, LibClangPath will contain the
    // path of the embedding binary, which for LLVM binaries will be in bin/.
    // ../lib gets us to lib/ in both cases.
    P = llvm::sys::path::parent_path(Dir);
    // This search path is also created in the COFF driver of lld, so any
    // changes here also needs to happen in lld/COFF/Driver.cpp
    llvm::sys::path::append(P, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                            CLANG_VERSION_MAJOR_STRING);
  }

  return std::string(P);
}

Driver::Driver(StringRef ClangExecutable, StringRef TargetTriple,
               DiagnosticsEngine &Diags, std::string Title,
               IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS)
    : Diags(Diags), VFS(std::move(VFS)), DumpDeviceCode(false), Mode(GCCMode),
      SaveTemps(SaveTempsNone), BitcodeEmbed(EmbedNone),
      Offload(OffloadHostDevice), CXX20HeaderType(HeaderMode_None),
      ModulesModeCXX20(false), LTOMode(LTOK_None), OffloadLTOMode(LTOK_None),
      ClangExecutable(ClangExecutable), SysRoot(DEFAULT_SYSROOT),
      DriverTitle(Title), CCCPrintBindings(false), CCPrintOptions(false),
      CCLogDiagnostics(false), CCGenDiagnostics(false),
      CCPrintProcessStats(false), CCPrintInternalStats(false),
      TargetTriple(TargetTriple), Saver(Alloc), PrependArg(nullptr),
      CheckInputsExist(true), ProbePrecompiled(true),
      SuppressMissingInputWarning(false) {
  // Provide a sane fallback if no VFS is specified.
  if (!this->VFS)
    this->VFS = llvm::vfs::getRealFileSystem();

  Name = std::string(llvm::sys::path::filename(ClangExecutable));
  Dir = std::string(llvm::sys::path::parent_path(ClangExecutable));
  InstalledDir = Dir; // Provide a sensible default installed dir.

  if ((!SysRoot.empty()) && llvm::sys::path::is_relative(SysRoot)) {
    // Prepend InstalledDir if SysRoot is relative
    SmallString<128> P(InstalledDir);
    llvm::sys::path::append(P, SysRoot);
    SysRoot = std::string(P);
  }

#if defined(CLANG_CONFIG_FILE_SYSTEM_DIR)
  SystemConfigDir = CLANG_CONFIG_FILE_SYSTEM_DIR;
#endif
#if defined(CLANG_CONFIG_FILE_USER_DIR)
  {
    SmallString<128> P;
    llvm::sys::fs::expand_tilde(CLANG_CONFIG_FILE_USER_DIR, P);
    UserConfigDir = static_cast<std::string>(P);
  }
#endif

  // Compute the path to the resource directory.
  ResourceDir = GetResourcesPath(ClangExecutable, CLANG_RESOURCE_DIR);
}

void Driver::setDriverMode(StringRef Value) {
  static StringRef OptName =
      getOpts().getOption(options::OPT_driver_mode).getPrefixedName();
  if (auto M = llvm::StringSwitch<std::optional<DriverMode>>(Value)
                   .Case("gcc", GCCMode)
                   .Case("g++", GXXMode)
                   .Case("cpp", CPPMode)
                   .Case("cl", CLMode)
                   .Case("flang", FlangMode)
                   .Case("dxc", DXCMode)
                   .Default(std::nullopt))
    Mode = *M;
  else
    Diag(diag::err_drv_unsupported_option_argument) << OptName << Value;
}

InputArgList Driver::ParseArgStrings(ArrayRef<const char *> ArgStrings,
                                     bool UseDriverMode, bool &ContainsError) {
  llvm::PrettyStackTraceString CrashInfo("Command line argument parsing");
  ContainsError = false;

  llvm::opt::Visibility VisibilityMask = getOptionVisibilityMask(UseDriverMode);
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args = getOpts().ParseArgs(ArgStrings, MissingArgIndex,
                                          MissingArgCount, VisibilityMask);

  // Check for missing argument error.
  if (MissingArgCount) {
    Diag(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    ContainsError |=
        Diags.getDiagnosticLevel(diag::err_drv_missing_argument,
                                 SourceLocation()) > DiagnosticsEngine::Warning;
  }

  // Check for unsupported options.
  for (const Arg *A : Args) {
    if (A->getOption().hasFlag(options::Unsupported)) {
      Diag(diag::err_drv_unsupported_opt) << A->getAsString(Args);
      ContainsError |= Diags.getDiagnosticLevel(diag::err_drv_unsupported_opt,
                                                SourceLocation()) >
                       DiagnosticsEngine::Warning;
      continue;
    }

    // Deprecated options emit a diagnostic about deprecation, but are still
    // supported until removed. It's possible to have a deprecated option which
    // aliases with a non-deprecated option, so always compute the argument
    // actually used before checking for deprecation.
    const Arg *Used = A;
    while (Used->getAlias())
      Used = Used->getAlias();
    if (Used->getOption().hasFlag(options::Deprecated)) {
      Diag(diag::warn_drv_deprecated_option_release) << Used->getAsString(Args);
      ContainsError |= Diags.getDiagnosticLevel(
                           diag::warn_drv_deprecated_option_release,
                           SourceLocation()) > DiagnosticsEngine::Warning;
    }

    // Warn about -mcpu= without an argument.
    if (A->getOption().matches(options::OPT_mcpu_EQ) && A->containsValue("")) {
      Diag(diag::warn_drv_empty_joined_argument) << A->getAsString(Args);
      ContainsError |= Diags.getDiagnosticLevel(
                           diag::warn_drv_empty_joined_argument,
                           SourceLocation()) > DiagnosticsEngine::Warning;
    }
  }

  for (const Arg *A : Args.filtered(options::OPT_UNKNOWN)) {
    unsigned DiagID;
    auto ArgString = A->getAsString(Args);
    std::string Nearest;
    if (getOpts().findNearest(ArgString, Nearest, VisibilityMask) > 1) {
      if (!IsCLMode() &&
          getOpts().findExact(ArgString, Nearest,
                              llvm::opt::Visibility(options::CC1Option))) {
        DiagID = diag::err_drv_unknown_argument_with_suggestion;
        Diags.Report(DiagID) << ArgString << "-Xclang " + Nearest;
      } else {
        DiagID = IsCLMode() ? diag::warn_drv_unknown_argument_clang_cl
                            : diag::err_drv_unknown_argument;
        Diags.Report(DiagID) << ArgString;
      }
    } else {
      DiagID = IsCLMode()
                   ? diag::warn_drv_unknown_argument_clang_cl_with_suggestion
                   : diag::err_drv_unknown_argument_with_suggestion;
      Diags.Report(DiagID) << ArgString << Nearest;
    }
    ContainsError |= Diags.getDiagnosticLevel(DiagID, SourceLocation()) >
                     DiagnosticsEngine::Warning;
  }

  for (const Arg *A : Args.filtered(options::OPT_o)) {
    if (ArgStrings[A->getIndex()] == A->getSpelling())
      continue;

    // Warn on joined arguments that are similar to a long argument.
    std::string ArgString = ArgStrings[A->getIndex()];
    std::string Nearest;
    if (getOpts().findExact("-" + ArgString, Nearest, VisibilityMask))
      Diags.Report(diag::warn_drv_potentially_misspelled_joined_argument)
          << A->getAsString(Args) << Nearest;
  }

  return Args;
}

// Determine which compilation mode we are in. We look for options which
// affect the phase, starting with the earliest phases, and record which
// option we used to determine the final phase.
phases::ID Driver::getFinalPhase(const DerivedArgList &DAL,
                                 Arg **FinalPhaseArg) const {
  Arg *PhaseArg = nullptr;
  phases::ID FinalPhase;

  // -{E,EP,P,M,MM} only run the preprocessor.
  if (CCCIsCPP() || (PhaseArg = DAL.getLastArg(options::OPT_E)) ||
      (PhaseArg = DAL.getLastArg(options::OPT__SLASH_EP)) ||
      (PhaseArg = DAL.getLastArg(options::OPT_M, options::OPT_MM)) ||
      (PhaseArg = DAL.getLastArg(options::OPT__SLASH_P)) ||
      CCGenDiagnostics) {
    FinalPhase = phases::Preprocess;

    // --precompile only runs up to precompilation.
    // Options that cause the output of C++20 compiled module interfaces or
    // header units have the same effect.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT__precompile)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_extract_api)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_fmodule_header,
                                        options::OPT_fmodule_header_EQ))) {
    FinalPhase = phases::Precompile;
    // -{fsyntax-only,-analyze,emit-ast} only run up to the compiler.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_fsyntax_only)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_print_supported_cpus)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_module_file_info)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_verify_pch)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_rewrite_objc)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_rewrite_legacy_objc)) ||
             (PhaseArg = DAL.getLastArg(options::OPT__migrate)) ||
             (PhaseArg = DAL.getLastArg(options::OPT__analyze)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_emit_ast))) {
    FinalPhase = phases::Compile;

  // -S only runs up to the backend.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_S)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_fsycl_device_only))) {
    FinalPhase = phases::Backend;

  // -c compilation only runs up to the assembler.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_c))) {
    FinalPhase = phases::Assemble;

  } else if ((PhaseArg = DAL.getLastArg(options::OPT_emit_interface_stubs))) {
    FinalPhase = phases::IfsMerge;

  // Otherwise do everything.
  } else
    FinalPhase = phases::Link;

  if (FinalPhaseArg)
    *FinalPhaseArg = PhaseArg;

  return FinalPhase;
}

static Arg *MakeInputArg(DerivedArgList &Args, const OptTable &Opts,
                         StringRef Value, bool Claim = true) {
  Arg *A = new Arg(Opts.getOption(options::OPT_INPUT), Value,
                   Args.getBaseArgs().MakeIndex(Value), Value.data());
  Args.AddSynthesizedArg(A);
  if (Claim)
    A->claim();
  return A;
}

DerivedArgList *Driver::TranslateInputArgs(const InputArgList &Args) const {
  const llvm::opt::OptTable &Opts = getOpts();
  DerivedArgList *DAL = new DerivedArgList(Args);

  bool HasNostdlib = Args.hasArg(options::OPT_nostdlib);
  bool HasNostdlibxx = Args.hasArg(options::OPT_nostdlibxx);
  bool HasNodefaultlib = Args.hasArg(options::OPT_nodefaultlibs);
  bool IgnoreUnused = false;
  for (Arg *A : Args) {
    if (IgnoreUnused)
      A->claim();

    if (A->getOption().matches(options::OPT_start_no_unused_arguments)) {
      IgnoreUnused = true;
      continue;
    }
    if (A->getOption().matches(options::OPT_end_no_unused_arguments)) {
      IgnoreUnused = false;
      continue;
    }

    // Unfortunately, we have to parse some forwarding options (-Xassembler,
    // -Xlinker, -Xpreprocessor) because we either integrate their functionality
    // (assembler and preprocessor), or bypass a previous driver ('collect2').

    // Rewrite linker options, to replace --no-demangle with a custom internal
    // option.
    if ((A->getOption().matches(options::OPT_Wl_COMMA) ||
         A->getOption().matches(options::OPT_Xlinker)) &&
        A->containsValue("--no-demangle")) {
      // Add the rewritten no-demangle argument.
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_Z_Xlinker__no_demangle));

      // Add the remaining values as Xlinker arguments.
      for (StringRef Val : A->getValues())
        if (Val != "--no-demangle")
          DAL->AddSeparateArg(A, Opts.getOption(options::OPT_Xlinker), Val);

      continue;
    }

    // Rewrite preprocessor options, to replace -Wp,-MD,FOO which is used by
    // some build systems. We don't try to be complete here because we don't
    // care to encourage this usage model.
    if (A->getOption().matches(options::OPT_Wp_COMMA) &&
        (A->getValue(0) == StringRef("-MD") ||
         A->getValue(0) == StringRef("-MMD"))) {
      // Rewrite to -MD/-MMD along with -MF.
      if (A->getValue(0) == StringRef("-MD"))
        DAL->AddFlagArg(A, Opts.getOption(options::OPT_MD));
      else
        DAL->AddFlagArg(A, Opts.getOption(options::OPT_MMD));
      if (A->getNumValues() == 2)
        DAL->AddSeparateArg(A, Opts.getOption(options::OPT_MF), A->getValue(1));
      continue;
    }

    // Rewrite reserved library names.
    if (A->getOption().matches(options::OPT_l)) {
      StringRef Value = A->getValue();

      // Rewrite unless -nostdlib is present.
      if (!HasNostdlib && !HasNodefaultlib && !HasNostdlibxx &&
          Value == "stdc++") {
        DAL->AddFlagArg(A, Opts.getOption(options::OPT_Z_reserved_lib_stdcxx));
        continue;
      }

      // Rewrite unconditionally.
      if (Value == "cc_kext") {
        DAL->AddFlagArg(A, Opts.getOption(options::OPT_Z_reserved_lib_cckext));
        continue;
      }
    }

    // Pick up inputs via the -- option.
    if (A->getOption().matches(options::OPT__DASH_DASH)) {
      A->claim();
      for (StringRef Val : A->getValues())
        DAL->append(MakeInputArg(*DAL, Opts, Val, false));
      continue;
    }

    if (A->getOption().matches(options::OPT_offload_lib_Group)) {
      if (!A->getNumValues()) {
        Diag(clang::diag::warn_drv_unused_argument) << A->getSpelling();
        continue;
      }
    }

    DAL->append(A);
  }

  // DXC mode quits before assembly if an output object file isn't specified.
  if (IsDXCMode() && !Args.hasArg(options::OPT_dxc_Fo))
    DAL->AddFlagArg(nullptr, Opts.getOption(options::OPT_S));

  // Enforce -static if -miamcu is present.
  if (Args.hasFlag(options::OPT_miamcu, options::OPT_mno_iamcu, false))
    DAL->AddFlagArg(nullptr, Opts.getOption(options::OPT_static));

  // Use of -fintelfpga implies -g and -fsycl
  if (Args.hasArg(options::OPT_fintelfpga)) {
    if (!Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false))
      DAL->AddFlagArg(0, Opts.getOption(options::OPT_fsycl));
    // if any -gN option is provided, use that.
    if (Arg *A = Args.getLastArg(options::OPT_gN_Group))
      DAL->append(A);
    else
      DAL->AddFlagArg(0, Opts.getOption(options::OPT_g_Flag));
  }

// Add a default value of -mlinker-version=, if one was given and the user
// didn't specify one.
#if defined(HOST_LINK_VERSION)
  if (!Args.hasArg(options::OPT_mlinker_version_EQ) &&
      strlen(HOST_LINK_VERSION) > 0) {
    DAL->AddJoinedArg(0, Opts.getOption(options::OPT_mlinker_version_EQ),
                      HOST_LINK_VERSION);
    DAL->getLastArg(options::OPT_mlinker_version_EQ)->claim();
  }
#endif

  return DAL;
}

/// Compute target triple from args.
///
/// This routine provides the logic to compute a target triple from various
/// args passed to the driver and the default triple string.
static llvm::Triple computeTargetTriple(const Driver &D,
                                        StringRef TargetTriple,
                                        const ArgList &Args,
                                        StringRef DarwinArchName = "") {
  // FIXME: Already done in Compilation *Driver::BuildCompilation
  if (const Arg *A = Args.getLastArg(options::OPT_target))
    TargetTriple = A->getValue();

  llvm::Triple Target(llvm::Triple::normalize(TargetTriple));

  // GNU/Hurd's triples should have been -hurd-gnu*, but were historically made
  // -gnu* only, and we can not change this, so we have to detect that case as
  // being the Hurd OS.
  if (TargetTriple.contains("-unknown-gnu") || TargetTriple.contains("-pc-gnu"))
    Target.setOSName("hurd");

  // Handle Apple-specific options available here.
  if (Target.isOSBinFormatMachO()) {
    // If an explicit Darwin arch name is given, that trumps all.
    if (!DarwinArchName.empty()) {
      tools::darwin::setTripleTypeForMachOArchName(Target, DarwinArchName,
                                                   Args);
      return Target;
    }

    // Handle the Darwin '-arch' flag.
    if (Arg *A = Args.getLastArg(options::OPT_arch)) {
      StringRef ArchName = A->getValue();
      tools::darwin::setTripleTypeForMachOArchName(Target, ArchName, Args);
    }
  }

  // Handle pseudo-target flags '-mlittle-endian'/'-EL' and
  // '-mbig-endian'/'-EB'.
  if (Arg *A = Args.getLastArgNoClaim(options::OPT_mlittle_endian,
                                      options::OPT_mbig_endian)) {
    llvm::Triple T = A->getOption().matches(options::OPT_mlittle_endian)
                         ? Target.getLittleEndianArchVariant()
                         : Target.getBigEndianArchVariant();
    if (T.getArch() != llvm::Triple::UnknownArch) {
      Target = std::move(T);
      Args.claimAllArgs(options::OPT_mlittle_endian, options::OPT_mbig_endian);
    }
  }

  // Skip further flag support on OSes which don't support '-m32' or '-m64'.
  if (Target.getArch() == llvm::Triple::tce)
    return Target;

  // On AIX, the env OBJECT_MODE may affect the resulting arch variant.
  if (Target.isOSAIX()) {
    if (std::optional<std::string> ObjectModeValue =
            llvm::sys::Process::GetEnv("OBJECT_MODE")) {
      StringRef ObjectMode = *ObjectModeValue;
      llvm::Triple::ArchType AT = llvm::Triple::UnknownArch;

      if (ObjectMode.equals("64")) {
        AT = Target.get64BitArchVariant().getArch();
      } else if (ObjectMode.equals("32")) {
        AT = Target.get32BitArchVariant().getArch();
      } else {
        D.Diag(diag::err_drv_invalid_object_mode) << ObjectMode;
      }

      if (AT != llvm::Triple::UnknownArch && AT != Target.getArch())
        Target.setArch(AT);
    }
  }

  // The `-maix[32|64]` flags are only valid for AIX targets.
  if (Arg *A = Args.getLastArgNoClaim(options::OPT_maix32, options::OPT_maix64);
      A && !Target.isOSAIX())
    D.Diag(diag::err_drv_unsupported_opt_for_target)
        << A->getAsString(Args) << Target.str();

  // Handle pseudo-target flags '-m64', '-mx32', '-m32' and '-m16'.
  Arg *A = Args.getLastArg(options::OPT_m64, options::OPT_mx32,
                           options::OPT_m32, options::OPT_m16,
                           options::OPT_maix32, options::OPT_maix64);
  if (A) {
    llvm::Triple::ArchType AT = llvm::Triple::UnknownArch;

    if (A->getOption().matches(options::OPT_m64) ||
        A->getOption().matches(options::OPT_maix64)) {
      AT = Target.get64BitArchVariant().getArch();
      if (Target.getEnvironment() == llvm::Triple::GNUX32)
        Target.setEnvironment(llvm::Triple::GNU);
      else if (Target.getEnvironment() == llvm::Triple::MuslX32)
        Target.setEnvironment(llvm::Triple::Musl);
    } else if (A->getOption().matches(options::OPT_mx32) &&
               Target.get64BitArchVariant().getArch() == llvm::Triple::x86_64) {
      AT = llvm::Triple::x86_64;
      if (Target.getEnvironment() == llvm::Triple::Musl)
        Target.setEnvironment(llvm::Triple::MuslX32);
      else
        Target.setEnvironment(llvm::Triple::GNUX32);
    } else if (A->getOption().matches(options::OPT_m32) ||
               A->getOption().matches(options::OPT_maix32)) {
      AT = Target.get32BitArchVariant().getArch();
      if (Target.getEnvironment() == llvm::Triple::GNUX32)
        Target.setEnvironment(llvm::Triple::GNU);
      else if (Target.getEnvironment() == llvm::Triple::MuslX32)
        Target.setEnvironment(llvm::Triple::Musl);
    } else if (A->getOption().matches(options::OPT_m16) &&
               Target.get32BitArchVariant().getArch() == llvm::Triple::x86) {
      AT = llvm::Triple::x86;
      Target.setEnvironment(llvm::Triple::CODE16);
    }

    if (AT != llvm::Triple::UnknownArch && AT != Target.getArch()) {
      Target.setArch(AT);
      if (Target.isWindowsGNUEnvironment())
        toolchains::MinGW::fixTripleArch(D, Target, Args);
    }
  }

  // Handle -miamcu flag.
  if (Args.hasFlag(options::OPT_miamcu, options::OPT_mno_iamcu, false)) {
    if (Target.get32BitArchVariant().getArch() != llvm::Triple::x86)
      D.Diag(diag::err_drv_unsupported_opt_for_target) << "-miamcu"
                                                       << Target.str();

    if (A && !A->getOption().matches(options::OPT_m32))
      D.Diag(diag::err_drv_argument_not_allowed_with)
          << "-miamcu" << A->getBaseArg().getAsString(Args);

    Target.setArch(llvm::Triple::x86);
    Target.setArchName("i586");
    Target.setEnvironment(llvm::Triple::UnknownEnvironment);
    Target.setEnvironmentName("");
    Target.setOS(llvm::Triple::ELFIAMCU);
    Target.setVendor(llvm::Triple::UnknownVendor);
    Target.setVendorName("intel");
  }

  // If target is MIPS adjust the target triple
  // accordingly to provided ABI name.
  if (Target.isMIPS()) {
    if ((A = Args.getLastArg(options::OPT_mabi_EQ))) {
      StringRef ABIName = A->getValue();
      if (ABIName == "32") {
        Target = Target.get32BitArchVariant();
        if (Target.getEnvironment() == llvm::Triple::GNUABI64 ||
            Target.getEnvironment() == llvm::Triple::GNUABIN32)
          Target.setEnvironment(llvm::Triple::GNU);
      } else if (ABIName == "n32") {
        Target = Target.get64BitArchVariant();
        if (Target.getEnvironment() == llvm::Triple::GNU ||
            Target.getEnvironment() == llvm::Triple::GNUABI64)
          Target.setEnvironment(llvm::Triple::GNUABIN32);
      } else if (ABIName == "64") {
        Target = Target.get64BitArchVariant();
        if (Target.getEnvironment() == llvm::Triple::GNU ||
            Target.getEnvironment() == llvm::Triple::GNUABIN32)
          Target.setEnvironment(llvm::Triple::GNUABI64);
      }
    }
  }

  // If target is RISC-V adjust the target triple according to
  // provided architecture name
  if (Target.isRISCV()) {
    if (Args.hasArg(options::OPT_march_EQ) ||
        Args.hasArg(options::OPT_mcpu_EQ)) {
      StringRef ArchName = tools::riscv::getRISCVArch(Args, Target);
      auto ISAInfo = llvm::RISCVISAInfo::parseArchString(
          ArchName, /*EnableExperimentalExtensions=*/true);
      if (!llvm::errorToBool(ISAInfo.takeError())) {
        unsigned XLen = (*ISAInfo)->getXLen();
        if (XLen == 32)
          Target.setArch(llvm::Triple::riscv32);
        else if (XLen == 64)
          Target.setArch(llvm::Triple::riscv64);
      }
    }
  }

  return Target;
}

// Parse the LTO options and record the type of LTO compilation
// based on which -f(no-)?lto(=.*)? or -f(no-)?offload-lto(=.*)?
// option occurs last.
static driver::LTOKind parseLTOMode(Driver &D, const llvm::opt::ArgList &Args,
                                    OptSpecifier OptEq, OptSpecifier OptNeg) {
  if (!Args.hasFlag(OptEq, OptNeg, false))
    return LTOK_None;

  const Arg *A = Args.getLastArg(OptEq);
  StringRef LTOName = A->getValue();

  driver::LTOKind LTOMode = llvm::StringSwitch<LTOKind>(LTOName)
                                .Case("full", LTOK_Full)
                                .Case("thin", LTOK_Thin)
                                .Default(LTOK_Unknown);

  if (LTOMode == LTOK_Unknown) {
    D.Diag(diag::err_drv_unsupported_option_argument)
        << A->getSpelling() << A->getValue();
    return LTOK_None;
  }
  return LTOMode;
}

// Parse the LTO options.
void Driver::setLTOMode(const llvm::opt::ArgList &Args) {
  LTOMode =
      parseLTOMode(*this, Args, options::OPT_flto_EQ, options::OPT_fno_lto);

  OffloadLTOMode = parseLTOMode(*this, Args, options::OPT_foffload_lto_EQ,
                                options::OPT_fno_offload_lto);

  // Try to enable `-foffload-lto=full` if `-fopenmp-target-jit` is on.
  if (Args.hasFlag(options::OPT_fopenmp_target_jit,
                   options::OPT_fno_openmp_target_jit, false)) {
    if (Arg *A = Args.getLastArg(options::OPT_foffload_lto_EQ,
                                 options::OPT_fno_offload_lto))
      if (OffloadLTOMode != LTOK_Full)
        Diag(diag::err_drv_incompatible_options)
            << A->getSpelling() << "-fopenmp-target-jit";
    OffloadLTOMode = LTOK_Full;
  }
}

/// Compute the desired OpenMP runtime from the flags provided.
Driver::OpenMPRuntimeKind Driver::getOpenMPRuntime(const ArgList &Args) const {
  StringRef RuntimeName(CLANG_DEFAULT_OPENMP_RUNTIME);

  const Arg *A = Args.getLastArg(options::OPT_fopenmp_EQ);
  if (A)
    RuntimeName = A->getValue();

  auto RT = llvm::StringSwitch<OpenMPRuntimeKind>(RuntimeName)
                .Case("libomp", OMPRT_OMP)
                .Case("libgomp", OMPRT_GOMP)
                .Case("libiomp5", OMPRT_IOMP5)
                .Default(OMPRT_Unknown);

  if (RT == OMPRT_Unknown) {
    if (A)
      Diag(diag::err_drv_unsupported_option_argument)
          << A->getSpelling() << A->getValue();
    else
      // FIXME: We could use a nicer diagnostic here.
      Diag(diag::err_drv_unsupported_opt) << "-fopenmp";
  }

  return RT;
}

static bool isValidSYCLTriple(llvm::Triple T) {
  // NVPTX is valid for SYCL.
  if (T.isNVPTX())
    return true;

  // AMDGCN is valid for SYCL
  if (T.isAMDGCN())
    return true;

  // Check for invalid SYCL device triple values.
  // Non-SPIR arch.
  if (!T.isSPIR())
    return false;
  // SPIR arch, but has invalid SubArch for AOT.
  StringRef A(T.getArchName());
  if (T.getSubArch() == llvm::Triple::NoSubArch &&
      ((T.getArch() == llvm::Triple::spir && !A.equals("spir")) ||
       (T.getArch() == llvm::Triple::spir64 && !A.equals("spir64"))))
    return false;
  return true;
}

static const char *getDefaultSYCLArch(Compilation &C) {
  if (C.getDefaultToolChain().getTriple().getArch() == llvm::Triple::x86)
    return "spir";
  return "spir64";
}

static bool addSYCLDefaultTriple(Compilation &C,
                                 SmallVectorImpl<llvm::Triple> &SYCLTriples) {
  /// Returns true if a triple is added to SYCLTriples, false otherwise
  if (!C.getDriver().isSYCLDefaultTripleImplied())
    return false;
  if (C.getInputArgs().hasArg(options::OPT_fsycl_force_target_EQ))
    return false;
  for (const auto &SYCLTriple : SYCLTriples) {
    if (SYCLTriple.getSubArch() == llvm::Triple::NoSubArch &&
        SYCLTriple.isSPIR())
      return false;
    // If we encounter a known non-spir* target, do not add the default triple.
    if (SYCLTriple.isNVPTX() || SYCLTriple.isAMDGCN())
      return false;
  }
  // Add the default triple as it was not found.
  llvm::Triple DefaultTriple =
      C.getDriver().MakeSYCLDeviceTriple(getDefaultSYCLArch(C));
  SYCLTriples.insert(SYCLTriples.begin(), DefaultTriple);
  return true;
}

void Driver::CreateOffloadingDeviceToolChains(Compilation &C,
                                              InputList &Inputs) {

  //
  // CUDA/HIP
  //
  // We need to generate a CUDA/HIP toolchain if any of the inputs has a CUDA
  // or HIP type. However, mixed CUDA/HIP compilation is not supported.
  using namespace tools::SYCL;
  bool IsCuda =
      llvm::any_of(Inputs, [](std::pair<types::ID, const llvm::opt::Arg *> &I) {
        return types::isCuda(I.first);
      });
  bool IsHIP =
      llvm::any_of(Inputs,
                   [](std::pair<types::ID, const llvm::opt::Arg *> &I) {
                     return types::isHIP(I.first);
                   }) ||
      C.getInputArgs().hasArg(options::OPT_hip_link) ||
      C.getInputArgs().hasArg(options::OPT_hipstdpar);
  if (IsCuda && IsHIP) {
    Diag(clang::diag::err_drv_mix_cuda_hip);
    return;
  }
  if (IsCuda) {
    const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
    const llvm::Triple &HostTriple = HostTC->getTriple();
    auto OFK = Action::OFK_Cuda;
    auto CudaTriple =
        getNVIDIAOffloadTargetTriple(*this, C.getInputArgs(), HostTriple);
    if (!CudaTriple)
      return;
    // Use the CUDA and host triples as the key into the ToolChains map,
    // because the device toolchain we create depends on both.
    auto &CudaTC = ToolChains[CudaTriple->str() + "/" + HostTriple.str()];
    if (!CudaTC) {
      CudaTC = std::make_unique<toolchains::CudaToolChain>(
          *this, *CudaTriple, *HostTC, C.getInputArgs(), OFK);

    // Emit a warning if the detected CUDA version is too new.
    CudaInstallationDetector &CudaInstallation =
          static_cast<toolchains::CudaToolChain &>(*CudaTC).CudaInstallation;
      if (CudaInstallation.isValid())
        CudaInstallation.WarnIfUnsupportedVersion();
    }
    C.addOffloadDeviceToolChain(CudaTC.get(), OFK);
  } else if (IsHIP) {
    if (auto *OMPTargetArg =
            C.getInputArgs().getLastArg(options::OPT_fopenmp_targets_EQ)) {
      Diag(clang::diag::err_drv_unsupported_opt_for_language_mode)
          << OMPTargetArg->getSpelling() << "HIP";
      return;
    }
    const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
    auto OFK = Action::OFK_HIP;
    auto HIPTriple = getHIPOffloadTargetTriple(*this, C.getInputArgs());
    if (!HIPTriple)
      return;
    auto *HIPTC = &getOffloadingDeviceToolChain(C.getInputArgs(), *HIPTriple,
                                                *HostTC, OFK);
    assert(HIPTC && "Could not create offloading device tool chain.");
    C.addOffloadDeviceToolChain(HIPTC, OFK);
  }

  //
  // OpenMP
  //
  // We need to generate an OpenMP toolchain if the user specified targets with
  // the -fopenmp-targets option or used --offload-arch with OpenMP enabled.
  bool IsOpenMPOffloading =
      C.getInputArgs().hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                               options::OPT_fno_openmp, false) &&
      (C.getInputArgs().hasArg(options::OPT_fopenmp_targets_EQ) ||
       C.getInputArgs().hasArg(options::OPT_offload_arch_EQ));
  if (IsOpenMPOffloading) {
    // We expect that -fopenmp-targets is always used in conjunction with the
    // option -fopenmp specifying a valid runtime with offloading support, i.e.
    // libomp or libiomp.
    OpenMPRuntimeKind RuntimeKind = getOpenMPRuntime(C.getInputArgs());
    if (RuntimeKind != OMPRT_OMP && RuntimeKind != OMPRT_IOMP5) {
      Diag(clang::diag::err_drv_expecting_fopenmp_with_fopenmp_targets);
      return;
    }

    llvm::StringMap<llvm::DenseSet<StringRef>> DerivedArchs;
    llvm::StringMap<StringRef> FoundNormalizedTriples;
    std::multiset<StringRef> OpenMPTriples;

    // If the user specified -fopenmp-targets= we create a toolchain for each
    // valid triple. Otherwise, if only --offload-arch= was specified we instead
    // attempt to derive the appropriate toolchains from the arguments.
    if (Arg *OpenMPTargets =
            C.getInputArgs().getLastArg(options::OPT_fopenmp_targets_EQ)) {
      if (OpenMPTargets && !OpenMPTargets->getNumValues()) {
        Diag(clang::diag::warn_drv_empty_joined_argument)
            << OpenMPTargets->getAsString(C.getInputArgs());
        return;
      }
      for (StringRef T : OpenMPTargets->getValues())
        OpenMPTriples.insert(T);
    } else if (C.getInputArgs().hasArg(options::OPT_offload_arch_EQ) &&
               !IsHIP && !IsCuda) {
      const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
      auto AMDTriple = getHIPOffloadTargetTriple(*this, C.getInputArgs());
      auto NVPTXTriple = getNVIDIAOffloadTargetTriple(*this, C.getInputArgs(),
                                                      HostTC->getTriple());

      // Attempt to deduce the offloading triple from the set of architectures.
      // We can only correctly deduce NVPTX / AMDGPU triples currently. We need
      // to temporarily create these toolchains so that we can access tools for
      // inferring architectures.
      llvm::DenseSet<StringRef> Archs;
      if (NVPTXTriple) {
        auto TempTC = std::make_unique<toolchains::CudaToolChain>(
            *this, *NVPTXTriple, *HostTC, C.getInputArgs(), Action::OFK_None);
        for (StringRef Arch : getOffloadArchs(
                 C, C.getArgs(), Action::OFK_OpenMP, &*TempTC, true))
          Archs.insert(Arch);
      }
      if (AMDTriple) {
        auto TempTC = std::make_unique<toolchains::AMDGPUOpenMPToolChain>(
            *this, *AMDTriple, *HostTC, C.getInputArgs());
        for (StringRef Arch : getOffloadArchs(
                 C, C.getArgs(), Action::OFK_OpenMP, &*TempTC, true))
          Archs.insert(Arch);
      }
      if (!AMDTriple && !NVPTXTriple) {
        for (StringRef Arch :
             getOffloadArchs(C, C.getArgs(), Action::OFK_OpenMP, nullptr, true))
          Archs.insert(Arch);
      }

      for (StringRef Arch : Archs) {
        if (NVPTXTriple && IsNVIDIAGpuArch(StringToCudaArch(
                               getProcessorFromTargetID(*NVPTXTriple, Arch)))) {
          DerivedArchs[NVPTXTriple->getTriple()].insert(Arch);
        } else if (AMDTriple &&
                   IsAMDGpuArch(StringToCudaArch(
                       getProcessorFromTargetID(*AMDTriple, Arch)))) {
          DerivedArchs[AMDTriple->getTriple()].insert(Arch);
        } else {
          Diag(clang::diag::err_drv_failed_to_deduce_target_from_arch) << Arch;
          return;
        }
      }

      // If the set is empty then we failed to find a native architecture.
      if (Archs.empty()) {
        Diag(clang::diag::err_drv_failed_to_deduce_target_from_arch)
            << "native";
        return;
      }

      for (const auto &TripleAndArchs : DerivedArchs)
        OpenMPTriples.insert(TripleAndArchs.first());
    }

    for (StringRef Val : OpenMPTriples) {
      llvm::Triple TT(ToolChain::getOpenMPTriple(Val));
      std::string NormalizedName = TT.normalize();

      // Make sure we don't have a duplicate triple.
      auto Duplicate = FoundNormalizedTriples.find(NormalizedName);
      if (Duplicate != FoundNormalizedTriples.end()) {
        Diag(clang::diag::warn_drv_omp_offload_target_duplicate)
            << Val << Duplicate->second;
        continue;
      }

      // Store the current triple so that we can check for duplicates in the
      // following iterations.
      FoundNormalizedTriples[NormalizedName] = Val;

      // If the specified target is invalid, emit a diagnostic.
      if (TT.getArch() == llvm::Triple::UnknownArch)
        Diag(clang::diag::err_drv_invalid_omp_target) << Val;
      else {
        const ToolChain *TC;
        // Device toolchains have to be selected differently. They pair host
        // and device in their implementation.
        if (TT.isNVPTX() || TT.isAMDGCN()) {
          const ToolChain *HostTC =
              C.getSingleOffloadToolChain<Action::OFK_Host>();
          assert(HostTC && "Host toolchain should be always defined.");
          auto &DeviceTC =
              ToolChains[TT.str() + "/" + HostTC->getTriple().normalize()];
          if (!DeviceTC) {
            if (TT.isNVPTX())
              DeviceTC = std::make_unique<toolchains::CudaToolChain>(
                  *this, TT, *HostTC, C.getInputArgs(), Action::OFK_OpenMP);
            else if (TT.isAMDGCN())
              DeviceTC = std::make_unique<toolchains::AMDGPUOpenMPToolChain>(
                  *this, TT, *HostTC, C.getInputArgs());
            else
              assert(DeviceTC && "Device toolchain not defined.");
          }

          TC = DeviceTC.get();
        } else
          TC = &getToolChain(C.getInputArgs(), TT);
        C.addOffloadDeviceToolChain(TC, Action::OFK_OpenMP);
        if (DerivedArchs.contains(TT.getTriple()))
          KnownArchs[TC] = DerivedArchs[TT.getTriple()];
      }
    }
  } else if (C.getInputArgs().hasArg(options::OPT_fopenmp_targets_EQ)) {
    Diag(clang::diag::err_drv_expecting_fopenmp_with_fopenmp_targets);
    return;
  }

  //
  // SYCL
  //
  // We need to generate a SYCL toolchain if the user specified targets with
  // the -fsycl-targets, -fsycl-add-targets or -fsycl-link-targets option.
  // If -fsycl is supplied without any of these we will assume SPIR-V.
  // Use of -fsycl-device-only overrides -fsycl.
  bool HasValidSYCLRuntime =
      C.getInputArgs().hasFlag(options::OPT_fsycl, options::OPT_fno_sycl,
                               false) ||
      C.getInputArgs().hasArg(options::OPT_fsycl_device_only);

  Arg *SYCLfpga = C.getInputArgs().getLastArg(options::OPT_fintelfpga);

  // Make -fintelfpga flag imply -fsycl.
  if (SYCLfpga && !HasValidSYCLRuntime)
    HasValidSYCLRuntime = true;

  // A mechanism for retrieving SYCL-specific options, erroring out
  // if SYCL offloading wasn't enabled prior to that
  auto getArgRequiringSYCLRuntime = [&](OptSpecifier OptId) -> Arg * {
    Arg *SYCLArg = C.getInputArgs().getLastArg(OptId);
    if (SYCLArg && !HasValidSYCLRuntime) {
      Diag(clang::diag::err_drv_expecting_fsycl_with_sycl_opt)
          // Dropping the '=' symbol, which would otherwise pollute
          // the diagnostics for the most of options
          << SYCLArg->getSpelling().split('=').first;
      return nullptr;
    }
    return SYCLArg;
  };

  Arg *SYCLTargets = getArgRequiringSYCLRuntime(options::OPT_fsycl_targets_EQ);
  Arg *SYCLLinkTargets =
      getArgRequiringSYCLRuntime(options::OPT_fsycl_link_targets_EQ);
  Arg *SYCLAddTargets =
      getArgRequiringSYCLRuntime(options::OPT_fsycl_add_targets_EQ);
  Arg *SYCLLink = getArgRequiringSYCLRuntime(options::OPT_fsycl_link_EQ);

  // Check if -fsycl-host-compiler is used in conjunction with -fsycl.
  Arg *SYCLHostCompiler =
      getArgRequiringSYCLRuntime(options::OPT_fsycl_host_compiler_EQ);
  Arg *SYCLHostCompilerOptions =
      getArgRequiringSYCLRuntime(options::OPT_fsycl_host_compiler_options_EQ);

  // -fsycl-targets cannot be used with -fsycl-link-targets
  if (SYCLTargets && SYCLLinkTargets)
    Diag(clang::diag::err_drv_option_conflict)
        << SYCLTargets->getSpelling() << SYCLLinkTargets->getSpelling();
  // -fsycl-link-targets and -fsycl-add-targets cannot be used together
  if (SYCLLinkTargets && SYCLAddTargets)
    Diag(clang::diag::err_drv_option_conflict)
        << SYCLLinkTargets->getSpelling() << SYCLAddTargets->getSpelling();
  // -fsycl-link-targets is not allowed with -fsycl-link
  if (SYCLLinkTargets && SYCLLink)
    Diag(clang::diag::err_drv_option_conflict)
        << SYCLLink->getSpelling() << SYCLLinkTargets->getSpelling();
  // -fsycl-targets cannot be used with -fintelfpga
  if (SYCLTargets && SYCLfpga)
    Diag(clang::diag::err_drv_option_conflict)
        << SYCLTargets->getSpelling() << SYCLfpga->getSpelling();
  // -fsycl-host-compiler-options cannot be used without -fsycl-host-compiler
  if (SYCLHostCompilerOptions && !SYCLHostCompiler)
    Diag(clang::diag::warn_drv_opt_requires_opt)
        << SYCLHostCompilerOptions->getSpelling().split('=').first
        << "-fsycl-host-compiler";

  auto argSYCLIncompatible = [&](OptSpecifier OptId) {
    if (!HasValidSYCLRuntime)
      return;
    if (Arg *IncompatArg = C.getInputArgs().getLastArg(OptId))
      Diag(clang::diag::err_drv_fsycl_unsupported_with_opt)
          << IncompatArg->getSpelling();
  };
  // -static-libstdc++ is not compatible with -fsycl.
  argSYCLIncompatible(options::OPT_static_libstdcxx);
  // -ffreestanding cannot be used with -fsycl
  argSYCLIncompatible(options::OPT_ffreestanding);

  // Diagnose incorrect inputs to SYCL options.
  // FIXME: Since the option definition includes the list of possible values,
  // the validation must be automatic, not requiring separate disjointed code
  // blocks accross the driver code. Long-term, the detection of incorrect
  // values must happen at the level of TableGen and Arg class design, with
  // Compilation/Driver class constructors handling the driver-specific
  // diagnostic output.
  auto checkSingleArgValidity = [&](Arg *A,
                                    SmallVector<StringRef, 4> AllowedValues) {
    if (!A)
      return;
    const char *ArgValue = A->getValue();
    for (const StringRef AllowedValue : AllowedValues)
      if (AllowedValue.equals(ArgValue))
        return;
    Diag(clang::diag::err_drv_invalid_argument_to_option)
        << ArgValue << A->getOption().getName();
  };
  Arg *DeviceCodeSplit =
      C.getInputArgs().getLastArg(options::OPT_fsycl_device_code_split_EQ);
  checkSingleArgValidity(SYCLLink, {"early", "image"});
  checkSingleArgValidity(DeviceCodeSplit,
                         {"per_kernel", "per_source", "auto", "off"});

  Arg *SYCLForceTarget =
      getArgRequiringSYCLRuntime(options::OPT_fsycl_force_target_EQ);
  if (SYCLForceTarget) {
    StringRef Val(SYCLForceTarget->getValue());
    llvm::Triple TT(MakeSYCLDeviceTriple(Val));
    if (!isValidSYCLTriple(TT))
      Diag(clang::diag::err_drv_invalid_sycl_target) << Val;
  }
  bool HasSYCLTargetsOption = SYCLTargets || SYCLLinkTargets || SYCLAddTargets;

  llvm::StringMap<StringRef> FoundNormalizedTriples;
  llvm::SmallVector<llvm::Triple, 4> UniqueSYCLTriplesVec;
  if (HasSYCLTargetsOption) {
    // At this point, we know we have a valid combination
    // of -fsycl*target options passed
    Arg *SYCLTargetsValues = SYCLTargets ? SYCLTargets : SYCLLinkTargets;
    if (SYCLTargetsValues) {
      if (SYCLTargetsValues->getNumValues()) {

        // Multiple targets are currently not supported when using
        // -fsycl-force-target as the bundler does not allow for multiple
        // outputs of the same target.
        if (SYCLForceTarget && SYCLTargetsValues->getNumValues() > 1)
          Diag(clang::diag::err_drv_multiple_target_with_forced_target)
              << SYCLTargetsValues->getAsString(C.getInputArgs())
              << SYCLForceTarget->getAsString(C.getInputArgs());

        for (StringRef Val : SYCLTargetsValues->getValues()) {
          StringRef UserTargetName(Val);
          if (auto Device = gen::isGPUTarget<gen::IntelGPU>(Val)) {
            if (Device->empty()) {
              Diag(clang::diag::err_drv_invalid_sycl_target) << Val;
              continue;
            }
            UserTargetName = "spir64_gen";
          } else if (auto Device = gen::isGPUTarget<gen::NvidiaGPU>(Val)) {
            if (Device->empty()) {
              Diag(clang::diag::err_drv_invalid_sycl_target) << Val;
              continue;
            }
            UserTargetName = "nvptx64-nvidia-cuda";
          } else if (auto Device = gen::isGPUTarget<gen::AmdGPU>(Val)) {
            if (Device->empty()) {
              Diag(clang::diag::err_drv_invalid_sycl_target) << Val;
              continue;
            }
            UserTargetName = "amdgcn-amd-amdhsa";
          } else if (Val == "native_cpu") {
            const ToolChain *HostTC =
                C.getSingleOffloadToolChain<Action::OFK_Host>();
            llvm::Triple HostTriple = HostTC->getTriple();
            UniqueSYCLTriplesVec.push_back(HostTriple);
            continue;
          }

          if (!isValidSYCLTriple(MakeSYCLDeviceTriple(UserTargetName))) {
            Diag(clang::diag::err_drv_invalid_sycl_target) << Val;
            continue;
          }

          // Make sure we don't have a duplicate triple.
          std::string NormalizedName = MakeSYCLDeviceTriple(Val).normalize();
          auto Duplicate = FoundNormalizedTriples.find(NormalizedName);
          if (Duplicate != FoundNormalizedTriples.end()) {
            Diag(clang::diag::warn_drv_sycl_offload_target_duplicate)
                << Val << Duplicate->second;
            continue;
          }

          // Store the current triple so that we can check for duplicates in
          // the following iterations.
          FoundNormalizedTriples[NormalizedName] = Val;
          UniqueSYCLTriplesVec.push_back(MakeSYCLDeviceTriple(UserTargetName));
        }
        addSYCLDefaultTriple(C, UniqueSYCLTriplesVec);
      } else
        Diag(clang::diag::warn_drv_empty_joined_argument)
            << SYCLTargetsValues->getAsString(C.getInputArgs());
    }
    // -fsycl-add-targets is a list of paired items (Triple and file) which are
    // gathered and used to be linked into the final device binary. This can
    // be used with -fsycl-targets to put together the final conglomerate binary
    if (SYCLAddTargets) {
      if (SYCLAddTargets->getNumValues()) {
        // Use of -fsycl-add-targets adds additional files to the SYCL device
        // link step.  Regular offload processing occurs below
        for (StringRef Val : SYCLAddTargets->getValues()) {
          // Parse out the Triple and Input (triple:binary) and create a
          // ToolChain for each entry.
          // The expected format is 'triple:file', any other format will
          // not be accepted.
          std::pair<StringRef, StringRef> I = Val.split(':');
          if (!I.first.empty() && !I.second.empty()) {
            llvm::Triple TT(I.first);
            if (!isValidSYCLTriple(TT)) {
              Diag(clang::diag::err_drv_invalid_sycl_target) << I.first;
              continue;
            }
            std::string NormalizedName = TT.normalize();

            // Make sure we don't have a duplicate triple.
            auto Duplicate = FoundNormalizedTriples.find(NormalizedName);
            if (Duplicate != FoundNormalizedTriples.end())
              // The toolchain for this triple was already created
              continue;

            // Store the current triple so that we can check for duplicates in
            // the following iterations.
            FoundNormalizedTriples[NormalizedName] = Val;
            UniqueSYCLTriplesVec.push_back(TT);
          } else {
            // No colon found, do not use the input
            C.getDriver().Diag(diag::err_drv_unsupported_option_argument)
                << SYCLAddTargets->getSpelling() << Val;
          }
        }
      } else
        Diag(clang::diag::warn_drv_empty_joined_argument)
            << SYCLAddTargets->getAsString(C.getInputArgs());
    }
  } else {
    // If -fsycl is supplied without -fsycl-*targets we will assume SPIR-V
    // unless -fintelfpga is supplied, which uses SPIR-V with fpga AOT.
    // For -fsycl-device-only, we also setup the implied triple as needed.
    if (HasValidSYCLRuntime) {
      StringRef SYCLTargetArch = getDefaultSYCLArch(C);
      if (SYCLfpga)
        // Triple for -fintelfpga is spir64_fpga.
        SYCLTargetArch = "spir64_fpga";
      UniqueSYCLTriplesVec.push_back(MakeSYCLDeviceTriple(SYCLTargetArch));
      addSYCLDefaultTriple(C, UniqueSYCLTriplesVec);
    }
  }
  // -fno-sycl-libspirv flag is reserved for very unusual cases where the
  // libspirv library is not linked when using CUDA/HIP: so output appropriate
  // warnings.
  if (C.getInputArgs().hasArg(options::OPT_fno_sycl_libspirv)) {
    for (auto &TT : UniqueSYCLTriplesVec) {
      if (TT.isNVPTX() || TT.isAMDGCN()) {
        Diag(diag::warn_flag_no_sycl_libspirv) << TT.getTriple();
      } else {
        Diag(diag::warn_drv_unsupported_option_for_target)
            << "-fno-sycl-libspirv" << TT.getTriple();
      }
    }
  }
  // Define macros associated with `any_device_has/all_devices_have` according
  // to the aspects defined in the DeviceConfigFile for the SYCL targets.
  populateSYCLDeviceTraitsMacrosArgs(C.getInputArgs(), UniqueSYCLTriplesVec);
  // We'll need to use the SYCL and host triples as the key into
  // getOffloadingDeviceToolChain, because the device toolchains we're
  // going to create will depend on both.
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  for (auto &TT : UniqueSYCLTriplesVec) {
    auto SYCLTC = &getOffloadingDeviceToolChain(C.getInputArgs(), TT, *HostTC,
                                                Action::OFK_SYCL);
    C.addOffloadDeviceToolChain(SYCLTC, Action::OFK_SYCL);
  }

  //
  // TODO: Add support for other offloading programming models here.
  //
}

static void appendOneArg(InputArgList &Args, const Arg *Opt,
                         const Arg *BaseArg) {
  // The args for config files or /clang: flags belong to different InputArgList
  // objects than Args. This copies an Arg from one of those other InputArgLists
  // to the ownership of Args.
  unsigned Index = Args.MakeIndex(Opt->getSpelling());
  Arg *Copy = new llvm::opt::Arg(Opt->getOption(), Args.getArgString(Index),
                                 Index, BaseArg);
  Copy->getValues() = Opt->getValues();
  if (Opt->isClaimed())
    Copy->claim();
  Copy->setOwnsValues(Opt->getOwnsValues());
  Opt->setOwnsValues(false);
  Args.append(Copy);
}

bool Driver::readConfigFile(StringRef FileName,
                            llvm::cl::ExpansionContext &ExpCtx) {
  // Try opening the given file.
  auto Status = getVFS().status(FileName);
  if (!Status) {
    Diag(diag::err_drv_cannot_open_config_file)
        << FileName << Status.getError().message();
    return true;
  }
  if (Status->getType() != llvm::sys::fs::file_type::regular_file) {
    Diag(diag::err_drv_cannot_open_config_file)
        << FileName << "not a regular file";
    return true;
  }

  // Try reading the given file.
  SmallVector<const char *, 32> NewCfgArgs;
  if (llvm::Error Err = ExpCtx.readConfigFile(FileName, NewCfgArgs)) {
    Diag(diag::err_drv_cannot_read_config_file)
        << FileName << toString(std::move(Err));
    return true;
  }

  // Read options from config file.
  llvm::SmallString<128> CfgFileName(FileName);
  llvm::sys::path::native(CfgFileName);
  bool ContainErrors;
  std::unique_ptr<InputArgList> NewOptions = std::make_unique<InputArgList>(
      ParseArgStrings(NewCfgArgs, /*UseDriverMode=*/true, ContainErrors));
  if (ContainErrors)
    return true;

  // Claim all arguments that come from a configuration file so that the driver
  // does not warn on any that is unused.
  for (Arg *A : *NewOptions)
    A->claim();

  if (!CfgOptions)
    CfgOptions = std::move(NewOptions);
  else {
    // If this is a subsequent config file, append options to the previous one.
    for (auto *Opt : *NewOptions) {
      const Arg *BaseArg = &Opt->getBaseArg();
      if (BaseArg == Opt)
        BaseArg = nullptr;
      appendOneArg(*CfgOptions, Opt, BaseArg);
    }
  }
  ConfigFiles.push_back(std::string(CfgFileName));
  return false;
}

bool Driver::loadConfigFiles() {
  llvm::cl::ExpansionContext ExpCtx(Saver.getAllocator(),
                                    llvm::cl::tokenizeConfigFile);
  ExpCtx.setVFS(&getVFS());

  // Process options that change search path for config files.
  if (CLOptions) {
    if (CLOptions->hasArg(options::OPT_config_system_dir_EQ)) {
      SmallString<128> CfgDir;
      CfgDir.append(
          CLOptions->getLastArgValue(options::OPT_config_system_dir_EQ));
      if (CfgDir.empty() || getVFS().makeAbsolute(CfgDir))
        SystemConfigDir.clear();
      else
        SystemConfigDir = static_cast<std::string>(CfgDir);
    }
    if (CLOptions->hasArg(options::OPT_config_user_dir_EQ)) {
      SmallString<128> CfgDir;
      llvm::sys::fs::expand_tilde(
          CLOptions->getLastArgValue(options::OPT_config_user_dir_EQ), CfgDir);
      if (CfgDir.empty() || getVFS().makeAbsolute(CfgDir))
        UserConfigDir.clear();
      else
        UserConfigDir = static_cast<std::string>(CfgDir);
    }
  }

  // Prepare list of directories where config file is searched for.
  StringRef CfgFileSearchDirs[] = {UserConfigDir, SystemConfigDir, Dir};
  ExpCtx.setSearchDirs(CfgFileSearchDirs);

  // First try to load configuration from the default files, return on error.
  if (loadDefaultConfigFiles(ExpCtx))
    return true;

  // Then load configuration files specified explicitly.
  SmallString<128> CfgFilePath;
  if (CLOptions) {
    for (auto CfgFileName : CLOptions->getAllArgValues(options::OPT_config)) {
      // If argument contains directory separator, treat it as a path to
      // configuration file.
      if (llvm::sys::path::has_parent_path(CfgFileName)) {
        CfgFilePath.assign(CfgFileName);
        if (llvm::sys::path::is_relative(CfgFilePath)) {
          if (getVFS().makeAbsolute(CfgFilePath)) {
            Diag(diag::err_drv_cannot_open_config_file)
                << CfgFilePath << "cannot get absolute path";
            return true;
          }
        }
      } else if (!ExpCtx.findConfigFile(CfgFileName, CfgFilePath)) {
        // Report an error that the config file could not be found.
        Diag(diag::err_drv_config_file_not_found) << CfgFileName;
        for (const StringRef &SearchDir : CfgFileSearchDirs)
          if (!SearchDir.empty())
            Diag(diag::note_drv_config_file_searched_in) << SearchDir;
        return true;
      }

      // Try to read the config file, return on error.
      if (readConfigFile(CfgFilePath, ExpCtx))
        return true;
    }
  }

  // No error occurred.
  return false;
}

bool Driver::loadDefaultConfigFiles(llvm::cl::ExpansionContext &ExpCtx) {
  // Disable default config if CLANG_NO_DEFAULT_CONFIG is set to a non-empty
  // value.
  if (const char *NoConfigEnv = ::getenv("CLANG_NO_DEFAULT_CONFIG")) {
    if (*NoConfigEnv)
      return false;
  }
  if (CLOptions && CLOptions->hasArg(options::OPT_no_default_config))
    return false;

  std::string RealMode = getExecutableForDriverMode(Mode);
  std::string Triple;

  // If name prefix is present, no --target= override was passed via CLOptions
  // and the name prefix is not a valid triple, force it for backwards
  // compatibility.
  if (!ClangNameParts.TargetPrefix.empty() &&
      computeTargetTriple(*this, "/invalid/", *CLOptions).str() ==
          "/invalid/") {
    llvm::Triple PrefixTriple{ClangNameParts.TargetPrefix};
    if (PrefixTriple.getArch() == llvm::Triple::UnknownArch ||
        PrefixTriple.isOSUnknown())
      Triple = PrefixTriple.str();
  }

  // Otherwise, use the real triple as used by the driver.
  if (Triple.empty()) {
    llvm::Triple RealTriple =
        computeTargetTriple(*this, TargetTriple, *CLOptions);
    Triple = RealTriple.str();
    assert(!Triple.empty());
  }

  // Search for config files in the following order:
  // 1. <triple>-<mode>.cfg using real driver mode
  //    (e.g. i386-pc-linux-gnu-clang++.cfg).
  // 2. <triple>-<mode>.cfg using executable suffix
  //    (e.g. i386-pc-linux-gnu-clang-g++.cfg for *clang-g++).
  // 3. <triple>.cfg + <mode>.cfg using real driver mode
  //    (e.g. i386-pc-linux-gnu.cfg + clang++.cfg).
  // 4. <triple>.cfg + <mode>.cfg using executable suffix
  //    (e.g. i386-pc-linux-gnu.cfg + clang-g++.cfg for *clang-g++).

  // Try loading <triple>-<mode>.cfg, and return if we find a match.
  SmallString<128> CfgFilePath;
  std::string CfgFileName = Triple + '-' + RealMode + ".cfg";
  if (ExpCtx.findConfigFile(CfgFileName, CfgFilePath))
    return readConfigFile(CfgFilePath, ExpCtx);

  bool TryModeSuffix = !ClangNameParts.ModeSuffix.empty() &&
                       ClangNameParts.ModeSuffix != RealMode;
  if (TryModeSuffix) {
    CfgFileName = Triple + '-' + ClangNameParts.ModeSuffix + ".cfg";
    if (ExpCtx.findConfigFile(CfgFileName, CfgFilePath))
      return readConfigFile(CfgFilePath, ExpCtx);
  }

  // Try loading <mode>.cfg, and return if loading failed.  If a matching file
  // was not found, still proceed on to try <triple>.cfg.
  CfgFileName = RealMode + ".cfg";
  if (ExpCtx.findConfigFile(CfgFileName, CfgFilePath)) {
    if (readConfigFile(CfgFilePath, ExpCtx))
      return true;
  } else if (TryModeSuffix) {
    CfgFileName = ClangNameParts.ModeSuffix + ".cfg";
    if (ExpCtx.findConfigFile(CfgFileName, CfgFilePath) &&
        readConfigFile(CfgFilePath, ExpCtx))
      return true;
  }

  // Try loading <triple>.cfg and return if we find a match.
  CfgFileName = Triple + ".cfg";
  if (ExpCtx.findConfigFile(CfgFileName, CfgFilePath))
    return readConfigFile(CfgFilePath, ExpCtx);

  // If we were unable to find a config file deduced from executable name,
  // that is not an error.
  return false;
}

Compilation *Driver::BuildCompilation(ArrayRef<const char *> ArgList) {
  llvm::PrettyStackTraceString CrashInfo("Compilation construction");

  // FIXME: Handle environment options which affect driver behavior, somewhere
  // (client?). GCC_EXEC_PREFIX, LPATH, CC_PRINT_OPTIONS.

  // We look for the driver mode option early, because the mode can affect
  // how other options are parsed.

  auto DriverMode = getDriverMode(ClangExecutable, ArgList.slice(1));
  if (!DriverMode.empty())
    setDriverMode(DriverMode);

  // FIXME: What are we going to do with -V and -b?

  // Arguments specified in command line.
  bool ContainsError;
  CLOptions = std::make_unique<InputArgList>(
      ParseArgStrings(ArgList.slice(1), /*UseDriverMode=*/true, ContainsError));

  // Try parsing configuration file.
  if (!ContainsError)
    ContainsError = loadConfigFiles();
  bool HasConfigFile = !ContainsError && (CfgOptions.get() != nullptr);

  // All arguments, from both config file and command line.
  InputArgList Args = std::move(HasConfigFile ? std::move(*CfgOptions)
                                              : std::move(*CLOptions));

  if (HasConfigFile)
    for (auto *Opt : *CLOptions) {
      if (Opt->getOption().matches(options::OPT_config))
        continue;
      const Arg *BaseArg = &Opt->getBaseArg();
      if (BaseArg == Opt)
        BaseArg = nullptr;
      appendOneArg(Args, Opt, BaseArg);
    }

  // In CL mode, look for any pass-through arguments
  if (IsCLMode() && !ContainsError) {
    SmallVector<const char *, 16> CLModePassThroughArgList;
    for (const auto *A : Args.filtered(options::OPT__SLASH_clang)) {
      A->claim();
      CLModePassThroughArgList.push_back(A->getValue());
    }

    if (!CLModePassThroughArgList.empty()) {
      // Parse any pass through args using default clang processing rather
      // than clang-cl processing.
      auto CLModePassThroughOptions = std::make_unique<InputArgList>(
          ParseArgStrings(CLModePassThroughArgList, /*UseDriverMode=*/false,
                          ContainsError));

      if (!ContainsError)
        for (auto *Opt : *CLModePassThroughOptions) {
          appendOneArg(Args, Opt, nullptr);
        }
    }
  }

  if (Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false) &&
      CCCIsCC())
    setDriverMode("g++");

  // Check for working directory option before accessing any files
  if (Arg *WD = Args.getLastArg(options::OPT_working_directory))
    if (VFS->setCurrentWorkingDirectory(WD->getValue()))
      Diag(diag::err_drv_unable_to_set_working_directory) << WD->getValue();

  // FIXME: This stuff needs to go into the Compilation, not the driver.
  bool CCCPrintPhases;

  // -canonical-prefixes, -no-canonical-prefixes are used very early in main.
  Args.ClaimAllArgs(options::OPT_canonical_prefixes);
  Args.ClaimAllArgs(options::OPT_no_canonical_prefixes);

  // f(no-)integated-cc1 is also used very early in main.
  Args.ClaimAllArgs(options::OPT_fintegrated_cc1);
  Args.ClaimAllArgs(options::OPT_fno_integrated_cc1);

  // Ignore -pipe.
  Args.ClaimAllArgs(options::OPT_pipe);

  // Extract -ccc args.
  //
  // FIXME: We need to figure out where this behavior should live. Most of it
  // should be outside in the client; the parts that aren't should have proper
  // options, either by introducing new ones or by overloading gcc ones like -V
  // or -b.
  CCCPrintPhases = Args.hasArg(options::OPT_ccc_print_phases);
  CCCPrintBindings = Args.hasArg(options::OPT_ccc_print_bindings);
  if (const Arg *A = Args.getLastArg(options::OPT_ccc_gcc_name))
    CCCGenericGCCName = A->getValue();

  // Process -fproc-stat-report options.
  if (const Arg *A = Args.getLastArg(options::OPT_fproc_stat_report_EQ)) {
    CCPrintProcessStats = true;
    CCPrintStatReportFilename = A->getValue();
  }
  if (Args.hasArg(options::OPT_fproc_stat_report))
    CCPrintProcessStats = true;

  // FIXME: TargetTriple is used by the target-prefixed calls to as/ld
  // and getToolChain is const.
  if (IsCLMode()) {
    // clang-cl targets MSVC-style Win32.
    llvm::Triple T(TargetTriple);
    T.setOS(llvm::Triple::Win32);
    T.setVendor(llvm::Triple::PC);
    T.setEnvironment(llvm::Triple::MSVC);
    T.setObjectFormat(llvm::Triple::COFF);
    if (Args.hasArg(options::OPT__SLASH_arm64EC))
      T.setArch(llvm::Triple::aarch64, llvm::Triple::AArch64SubArch_arm64ec);
    TargetTriple = T.str();
  } else if (IsDXCMode()) {
    // Build TargetTriple from target_profile option for clang-dxc.
    if (const Arg *A = Args.getLastArg(options::OPT_target_profile)) {
      StringRef TargetProfile = A->getValue();
      if (auto Triple =
              toolchains::HLSLToolChain::parseTargetProfile(TargetProfile))
        TargetTriple = *Triple;
      else
        Diag(diag::err_drv_invalid_directx_shader_module) << TargetProfile;

      A->claim();

      if (Args.hasArg(options::OPT_spirv)) {
        llvm::Triple T(TargetTriple);
        T.setArch(llvm::Triple::spirv);
        T.setOS(llvm::Triple::Vulkan);

        // Set specific Vulkan version if applicable.
        if (const Arg *A = Args.getLastArg(options::OPT_fspv_target_env_EQ)) {
          const llvm::StringSet<> ValidValues = {"vulkan1.2", "vulkan1.3"};
          if (ValidValues.contains(A->getValue())) {
            T.setOSName(A->getValue());
          } else {
            Diag(diag::err_drv_invalid_value)
                << A->getAsString(Args) << A->getValue();
          }
          A->claim();
        }

        TargetTriple = T.str();
      }
    } else {
      Diag(diag::err_drv_dxc_missing_target_profile);
    }
  }

  if (const Arg *A = Args.getLastArg(options::OPT_target))
    TargetTriple = A->getValue();
  if (const Arg *A = Args.getLastArg(options::OPT_ccc_install_dir))
    Dir = InstalledDir = A->getValue();
  for (const Arg *A : Args.filtered(options::OPT_B)) {
    A->claim();
    PrefixDirs.push_back(A->getValue(0));
  }
  if (std::optional<std::string> CompilerPathValue =
          llvm::sys::Process::GetEnv("COMPILER_PATH")) {
    StringRef CompilerPath = *CompilerPathValue;
    while (!CompilerPath.empty()) {
      std::pair<StringRef, StringRef> Split =
          CompilerPath.split(llvm::sys::EnvPathSeparator);
      PrefixDirs.push_back(std::string(Split.first));
      CompilerPath = Split.second;
    }
  }
  if (const Arg *A = Args.getLastArg(options::OPT__sysroot_EQ))
    SysRoot = A->getValue();
  if (const Arg *A = Args.getLastArg(options::OPT__dyld_prefix_EQ))
    DyldPrefix = A->getValue();

  if (const Arg *A = Args.getLastArg(options::OPT_resource_dir))
    ResourceDir = A->getValue();

  if (const Arg *A = Args.getLastArg(options::OPT_save_temps_EQ)) {
    SaveTemps = llvm::StringSwitch<SaveTempsMode>(A->getValue())
                    .Case("cwd", SaveTempsCwd)
                    .Case("obj", SaveTempsObj)
                    .Default(SaveTempsCwd);
  }

  if (Args.getLastArg(options::OPT_fsycl_dump_device_code_EQ))
    DumpDeviceCode = true;

  if (const Arg *A = Args.getLastArg(options::OPT_offload_host_only,
                                     options::OPT_offload_device_only,
                                     options::OPT_offload_host_device)) {
    if (A->getOption().matches(options::OPT_offload_host_only))
      Offload = OffloadHost;
    else if (A->getOption().matches(options::OPT_offload_device_only))
      Offload = OffloadDevice;
    else
      Offload = OffloadHostDevice;
  }

  setLTOMode(Args);

  // Process -fembed-bitcode= flags.
  if (Arg *A = Args.getLastArg(options::OPT_fembed_bitcode_EQ)) {
    StringRef Name = A->getValue();
    unsigned Model = llvm::StringSwitch<unsigned>(Name)
        .Case("off", EmbedNone)
        .Case("all", EmbedBitcode)
        .Case("bitcode", EmbedBitcode)
        .Case("marker", EmbedMarker)
        .Default(~0U);
    if (Model == ~0U) {
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << Name;
    } else
      BitcodeEmbed = static_cast<BitcodeEmbedMode>(Model);
  }

  // Remove existing compilation database so that each job can append to it.
  if (Arg *A = Args.getLastArg(options::OPT_MJ))
    llvm::sys::fs::remove(A->getValue());

  // Setting up the jobs for some precompile cases depends on whether we are
  // treating them as PCH, implicit modules or C++20 ones.
  // TODO: inferring the mode like this seems fragile (it meets the objective
  // of not requiring anything new for operation, however).
  const Arg *Std = Args.getLastArg(options::OPT_std_EQ);
  ModulesModeCXX20 =
      !Args.hasArg(options::OPT_fmodules) && Std &&
      (Std->containsValue("c++20") || Std->containsValue("c++2a") ||
       Std->containsValue("c++23") || Std->containsValue("c++2b") ||
       Std->containsValue("c++26") || Std->containsValue("c++2c") ||
       Std->containsValue("c++latest"));

  // Process -fmodule-header{=} flags.
  if (Arg *A = Args.getLastArg(options::OPT_fmodule_header_EQ,
                               options::OPT_fmodule_header)) {
    // These flags force C++20 handling of headers.
    ModulesModeCXX20 = true;
    if (A->getOption().matches(options::OPT_fmodule_header))
      CXX20HeaderType = HeaderMode_Default;
    else {
      StringRef ArgName = A->getValue();
      unsigned Kind = llvm::StringSwitch<unsigned>(ArgName)
                          .Case("user", HeaderMode_User)
                          .Case("system", HeaderMode_System)
                          .Default(~0U);
      if (Kind == ~0U) {
        Diags.Report(diag::err_drv_invalid_value)
            << A->getAsString(Args) << ArgName;
      } else
        CXX20HeaderType = static_cast<ModuleHeaderMode>(Kind);
    }
  }

  std::unique_ptr<llvm::opt::InputArgList> UArgs =
      std::make_unique<InputArgList>(std::move(Args));

  // Perform the default argument translations.
  DerivedArgList *TranslatedArgs = TranslateInputArgs(*UArgs);

  // Owned by the host.
  const ToolChain &TC = getToolChain(
      *UArgs, computeTargetTriple(*this, TargetTriple, *UArgs));

  // Check if the environment version is valid except wasm case.
  llvm::Triple Triple = TC.getTriple();
  if (!Triple.isWasm()) {
    StringRef TripleVersionName = Triple.getEnvironmentVersionString();
    StringRef TripleObjectFormat =
        Triple.getObjectFormatTypeName(Triple.getObjectFormat());
    if (Triple.getEnvironmentVersion().empty() && TripleVersionName != "" &&
        TripleVersionName != TripleObjectFormat) {
      Diags.Report(diag::err_drv_triple_version_invalid)
          << TripleVersionName << TC.getTripleString();
      ContainsError = true;
    }
  }

  // Report warning when arm64EC option is overridden by specified target
  if ((TC.getTriple().getArch() != llvm::Triple::aarch64 ||
       TC.getTriple().getSubArch() != llvm::Triple::AArch64SubArch_arm64ec) &&
      UArgs->hasArg(options::OPT__SLASH_arm64EC)) {
    getDiags().Report(clang::diag::warn_target_override_arm64ec)
        << TC.getTriple().str();
  }

  // A common user mistake is specifying a target of aarch64-none-eabi or
  // arm-none-elf whereas the correct names are aarch64-none-elf &
  // arm-none-eabi. Detect these cases and issue a warning.
  if (TC.getTriple().getOS() == llvm::Triple::UnknownOS &&
      TC.getTriple().getVendor() == llvm::Triple::UnknownVendor) {
    switch (TC.getTriple().getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      if (TC.getTriple().getEnvironmentName() == "elf") {
        Diag(diag::warn_target_unrecognized_env)
            << TargetTriple
            << (TC.getTriple().getArchName().str() + "-none-eabi");
      }
      break;
    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_be:
    case llvm::Triple::aarch64_32:
      if (TC.getTriple().getEnvironmentName().starts_with("eabi")) {
        Diag(diag::warn_target_unrecognized_env)
            << TargetTriple
            << (TC.getTriple().getArchName().str() + "-none-elf");
      }
      break;
    default:
      break;
    }
  }

  // The compilation takes ownership of Args.
  Compilation *C = new Compilation(*this, TC, UArgs.release(), TranslatedArgs,
                                   ContainsError);

  if (!HandleImmediateArgs(*C))
    return C;

  // Construct the list of inputs.
  InputList Inputs;
  BuildInputs(C->getDefaultToolChain(), *TranslatedArgs, Inputs);

  // Determine if there are any offload static libraries.
  if (checkForOffloadStaticLib(*C, *TranslatedArgs))
    setOffloadStaticLibSeen();

  // Check for any objects/archives that need to be compiled with the default
  // triple.
  if (checkForSYCLDefaultDevice(*C, *TranslatedArgs))
    setSYCLDefaultTriple(true);

  // Populate the tool chains for the offloading devices, if any.
  CreateOffloadingDeviceToolChains(*C, Inputs);

  // Use new offloading path for OpenMP.  This is disabled as the SYCL
  // offloading path is not properly setup to use the updated device linking
  // scheme.
  if ((C->isOffloadingHostKind(Action::OFK_OpenMP) &&
       TranslatedArgs->hasFlag(options::OPT_fopenmp_new_driver,
                               options::OPT_no_offload_new_driver, true)) ||
      TranslatedArgs->hasFlag(options::OPT_offload_new_driver,
                              options::OPT_no_offload_new_driver, false))
    setUseNewOffloadingDriver();

  // Determine FPGA emulation status.
  if (C->hasOffloadToolChain<Action::OFK_SYCL>()) {
    auto SYCLTCRange = C->getOffloadToolChains<Action::OFK_SYCL>();
    for (auto TI = SYCLTCRange.first, TE = SYCLTCRange.second; TI != TE; ++TI) {
      if (TI->second->getTriple().getSubArch() !=
          llvm::Triple::SPIRSubArch_fpga)
        continue;
      ArgStringList TargetArgs;
      const toolchains::SYCLToolChain *FPGATC =
          static_cast<const toolchains::SYCLToolChain *>(TI->second);
      FPGATC->TranslateBackendTargetArgs(FPGATC->getTriple(), *TranslatedArgs,
                                         TargetArgs);
      // By default, FPGAEmulationMode is true due to the fact that
      // an external option setting is required to target hardware.
      setOffloadCompileMode(FPGAEmulationMode);
      for (StringRef ArgString : TargetArgs) {
        if (ArgString.equals("-hardware") || ArgString.equals("-simulation")) {
          setOffloadCompileMode(FPGAHWMode);
          break;
        }
      }
      break;
    }
  }

  // Construct the list of abstract actions to perform for this compilation. On
  // MachO targets this uses the driver-driver and universal actions.
  if (TC.getTriple().isOSBinFormatMachO())
    BuildUniversalActions(*C, C->getDefaultToolChain(), Inputs);
  else
    BuildActions(*C, C->getArgs(), Inputs, C->getActions());

  if (CCCPrintPhases) {
    PrintActions(*C);
    return C;
  }

  BuildJobs(*C);

  return C;
}

static void printArgList(raw_ostream &OS, const llvm::opt::ArgList &Args) {
  llvm::opt::ArgStringList ASL;
  for (const auto *A : Args) {
    // Use user's original spelling of flags. For example, use
    // `/source-charset:utf-8` instead of `-finput-charset=utf-8` if the user
    // wrote the former.
    while (A->getAlias())
      A = A->getAlias();
    A->render(Args, ASL);
  }

  for (auto I = ASL.begin(), E = ASL.end(); I != E; ++I) {
    if (I != ASL.begin())
      OS << ' ';
    llvm::sys::printArg(OS, *I, true);
  }
  OS << '\n';
}

bool Driver::getCrashDiagnosticFile(StringRef ReproCrashFilename,
                                    SmallString<128> &CrashDiagDir) {
  using namespace llvm::sys;
  assert(llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin() &&
         "Only knows about .crash files on Darwin");

  // The .crash file can be found on at ~/Library/Logs/DiagnosticReports/
  // (or /Library/Logs/DiagnosticReports for root) and has the filename pattern
  // clang-<VERSION>_<YYYY-MM-DD-HHMMSS>_<hostname>.crash.
  path::home_directory(CrashDiagDir);
  if (CrashDiagDir.starts_with("/var/root"))
    CrashDiagDir = "/";
  path::append(CrashDiagDir, "Library/Logs/DiagnosticReports");
  int PID =
#if LLVM_ON_UNIX
      getpid();
#else
      0;
#endif
  std::error_code EC;
  fs::file_status FileStatus;
  TimePoint<> LastAccessTime;
  SmallString<128> CrashFilePath;
  // Lookup the .crash files and get the one generated by a subprocess spawned
  // by this driver invocation.
  for (fs::directory_iterator File(CrashDiagDir, EC), FileEnd;
       File != FileEnd && !EC; File.increment(EC)) {
    StringRef FileName = path::filename(File->path());
    if (!FileName.starts_with(Name))
      continue;
    if (fs::status(File->path(), FileStatus))
      continue;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> CrashFile =
        llvm::MemoryBuffer::getFile(File->path());
    if (!CrashFile)
      continue;
    // The first line should start with "Process:", otherwise this isn't a real
    // .crash file.
    StringRef Data = CrashFile.get()->getBuffer();
    if (!Data.starts_with("Process:"))
      continue;
    // Parse parent process pid line, e.g: "Parent Process: clang-4.0 [79141]"
    size_t ParentProcPos = Data.find("Parent Process:");
    if (ParentProcPos == StringRef::npos)
      continue;
    size_t LineEnd = Data.find_first_of("\n", ParentProcPos);
    if (LineEnd == StringRef::npos)
      continue;
    StringRef ParentProcess = Data.slice(ParentProcPos+15, LineEnd).trim();
    int OpenBracket = -1, CloseBracket = -1;
    for (size_t i = 0, e = ParentProcess.size(); i < e; ++i) {
      if (ParentProcess[i] == '[')
        OpenBracket = i;
      if (ParentProcess[i] == ']')
        CloseBracket = i;
    }
    // Extract the parent process PID from the .crash file and check whether
    // it matches this driver invocation pid.
    int CrashPID;
    if (OpenBracket < 0 || CloseBracket < 0 ||
        ParentProcess.slice(OpenBracket + 1, CloseBracket)
            .getAsInteger(10, CrashPID) || CrashPID != PID) {
      continue;
    }

    // Found a .crash file matching the driver pid. To avoid getting an older
    // and misleading crash file, continue looking for the most recent.
    // FIXME: the driver can dispatch multiple cc1 invocations, leading to
    // multiple crashes poiting to the same parent process. Since the driver
    // does not collect pid information for the dispatched invocation there's
    // currently no way to distinguish among them.
    const auto FileAccessTime = FileStatus.getLastModificationTime();
    if (FileAccessTime > LastAccessTime) {
      CrashFilePath.assign(File->path());
      LastAccessTime = FileAccessTime;
    }
  }

  // If found, copy it over to the location of other reproducer files.
  if (!CrashFilePath.empty()) {
    EC = fs::copy_file(CrashFilePath, ReproCrashFilename);
    if (EC)
      return false;
    return true;
  }

  return false;
}

static const char BugReporMsg[] =
    "\n********************\n\n"
    "PLEASE ATTACH THE FOLLOWING FILES TO THE BUG REPORT:\n"
    "Preprocessed source(s) and associated run script(s) are located at:";

// When clang crashes, produce diagnostic information including the fully
// preprocessed source file(s).  Request that the developer attach the
// diagnostic information to a bug report.
void Driver::generateCompilationDiagnostics(
    Compilation &C, const Command &FailingCommand,
    StringRef AdditionalInformation, CompilationDiagnosticReport *Report) {
  if (C.getArgs().hasArg(options::OPT_fno_crash_diagnostics))
    return;

  unsigned Level = 1;
  if (Arg *A = C.getArgs().getLastArg(options::OPT_fcrash_diagnostics_EQ)) {
    Level = llvm::StringSwitch<unsigned>(A->getValue())
                .Case("off", 0)
                .Case("compiler", 1)
                .Case("all", 2)
                .Default(1);
  }
  if (!Level)
    return;

  // Don't try to generate diagnostics for dsymutil jobs.
  if (FailingCommand.getCreator().isDsymutilJob())
    return;

  bool IsLLD = false;
  TempFileList SavedTemps;
  if (FailingCommand.getCreator().isLinkJob()) {
    C.getDefaultToolChain().GetLinkerPath(&IsLLD);
    if (!IsLLD || Level < 2)
      return;

    // If lld crashed, we will re-run the same command with the input it used
    // to have. In that case we should not remove temp files in
    // initCompilationForDiagnostics yet. They will be added back and removed
    // later.
    SavedTemps = std::move(C.getTempFiles());
    assert(!C.getTempFiles().size());
  }

  // Print the version of the compiler.
  PrintVersion(C, llvm::errs());

  // Suppress driver output and emit preprocessor output to temp file.
  CCGenDiagnostics = true;

  // Save the original job command(s).
  Command Cmd = FailingCommand;

  // Keep track of whether we produce any errors while trying to produce
  // preprocessed sources.
  DiagnosticErrorTrap Trap(Diags);

  // Suppress tool output.
  C.initCompilationForDiagnostics();

  // If lld failed, rerun it again with --reproduce.
  if (IsLLD) {
    const char *TmpName = CreateTempFile(C, "linker-crash", "tar");
    Command NewLLDInvocation = Cmd;
    llvm::opt::ArgStringList ArgList = NewLLDInvocation.getArguments();
    StringRef ReproduceOption =
        C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment()
            ? "/reproduce:"
            : "--reproduce=";
    ArgList.push_back(Saver.save(Twine(ReproduceOption) + TmpName).data());
    NewLLDInvocation.replaceArguments(std::move(ArgList));

    // Redirect stdout/stderr to /dev/null.
    NewLLDInvocation.Execute({std::nullopt, {""}, {""}}, nullptr, nullptr);
    Diag(clang::diag::note_drv_command_failed_diag_msg) << BugReporMsg;
    Diag(clang::diag::note_drv_command_failed_diag_msg) << TmpName;
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "\n\n********************";
    if (Report)
      Report->TemporaryFiles.push_back(TmpName);
    return;
  }

  // Construct the list of inputs.
  InputList Inputs;
  BuildInputs(C.getDefaultToolChain(), C.getArgs(), Inputs);

  for (InputList::iterator it = Inputs.begin(), ie = Inputs.end(); it != ie;) {
    bool IgnoreInput = false;

    // Ignore input from stdin or any inputs that cannot be preprocessed.
    // Check type first as not all linker inputs have a value.
    if (types::getPreprocessedType(it->first) == types::TY_INVALID) {
      IgnoreInput = true;
    } else if (!strcmp(it->second->getValue(), "-")) {
      Diag(clang::diag::note_drv_command_failed_diag_msg)
          << "Error generating preprocessed source(s) - "
             "ignoring input from stdin.";
      IgnoreInput = true;
    }

    if (IgnoreInput) {
      it = Inputs.erase(it);
      ie = Inputs.end();
    } else {
      ++it;
    }
  }

  if (Inputs.empty()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s) - "
           "no preprocessable inputs.";
    return;
  }

  // Don't attempt to generate preprocessed files if multiple -arch options are
  // used, unless they're all duplicates.
  llvm::StringSet<> ArchNames;
  for (const Arg *A : C.getArgs()) {
    if (A->getOption().matches(options::OPT_arch)) {
      StringRef ArchName = A->getValue();
      ArchNames.insert(ArchName);
    }
  }
  if (ArchNames.size() > 1) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s) - cannot generate "
           "preprocessed source with multiple -arch options.";
    return;
  }

  // Construct the list of abstract actions to perform for this compilation. On
  // Darwin OSes this uses the driver-driver and builds universal actions.
  const ToolChain &TC = C.getDefaultToolChain();
  if (TC.getTriple().isOSBinFormatMachO())
    BuildUniversalActions(C, TC, Inputs);
  else
    BuildActions(C, C.getArgs(), Inputs, C.getActions());

  BuildJobs(C);

  // If there were errors building the compilation, quit now.
  if (Trap.hasErrorOccurred()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s).";
    return;
  }

  // Generate preprocessed output.
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  C.ExecuteJobs(C.getJobs(), FailingCommands);

  // If any of the preprocessing commands failed, clean up and exit.
  if (!FailingCommands.empty()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s).";
    return;
  }

  const TempFileList &TempFiles = C.getTempFiles();
  if (TempFiles.empty()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s).";
    return;
  }

  Diag(clang::diag::note_drv_command_failed_diag_msg) << BugReporMsg;

  SmallString<128> VFS;
  SmallString<128> ReproCrashFilename;
  for (auto &TempFile : TempFiles) {
    Diag(clang::diag::note_drv_command_failed_diag_msg) << TempFile.first;
    if (Report)
      Report->TemporaryFiles.push_back(TempFile.first);
    if (ReproCrashFilename.empty()) {
      ReproCrashFilename = TempFile.first;
      llvm::sys::path::replace_extension(ReproCrashFilename, ".crash");
    }
    if (StringRef(TempFile.first).ends_with(".cache")) {
      // In some cases (modules) we'll dump extra data to help with reproducing
      // the crash into a directory next to the output.
      VFS = llvm::sys::path::filename(TempFile.first);
      llvm::sys::path::append(VFS, "vfs", "vfs.yaml");
    }
  }

  for (auto &TempFile : SavedTemps)
    C.addTempFile(TempFile.first);

  // Assume associated files are based off of the first temporary file.
  CrashReportInfo CrashInfo(TempFiles[0].first, VFS);

  llvm::SmallString<128> Script(CrashInfo.Filename);
  llvm::sys::path::replace_extension(Script, "sh");
  std::error_code EC;
  llvm::raw_fd_ostream ScriptOS(Script, EC, llvm::sys::fs::CD_CreateNew,
                                llvm::sys::fs::FA_Write,
                                llvm::sys::fs::OF_Text);
  if (EC) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating run script: " << Script << " " << EC.message();
  } else {
    ScriptOS << "# Crash reproducer for " << getClangFullVersion() << "\n"
             << "# Driver args: ";
    printArgList(ScriptOS, C.getInputArgs());
    ScriptOS << "# Original command: ";
    Cmd.Print(ScriptOS, "\n", /*Quote=*/true);
    Cmd.Print(ScriptOS, "\n", /*Quote=*/true, &CrashInfo);
    if (!AdditionalInformation.empty())
      ScriptOS << "\n# Additional information: " << AdditionalInformation
               << "\n";
    if (Report)
      Report->TemporaryFiles.push_back(std::string(Script));
    Diag(clang::diag::note_drv_command_failed_diag_msg) << Script;
  }

  // On darwin, provide information about the .crash diagnostic report.
  if (llvm::Triple(llvm::sys::getProcessTriple()).isOSDarwin()) {
    SmallString<128> CrashDiagDir;
    if (getCrashDiagnosticFile(ReproCrashFilename, CrashDiagDir)) {
      Diag(clang::diag::note_drv_command_failed_diag_msg)
          << ReproCrashFilename.str();
    } else { // Suggest a directory for the user to look for .crash files.
      llvm::sys::path::append(CrashDiagDir, Name);
      CrashDiagDir += "_<YYYY-MM-DD-HHMMSS>_<hostname>.crash";
      Diag(clang::diag::note_drv_command_failed_diag_msg)
          << "Crash backtrace is located in";
      Diag(clang::diag::note_drv_command_failed_diag_msg)
          << CrashDiagDir.str();
      Diag(clang::diag::note_drv_command_failed_diag_msg)
          << "(choose the .crash file that corresponds to your crash)";
    }
  }

  Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "\n\n********************";
}

void Driver::setUpResponseFiles(Compilation &C, Command &Cmd) {
  // Since commandLineFitsWithinSystemLimits() may underestimate system's
  // capacity if the tool does not support response files, there is a chance/
  // that things will just work without a response file, so we silently just
  // skip it.
  if (Cmd.getResponseFileSupport().ResponseKind ==
          ResponseFileSupport::RF_None ||
      llvm::sys::commandLineFitsWithinSystemLimits(Cmd.getExecutable(),
                                                   Cmd.getArguments()))
    return;

  std::string TmpName = GetTemporaryPath("response", "txt");
  Cmd.setResponseFile(C.addTempFile(C.getArgs().MakeArgString(TmpName)));
}

int Driver::ExecuteCompilation(
    Compilation &C,
    SmallVectorImpl<std::pair<int, const Command *>> &FailingCommands) {
  if (C.getArgs().hasArg(options::OPT_fdriver_only)) {
    if (C.getArgs().hasArg(options::OPT_v))
      C.getJobs().Print(llvm::errs(), "\n", true);

    C.ExecuteJobs(C.getJobs(), FailingCommands, /*LogOnly=*/true);

    // If there were errors building the compilation, quit now.
    if (!FailingCommands.empty() || Diags.hasErrorOccurred())
      return 1;

    return 0;
  }

  // Just print if -### was present.
  if (C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    C.getJobs().Print(llvm::errs(), "\n", true);
    return Diags.hasErrorOccurred() ? 1 : 0;
  }

  // If there were errors building the compilation, quit now.
  if (Diags.hasErrorOccurred())
    return 1;

  // Set up response file names for each command, if necessary.
  for (auto &Job : C.getJobs())
    setUpResponseFiles(C, Job);

  C.ExecuteJobs(C.getJobs(), FailingCommands);

  // If the command succeeded, we are done.
  if (FailingCommands.empty())
    return 0;

  // Otherwise, remove result files and print extra information about abnormal
  // failures.
  int Res = 0;
  for (const auto &CmdPair : FailingCommands) {
    int CommandRes = CmdPair.first;
    const Command *FailingCommand = CmdPair.second;

    // Remove result files if we're not saving temps.
    if (!isSaveTempsEnabled()) {
      const JobAction *JA = cast<JobAction>(&FailingCommand->getSource());
      C.CleanupFileMap(C.getResultFiles(), JA, true);

      // Failure result files are valid unless we crashed.
      if (CommandRes < 0)
        C.CleanupFileMap(C.getFailureResultFiles(), JA, true);
    }

    // llvm/lib/Support/*/Signals.inc will exit with a special return code
    // for SIGPIPE. Do not print diagnostics for this case.
    if (CommandRes == EX_IOERR) {
      Res = CommandRes;
      continue;
    }

    // Print extra information about abnormal failures, if possible.
    //
    // This is ad-hoc, but we don't want to be excessively noisy. If the result
    // status was 1, assume the command failed normally. In particular, if it
    // was the compiler then assume it gave a reasonable error code. Failures
    // in other tools are less common, and they generally have worse
    // diagnostics, so always print the diagnostic there.
    const Tool &FailingTool = FailingCommand->getCreator();

    if (!FailingTool.hasGoodDiagnostics() || CommandRes != 1) {
      // FIXME: See FIXME above regarding result code interpretation.
      if (CommandRes < 0)
        Diag(clang::diag::err_drv_command_signalled)
            << FailingTool.getShortName();
      else
        Diag(clang::diag::err_drv_command_failed)
            << FailingTool.getShortName() << CommandRes;
    }

    auto CustomDiag = FailingCommand->getDiagForErrorCode(CommandRes);
    if (!CustomDiag.empty())
      Diag(clang::diag::note_drv_command_failed_diag_msg) << CustomDiag;
  }
  return Res;
}

void Driver::PrintHelp(bool ShowHidden) const {
  llvm::opt::Visibility VisibilityMask = getOptionVisibilityMask();

  std::string Usage = llvm::formatv("{0} [options] file...", Name).str();
  getOpts().printHelp(llvm::outs(), Usage.c_str(), DriverTitle.c_str(),
                      ShowHidden, /*ShowAllAliases=*/false,
                      VisibilityMask);
}

llvm::Triple Driver::MakeSYCLDeviceTriple(StringRef TargetArch) const {
  SmallVector<StringRef, 5> SYCLAlias = {"spir", "spir64", "spir64_fpga",
                                         "spir64_x86_64", "spir64_gen"};
  if (std::find(SYCLAlias.begin(), SYCLAlias.end(), TargetArch) !=
      SYCLAlias.end()) {
    llvm::Triple TT;
    TT.setArchName(TargetArch);
    TT.setVendor(llvm::Triple::UnknownVendor);
    TT.setOS(llvm::Triple::UnknownOS);
    return TT;
  }
  return llvm::Triple(TargetArch);
}

// Print the help from any of the given tools which are used for AOT
// compilation for SYCL
void Driver::PrintSYCLToolHelp(const Compilation &C) const {
  SmallVector<std::tuple<llvm::Triple, StringRef, StringRef, StringRef>, 4>
      HelpArgs;
  // Populate the vector with the tools and help options
  if (Arg *A = C.getArgs().getLastArg(options::OPT_fsycl_help_EQ)) {
    StringRef AV(A->getValue());
    llvm::Triple T;
    if (AV == "gen" || AV == "all")
      HelpArgs.push_back(std::make_tuple(MakeSYCLDeviceTriple("spir64_gen"),
                                         "ocloc", "--help", ""));
    if (AV == "fpga" || AV == "all")
      HelpArgs.push_back(std::make_tuple(MakeSYCLDeviceTriple("spir64_fpga"),
                                         "aoc", "-help", "-sycl"));
    if (AV == "x86_64" || AV == "all")
      HelpArgs.push_back(std::make_tuple(MakeSYCLDeviceTriple("spir64_x86_64"),
                                         "opencl-aot", "--help", ""));
    if (HelpArgs.empty()) {
      C.getDriver().Diag(diag::err_drv_unsupported_option_argument)
                         << A->getSpelling() << AV;
      return;
    }
  }

  // Go through the args and emit the help information for each.
  for (auto &HA : HelpArgs) {
    llvm::outs() << "Emitting help information for " << std::get<1>(HA) << '\n'
        << "Use triple of '" << std::get<0>(HA).normalize() <<
        "' to enable ahead of time compilation\n";
    // Flush out the buffer before calling the external tool.
    llvm::outs().flush();
    std::vector<StringRef> ToolArgs = {std::get<1>(HA), std::get<2>(HA),
                                       std::get<3>(HA)};
    SmallString<128> ExecPath(
        C.getDefaultToolChain().GetProgramPath(std::get<1>(HA).data()));
    // do not run the tools with -###.
    if (C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
      llvm::errs() << "\"" << ExecPath << "\" \"" << ToolArgs[1] << "\"";
      if (!ToolArgs[2].empty())
        llvm::errs() << " \"" << ToolArgs[2] << "\"";
      llvm::errs() << "\n";
      continue;
    }
    auto ToolBinary = llvm::sys::findProgramByName(ExecPath);
    if (ToolBinary.getError()) {
      C.getDriver().Diag(diag::err_drv_command_failure) << ExecPath;
      continue;
    }
    // Run the Tool.
    llvm::sys::ExecuteAndWait(ToolBinary.get(), ToolArgs);
  }
}

void Driver::PrintVersion(const Compilation &C, raw_ostream &OS) const {
  if (IsFlangMode()) {
    OS << getClangToolFullVersion("flang-new") << '\n';
  } else {
    // FIXME: The following handlers should use a callback mechanism, we don't
    // know what the client would like to do.
    OS << getClangFullVersion() << '\n';
  }
  const ToolChain &TC = C.getDefaultToolChain();
  OS << "Target: " << TC.getTripleString() << '\n';

  // Print the threading model.
  if (Arg *A = C.getArgs().getLastArg(options::OPT_mthread_model)) {
    // Don't print if the ToolChain would have barfed on it already
    if (TC.isThreadModelSupported(A->getValue()))
      OS << "Thread model: " << A->getValue();
  } else
    OS << "Thread model: " << TC.getThreadModel();
  OS << '\n';

  // Print out the install directory.
  OS << "InstalledDir: " << InstalledDir << '\n';

  // If configuration files were used, print their paths.
  for (auto ConfigFile : ConfigFiles)
    OS << "Configuration file: " << ConfigFile << '\n';
}

/// PrintDiagnosticCategories - Implement the --print-diagnostic-categories
/// option.
static void PrintDiagnosticCategories(raw_ostream &OS) {
  // Skip the empty category.
  for (unsigned i = 1, max = DiagnosticIDs::getNumberOfCategories(); i != max;
       ++i)
    OS << i << ',' << DiagnosticIDs::getCategoryNameFromID(i) << '\n';
}

void Driver::HandleAutocompletions(StringRef PassedFlags) const {
  if (PassedFlags == "")
    return;
  // Print out all options that start with a given argument. This is used for
  // shell autocompletion.
  std::vector<std::string> SuggestedCompletions;
  std::vector<std::string> Flags;

  llvm::opt::Visibility VisibilityMask(options::ClangOption);

  // Make sure that Flang-only options don't pollute the Clang output
  // TODO: Make sure that Clang-only options don't pollute Flang output
  if (IsFlangMode())
    VisibilityMask = llvm::opt::Visibility(options::FlangOption);

  // Distinguish "--autocomplete=-someflag" and "--autocomplete=-someflag,"
  // because the latter indicates that the user put space before pushing tab
  // which should end up in a file completion.
  const bool HasSpace = PassedFlags.ends_with(",");

  // Parse PassedFlags by "," as all the command-line flags are passed to this
  // function separated by ","
  StringRef TargetFlags = PassedFlags;
  while (TargetFlags != "") {
    StringRef CurFlag;
    std::tie(CurFlag, TargetFlags) = TargetFlags.split(",");
    Flags.push_back(std::string(CurFlag));
  }

  // We want to show cc1-only options only when clang is invoked with -cc1 or
  // -Xclang.
  if (llvm::is_contained(Flags, "-Xclang") || llvm::is_contained(Flags, "-cc1"))
    VisibilityMask = llvm::opt::Visibility(options::CC1Option);

  const llvm::opt::OptTable &Opts = getOpts();
  StringRef Cur;
  Cur = Flags.at(Flags.size() - 1);
  StringRef Prev;
  if (Flags.size() >= 2) {
    Prev = Flags.at(Flags.size() - 2);
    SuggestedCompletions = Opts.suggestValueCompletions(Prev, Cur);
  }

  if (SuggestedCompletions.empty())
    SuggestedCompletions = Opts.suggestValueCompletions(Cur, "");

  // If Flags were empty, it means the user typed `clang [tab]` where we should
  // list all possible flags. If there was no value completion and the user
  // pressed tab after a space, we should fall back to a file completion.
  // We're printing a newline to be consistent with what we print at the end of
  // this function.
  if (SuggestedCompletions.empty() && HasSpace && !Flags.empty()) {
    llvm::outs() << '\n';
    return;
  }

  // When flag ends with '=' and there was no value completion, return empty
  // string and fall back to the file autocompletion.
  if (SuggestedCompletions.empty() && !Cur.ends_with("=")) {
    // If the flag is in the form of "--autocomplete=-foo",
    // we were requested to print out all option names that start with "-foo".
    // For example, "--autocomplete=-fsyn" is expanded to "-fsyntax-only".
    SuggestedCompletions = Opts.findByPrefix(
        Cur, VisibilityMask,
        /*DisableFlags=*/options::Unsupported | options::Ignored);

    // We have to query the -W flags manually as they're not in the OptTable.
    // TODO: Find a good way to add them to OptTable instead and them remove
    // this code.
    for (StringRef S : DiagnosticIDs::getDiagnosticFlags())
      if (S.starts_with(Cur))
        SuggestedCompletions.push_back(std::string(S));
  }

  // Sort the autocomplete candidates so that shells print them out in a
  // deterministic order. We could sort in any way, but we chose
  // case-insensitive sorting for consistency with the -help option
  // which prints out options in the case-insensitive alphabetical order.
  llvm::sort(SuggestedCompletions, [](StringRef A, StringRef B) {
    if (int X = A.compare_insensitive(B))
      return X < 0;
    return A.compare(B) > 0;
  });

  llvm::outs() << llvm::join(SuggestedCompletions, "\n") << '\n';
}

bool Driver::HandleImmediateArgs(const Compilation &C) {
  // The order these options are handled in gcc is all over the place, but we
  // don't expect inconsistencies w.r.t. that to matter in practice.

  if (C.getArgs().hasArg(options::OPT_dumpmachine)) {
    llvm::outs() << C.getDefaultToolChain().getTripleString() << '\n';
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_dumpversion)) {
    // Since -dumpversion is only implemented for pedantic GCC compatibility, we
    // return an answer which matches our definition of __VERSION__.
    llvm::outs() << CLANG_VERSION_STRING << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT__print_diagnostic_categories)) {
    PrintDiagnosticCategories(llvm::outs());
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_help) ||
      C.getArgs().hasArg(options::OPT__help_hidden)) {
    PrintHelp(C.getArgs().hasArg(options::OPT__help_hidden));
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_fsycl_help_EQ)) {
    PrintSYCLToolHelp(C);
    return false;
  }

  if (C.getArgs().hasArg(options::OPT__version)) {
    // Follow gcc behavior and use stdout for --version and stderr for -v.
    PrintVersion(C, llvm::outs());
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_v) ||
      C.getArgs().hasArg(options::OPT__HASH_HASH_HASH) ||
      C.getArgs().hasArg(options::OPT_print_supported_cpus) ||
      C.getArgs().hasArg(options::OPT_print_supported_extensions)) {
    PrintVersion(C, llvm::errs());
    SuppressMissingInputWarning = true;
  }

  if (C.getArgs().hasArg(options::OPT_v)) {
    if (!SystemConfigDir.empty())
      llvm::errs() << "System configuration file directory: "
                   << SystemConfigDir << "\n";
    if (!UserConfigDir.empty())
      llvm::errs() << "User configuration file directory: "
                   << UserConfigDir << "\n";
  }

  const ToolChain &TC = C.getDefaultToolChain();

  if (C.getArgs().hasArg(options::OPT_v))
    TC.printVerboseInfo(llvm::errs());

  if (C.getArgs().hasArg(options::OPT_print_resource_dir)) {
    llvm::outs() << ResourceDir << '\n';
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_search_dirs)) {
    llvm::outs() << "programs: =";
    bool separator = false;
    // Print -B and COMPILER_PATH.
    for (const std::string &Path : PrefixDirs) {
      if (separator)
        llvm::outs() << llvm::sys::EnvPathSeparator;
      llvm::outs() << Path;
      separator = true;
    }
    for (const std::string &Path : TC.getProgramPaths()) {
      if (separator)
        llvm::outs() << llvm::sys::EnvPathSeparator;
      llvm::outs() << Path;
      separator = true;
    }
    llvm::outs() << "\n";
    llvm::outs() << "libraries: =" << ResourceDir;

    StringRef sysroot = C.getSysRoot();

    for (const std::string &Path : TC.getFilePaths()) {
      // Always print a separator. ResourceDir was the first item shown.
      llvm::outs() << llvm::sys::EnvPathSeparator;
      // Interpretation of leading '=' is needed only for NetBSD.
      if (Path[0] == '=')
        llvm::outs() << sysroot << Path.substr(1);
      else
        llvm::outs() << Path;
    }
    llvm::outs() << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_runtime_dir)) {
    if (std::optional<std::string> RuntimePath = TC.getRuntimePath())
      llvm::outs() << *RuntimePath << '\n';
    else
      llvm::outs() << TC.getCompilerRTPath() << '\n';
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_diagnostic_options)) {
    std::vector<std::string> Flags = DiagnosticIDs::getDiagnosticFlags();
    for (std::size_t I = 0; I != Flags.size(); I += 2)
      llvm::outs() << "  " << Flags[I] << "\n  " << Flags[I + 1] << "\n\n";
    return false;
  }

  // FIXME: The following handlers should use a callback mechanism, we don't
  // know what the client would like to do.
  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_file_name_EQ)) {
    llvm::outs() << GetFilePath(A->getValue(), TC) << "\n";
    return false;
  }

  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_prog_name_EQ)) {
    StringRef ProgName = A->getValue();

    // Null program name cannot have a path.
    if (! ProgName.empty())
      llvm::outs() << GetProgramPath(ProgName, TC);

    llvm::outs() << "\n";
    return false;
  }

  if (Arg *A = C.getArgs().getLastArg(options::OPT_autocomplete)) {
    StringRef PassedFlags = A->getValue();
    HandleAutocompletions(PassedFlags);
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_libgcc_file_name)) {
    ToolChain::RuntimeLibType RLT = TC.GetRuntimeLibType(C.getArgs());
    const llvm::Triple Triple(TC.ComputeEffectiveClangTriple(C.getArgs()));
    RegisterEffectiveTriple TripleRAII(TC, Triple);
    switch (RLT) {
    case ToolChain::RLT_CompilerRT:
      llvm::outs() << TC.getCompilerRT(C.getArgs(), "builtins") << "\n";
      break;
    case ToolChain::RLT_Libgcc:
      llvm::outs() << GetFilePath("libgcc.a", TC) << "\n";
      break;
    }
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_multi_lib)) {
    for (const Multilib &Multilib : TC.getMultilibs())
      llvm::outs() << Multilib << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_multi_flags)) {
    Multilib::flags_list ArgFlags = TC.getMultilibFlags(C.getArgs());
    llvm::StringSet<> ExpandedFlags = TC.getMultilibs().expandFlags(ArgFlags);
    std::set<llvm::StringRef> SortedFlags;
    for (const auto &FlagEntry : ExpandedFlags)
      SortedFlags.insert(FlagEntry.getKey());
    for (auto Flag : SortedFlags)
      llvm::outs() << Flag << '\n';
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_multi_directory)) {
    for (const Multilib &Multilib : TC.getSelectedMultilibs()) {
      if (Multilib.gccSuffix().empty())
        llvm::outs() << ".\n";
      else {
        StringRef Suffix(Multilib.gccSuffix());
        assert(Suffix.front() == '/');
        llvm::outs() << Suffix.substr(1) << "\n";
      }
    }
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_target_triple)) {
    llvm::outs() << TC.getTripleString() << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_effective_triple)) {
    const llvm::Triple Triple(TC.ComputeEffectiveClangTriple(C.getArgs()));
    llvm::outs() << Triple.getTriple() << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_targets)) {
    llvm::TargetRegistry::printRegisteredTargetsForVersion(llvm::outs());
    return false;
  }

  return true;
}

enum {
  TopLevelAction = 0,
  HeadSibAction = 1,
  OtherSibAction = 2,
};

// Display an action graph human-readably.  Action A is the "sink" node
// and latest-occuring action. Traversal is in pre-order, visiting the
// inputs to each action before printing the action itself.
static unsigned PrintActions1(const Compilation &C, Action *A,
                              std::map<Action *, unsigned> &Ids,
                              Twine Indent = {}, int Kind = TopLevelAction) {
  if (Ids.count(A)) // A was already visited.
    return Ids[A];

  std::string str;
  llvm::raw_string_ostream os(str);

  auto getSibIndent = [](int K) -> Twine {
    return (K == HeadSibAction) ? "   " : (K == OtherSibAction) ? "|  " : "";
  };

  Twine SibIndent = Indent + getSibIndent(Kind);
  int SibKind = HeadSibAction;
  os << Action::getClassName(A->getKind()) << ", ";
  if (InputAction *IA = dyn_cast<InputAction>(A)) {
    os << "\"" << IA->getInputArg().getValue() << "\"";
  } else if (BindArchAction *BIA = dyn_cast<BindArchAction>(A)) {
    os << '"' << BIA->getArchName() << '"' << ", {"
       << PrintActions1(C, *BIA->input_begin(), Ids, SibIndent, SibKind) << "}";
  } else if (OffloadAction *OA = dyn_cast<OffloadAction>(A)) {
    bool IsFirst = true;
    OA->doOnEachDependence(
        [&](Action *A, const ToolChain *TC, const char *BoundArch) {
          assert(TC && "Unknown host toolchain");
          // E.g. for two CUDA device dependences whose bound arch is sm_20 and
          // sm_35 this will generate:
          // "cuda-device" (nvptx64-nvidia-cuda:sm_20) {#ID}, "cuda-device"
          // (nvptx64-nvidia-cuda:sm_35) {#ID}
          if (!IsFirst)
            os << ", ";
          os << '"';
          os << A->getOffloadingKindPrefix();
          os << " (";
          os << TC->getTriple().normalize();
          if (BoundArch)
            os << ":" << BoundArch;
          os << ")";
          os << '"';
          os << " {" << PrintActions1(C, A, Ids, SibIndent, SibKind) << "}";
          IsFirst = false;
          SibKind = OtherSibAction;
        });
  } else {
    const ActionList *AL = &A->getInputs();

    if (AL->size()) {
      const char *Prefix = "{";
      for (Action *PreRequisite : *AL) {
        os << Prefix << PrintActions1(C, PreRequisite, Ids, SibIndent, SibKind);
        Prefix = ", ";
        SibKind = OtherSibAction;
      }
      os << "}";
    } else
      os << "{}";
  }

  // Append offload info for all options other than the offloading action
  // itself (e.g. (cuda-device, sm_20) or (cuda-host)).
  std::string offload_str;
  llvm::raw_string_ostream offload_os(offload_str);
  if (!isa<OffloadAction>(A)) {
    auto S = A->getOffloadingKindPrefix();
    if (!S.empty()) {
      offload_os << ", (" << S;
      if (A->getOffloadingArch())
        offload_os << ", " << A->getOffloadingArch();
      offload_os << ")";
    }
  }

  auto getSelfIndent = [](int K) -> Twine {
    return (K == HeadSibAction) ? "+- " : (K == OtherSibAction) ? "|- " : "";
  };

  unsigned Id = Ids.size();
  Ids[A] = Id;
  llvm::errs() << Indent + getSelfIndent(Kind) << Id << ": " << os.str() << ", "
               << types::getTypeName(A->getType()) << offload_os.str() << "\n";

  return Id;
}

// Print the action graphs in a compilation C.
// For example "clang -c file1.c file2.c" is composed of two subgraphs.
void Driver::PrintActions(const Compilation &C) const {
  std::map<Action *, unsigned> Ids;
  for (Action *A : C.getActions())
    PrintActions1(C, A, Ids);
}

/// Check whether the given input tree contains any compilation or
/// assembly actions.
static bool ContainsCompileOrAssembleAction(const Action *A) {
  if (isa<CompileJobAction>(A) || isa<BackendJobAction>(A) ||
      isa<AssembleJobAction>(A))
    return true;

  return llvm::any_of(A->inputs(), ContainsCompileOrAssembleAction);
}

void Driver::BuildUniversalActions(Compilation &C, const ToolChain &TC,
                                   const InputList &BAInputs) const {
  DerivedArgList &Args = C.getArgs();
  ActionList &Actions = C.getActions();
  llvm::PrettyStackTraceString CrashInfo("Building universal build actions");
  // Collect the list of architectures. Duplicates are allowed, but should only
  // be handled once (in the order seen).
  llvm::StringSet<> ArchNames;
  SmallVector<const char *, 4> Archs;
  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_arch)) {
      // Validate the option here; we don't save the type here because its
      // particular spelling may participate in other driver choices.
      llvm::Triple::ArchType Arch =
          tools::darwin::getArchTypeForMachOArchName(A->getValue());
      if (Arch == llvm::Triple::UnknownArch) {
        Diag(clang::diag::err_drv_invalid_arch_name) << A->getAsString(Args);
        continue;
      }

      A->claim();
      if (ArchNames.insert(A->getValue()).second)
        Archs.push_back(A->getValue());
    }
  }

  // When there is no explicit arch for this platform, make sure we still bind
  // the architecture (to the default) so that -Xarch_ is handled correctly.
  if (!Archs.size())
    Archs.push_back(Args.MakeArgString(TC.getDefaultUniversalArchName()));

  ActionList SingleActions;
  BuildActions(C, Args, BAInputs, SingleActions);

  // Add in arch bindings for every top level action, as well as lipo and
  // dsymutil steps if needed.
  for (Action* Act : SingleActions) {
    // Make sure we can lipo this kind of output. If not (and it is an actual
    // output) then we disallow, since we can't create an output file with the
    // right name without overwriting it. We could remove this oddity by just
    // changing the output names to include the arch, which would also fix
    // -save-temps. Compatibility wins for now.

    if (Archs.size() > 1 && !types::canLipoType(Act->getType()))
      Diag(clang::diag::err_drv_invalid_output_with_multiple_archs)
          << types::getTypeName(Act->getType());

    ActionList Inputs;
    for (unsigned i = 0, e = Archs.size(); i != e; ++i)
      Inputs.push_back(C.MakeAction<BindArchAction>(Act, Archs[i]));

    // Lipo if necessary, we do it this way because we need to set the arch flag
    // so that -Xarch_ gets overwritten.
    if (Inputs.size() == 1 || Act->getType() == types::TY_Nothing)
      Actions.append(Inputs.begin(), Inputs.end());
    else
      Actions.push_back(C.MakeAction<LipoJobAction>(Inputs, Act->getType()));

    // Handle debug info queries.
    Arg *A = Args.getLastArg(options::OPT_g_Group);
    bool enablesDebugInfo = A && !A->getOption().matches(options::OPT_g0) &&
                            !A->getOption().matches(options::OPT_gstabs);
    if ((enablesDebugInfo || willEmitRemarks(Args)) &&
        ContainsCompileOrAssembleAction(Actions.back())) {

      // Add a 'dsymutil' step if necessary, when debug info is enabled and we
      // have a compile input. We need to run 'dsymutil' ourselves in such cases
      // because the debug info will refer to a temporary object file which
      // will be removed at the end of the compilation process.
      if (Act->getType() == types::TY_Image) {
        ActionList Inputs;
        Inputs.push_back(Actions.back());
        Actions.pop_back();
        Actions.push_back(
            C.MakeAction<DsymutilJobAction>(Inputs, types::TY_dSYM));
      }

      // Verify the debug info output.
      if (Args.hasArg(options::OPT_verify_debug_info)) {
        Action* LastAction = Actions.back();
        Actions.pop_back();
        Actions.push_back(C.MakeAction<VerifyDebugInfoJobAction>(
            LastAction, types::TY_Nothing));
      }
    }
  }
}

bool Driver::DiagnoseInputExistence(const DerivedArgList &Args, StringRef Value,
                                    types::ID Ty, bool TypoCorrect) const {
  if (!getCheckInputsExist())
    return true;

  // stdin always exists.
  if (Value == "-")
    return true;

  // If it's a header to be found in the system or user search path, then defer
  // complaints about its absence until those searches can be done.  When we
  // are definitely processing headers for C++20 header units, extend this to
  // allow the user to put "-fmodule-header -xc++-header vector" for example.
  if (Ty == types::TY_CXXSHeader || Ty == types::TY_CXXUHeader ||
      (ModulesModeCXX20 && Ty == types::TY_CXXHeader))
    return true;

  if (getVFS().exists(Value))
    return true;

  if (TypoCorrect) {
    // Check if the filename is a typo for an option flag. OptTable thinks
    // that all args that are not known options and that start with / are
    // filenames, but e.g. `/diagnostic:caret` is more likely a typo for
    // the option `/diagnostics:caret` than a reference to a file in the root
    // directory.
    std::string Nearest;
    if (getOpts().findNearest(Value, Nearest, getOptionVisibilityMask()) <= 1) {
      Diag(clang::diag::err_drv_no_such_file_with_suggestion)
          << Value << Nearest;
      return false;
    }
  }

  // In CL mode, don't error on apparently non-existent linker inputs, because
  // they can be influenced by linker flags the clang driver might not
  // understand.
  // Examples:
  // - `clang-cl main.cc ole32.lib` in a non-MSVC shell will make the driver
  //   module look for an MSVC installation in the registry. (We could ask
  //   the MSVCToolChain object if it can find `ole32.lib`, but the logic to
  //   look in the registry might move into lld-link in the future so that
  //   lld-link invocations in non-MSVC shells just work too.)
  // - `clang-cl ... /link ...` can pass arbitrary flags to the linker,
  //   including /libpath:, which is used to find .lib and .obj files.
  // So do not diagnose this on the driver level. Rely on the linker diagnosing
  // it. (If we don't end up invoking the linker, this means we'll emit a
  // "'linker' input unused [-Wunused-command-line-argument]" warning instead
  // of an error.)
  //
  // Only do this skip after the typo correction step above. `/Brepo` is treated
  // as TY_Object, but it's clearly a typo for `/Brepro`. It seems fine to emit
  // an error if we have a flag that's within an edit distance of 1 from a
  // flag. (Users can use `-Wl,` or `/linker` to launder the flag past the
  // driver in the unlikely case they run into this.)
  //
  // Don't do this for inputs that start with a '/', else we'd pass options
  // like /libpath: through to the linker silently.
  //
  // Emitting an error for linker inputs can also cause incorrect diagnostics
  // with the gcc driver. The command
  //     clang -fuse-ld=lld -Wl,--chroot,some/dir /file.o
  // will make lld look for some/dir/file.o, while we will diagnose here that
  // `/file.o` does not exist. However, configure scripts check if
  // `clang /GR-` compiles without error to see if the compiler is cl.exe,
  // so we can't downgrade diagnostics for `/GR-` from an error to a warning
  // in cc mode. (We can in cl mode because cl.exe itself only warns on
  // unknown flags.)
  if (IsCLMode() && Ty == types::TY_Object && !Value.starts_with("/"))
    return true;

  Diag(clang::diag::err_drv_no_such_file) << Value;
  return false;
}

// Get the C++20 Header Unit type corresponding to the input type.
static types::ID CXXHeaderUnitType(ModuleHeaderMode HM) {
  switch (HM) {
  case HeaderMode_User:
    return types::TY_CXXUHeader;
  case HeaderMode_System:
    return types::TY_CXXSHeader;
  case HeaderMode_Default:
    break;
  case HeaderMode_None:
    llvm_unreachable("should not be called in this case");
  }
  return types::TY_CXXHUHeader;
}

// Construct a the list of inputs and their types.
void Driver::BuildInputs(const ToolChain &TC, DerivedArgList &Args,
                         InputList &Inputs) const {
  const llvm::opt::OptTable &Opts = getOpts();
  // Track the current user specified (-x) input. We also explicitly track the
  // argument used to set the type; we only want to claim the type when we
  // actually use it, so we warn about unused -x arguments.
  types::ID InputType = types::TY_Nothing;
  Arg *InputTypeArg = nullptr;
  bool IsSYCL =
      Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false) ||
      Args.hasArg(options::OPT_fsycl_device_only);

  // The last /TC or /TP option sets the input type to C or C++ globally.
  if (Arg *TCTP = Args.getLastArgNoClaim(options::OPT__SLASH_TC,
                                         options::OPT__SLASH_TP)) {
    InputTypeArg = TCTP;
    InputType = TCTP->getOption().matches(options::OPT__SLASH_TC) && !IsSYCL
                    ? types::TY_C
                    : types::TY_CXX;

    Arg *Previous = nullptr;
    bool ShowNote = false;
    for (Arg *A :
         Args.filtered(options::OPT__SLASH_TC, options::OPT__SLASH_TP)) {
      if (Previous) {
        Diag(clang::diag::warn_drv_overriding_option)
            << Previous->getSpelling() << A->getSpelling();
        ShowNote = true;
      }
      Previous = A;
    }
    if (ShowNote)
      Diag(clang::diag::note_drv_t_option_is_global);
  }

  // CUDA/HIP and their preprocessor expansions can be accepted by CL mode.
  // Warn -x after last input file has no effect
  auto LastXArg = Args.getLastArgValue(options::OPT_x);
  const llvm::StringSet<> ValidXArgs = {"cuda", "hip", "cui", "hipi"};
  if (!IsCLMode() || ValidXArgs.contains(LastXArg)) {
    Arg *LastXArg = Args.getLastArgNoClaim(options::OPT_x);
    Arg *LastInputArg = Args.getLastArgNoClaim(options::OPT_INPUT);
    if (LastXArg && LastInputArg &&
        LastInputArg->getIndex() < LastXArg->getIndex())
      Diag(clang::diag::warn_drv_unused_x) << LastXArg->getValue();
  } else {
    // In CL mode suggest /TC or /TP since -x doesn't make sense if passed via
    // /clang:.
    if (auto *A = Args.getLastArg(options::OPT_x))
      Diag(diag::err_drv_unsupported_opt_with_suggestion)
          << A->getAsString(Args) << "/TC' or '/TP";
  }

  for (Arg *A : Args) {
    if (A->getOption().getKind() == Option::InputClass) {
      const char *Value = A->getValue();
      types::ID Ty = types::TY_INVALID;

      // Infer the input type if necessary.
      if (InputType == types::TY_Nothing) {
        // If there was an explicit arg for this, claim it.
        if (InputTypeArg)
          InputTypeArg->claim();

        types::ID CType = types::TY_C;
        // For SYCL, all source file inputs are considered C++.
        if (IsSYCL)
          CType = types::TY_CXX;

        // stdin must be handled specially.
        if (memcmp(Value, "-", 2) == 0) {
          if (IsFlangMode()) {
            Ty = types::TY_Fortran;
          } else if (IsDXCMode()) {
            Ty = types::TY_HLSL;
          } else {
            // If running with -E, treat as a C input (this changes the
            // builtin macros, for example). This may be overridden by -ObjC
            // below.
            //
            // Otherwise emit an error but still use a valid type to avoid
            // spurious errors (e.g., no inputs).
            assert(!CCGenDiagnostics && "stdin produces no crash reproducer");
            if (!Args.hasArgNoClaim(options::OPT_E) && !CCCIsCPP())
              Diag(IsCLMode() ? clang::diag::err_drv_unknown_stdin_type_clang_cl
                              : clang::diag::err_drv_unknown_stdin_type);
            Ty = types::TY_C;
          }
        } else {
          // Otherwise lookup by extension.
          // Fallback is C if invoked as C preprocessor, C++ if invoked with
          // clang-cl /E, or Object otherwise.
          // We use a host hook here because Darwin at least has its own
          // idea of what .s is.
          if (const char *Ext = strrchr(Value, '.'))
            Ty = TC.LookupTypeForExtension(Ext + 1);

          // For SYCL, convert C-type sources to C++-type sources.
          if (IsSYCL) {
            types::ID OldTy = Ty;
            switch (Ty) {
            case types::TY_C:
              Ty = types::TY_CXX;
              break;
            case types::TY_CHeader:
              Ty = types::TY_CXXHeader;
              break;
            case types::TY_PP_C:
              Ty = types::TY_PP_CXX;
              break;
            case types::TY_PP_CHeader:
              Ty = types::TY_PP_CXXHeader;
              break;
            default:
              break;
            }
            if (OldTy != Ty) {
              Diag(clang::diag::warn_drv_fsycl_with_c_type)
                  << getTypeName(OldTy) << getTypeName(Ty);
            }
          }

          if (Ty == types::TY_INVALID) {
            if (IsCLMode() && (Args.hasArgNoClaim(options::OPT_E) || CCGenDiagnostics))
              Ty = types::TY_CXX;
            else if (CCCIsCPP() || CCGenDiagnostics)
              Ty = CType;
            else
              Ty = types::TY_Object;
          }

          // If the driver is invoked as C++ compiler (like clang++ or c++) it
          // should autodetect some input files as C++ for g++ compatibility.
          if (CCCIsCXX()) {
            types::ID OldTy = Ty;
            Ty = types::lookupCXXTypeForCType(Ty);

            // Do not complain about foo.h, when we are known to be processing
            // it as a C++20 header unit.
            if (Ty != OldTy && !(OldTy == types::TY_CHeader && hasHeaderMode()))
              Diag(clang::diag::warn_drv_treating_input_as_cxx)
                  << getTypeName(OldTy) << getTypeName(Ty);
          }

          // If running with -fthinlto-index=, extensions that normally identify
          // native object files actually identify LLVM bitcode files.
          if (Args.hasArgNoClaim(options::OPT_fthinlto_index_EQ) &&
              Ty == types::TY_Object)
            Ty = types::TY_LLVM_BC;
        }

        // -ObjC and -ObjC++ override the default language, but only for "source
        // files". We just treat everything that isn't a linker input as a
        // source file.
        //
        // FIXME: Clean this up if we move the phase sequence into the type.
        if (Ty != types::TY_Object) {
          if (Args.hasArg(options::OPT_ObjC))
            Ty = types::TY_ObjC;
          else if (Args.hasArg(options::OPT_ObjCXX))
            Ty = types::TY_ObjCXX;
        }

        // Disambiguate headers that are meant to be header units from those
        // intended to be PCH.  Avoid missing '.h' cases that are counted as
        // C headers by default - we know we are in C++ mode and we do not
        // want to issue a complaint about compiling things in the wrong mode.
        if ((Ty == types::TY_CXXHeader || Ty == types::TY_CHeader) &&
            hasHeaderMode())
          Ty = CXXHeaderUnitType(CXX20HeaderType);
      } else {
        assert(InputTypeArg && "InputType set w/o InputTypeArg");
        if (!InputTypeArg->getOption().matches(options::OPT_x)) {
          // If emulating cl.exe, make sure that /TC and /TP don't affect input
          // object files.
          const char *Ext = strrchr(Value, '.');
          if (Ext && TC.LookupTypeForExtension(Ext + 1) == types::TY_Object)
            Ty = types::TY_Object;
        }
        if (Ty == types::TY_INVALID) {
          Ty = InputType;
          InputTypeArg->claim();
        }
      }

      if ((Ty == types::TY_C || Ty == types::TY_CXX) &&
          Args.hasArgNoClaim(options::OPT_hipstdpar))
        Ty = types::TY_HIP;

      if (DiagnoseInputExistence(Args, Value, Ty, /*TypoCorrect=*/true))
        Inputs.push_back(std::make_pair(Ty, A));

    } else if (A->getOption().matches(options::OPT__SLASH_Tc)) {
      StringRef Value = A->getValue();
      if (DiagnoseInputExistence(Args, Value, types::TY_C,
                                 /*TypoCorrect=*/false)) {
        Arg *InputArg = MakeInputArg(Args, Opts, A->getValue());
        Inputs.push_back(
            std::make_pair(IsSYCL ? types::TY_CXX : types::TY_C, InputArg));
      }
      A->claim();
    } else if (A->getOption().matches(options::OPT__SLASH_Tp)) {
      StringRef Value = A->getValue();
      if (DiagnoseInputExistence(Args, Value, types::TY_CXX,
                                 /*TypoCorrect=*/false)) {
        Arg *InputArg = MakeInputArg(Args, Opts, A->getValue());
        Inputs.push_back(std::make_pair(types::TY_CXX, InputArg));
      }
      A->claim();
    } else if (A->getOption().hasFlag(options::LinkerInput)) {
      // Just treat as object type, we could make a special type for this if
      // necessary.
      Inputs.push_back(std::make_pair(types::TY_Object, A));

    } else if (A->getOption().matches(options::OPT_x)) {
      InputTypeArg = A;
      InputType = types::lookupTypeForTypeSpecifier(A->getValue());
      A->claim();

      // Follow gcc behavior and treat as linker input for invalid -x
      // options. Its not clear why we shouldn't just revert to unknown; but
      // this isn't very important, we might as well be bug compatible.
      if (!InputType) {
        Diag(clang::diag::err_drv_unknown_language) << A->getValue();
        InputType = types::TY_Object;
      }
      // Emit an error if c-compilation is forced in -fsycl mode
      if (IsSYCL && (InputType == types::TY_C || InputType == types::TY_PP_C ||
                     InputType == types::TY_CHeader))
        Diag(clang::diag::err_drv_fsycl_with_c_type) << A->getAsString(Args);

      // If the user has put -fmodule-header{,=} then we treat C++ headers as
      // header unit inputs.  So we 'promote' -xc++-header appropriately.
      if (InputType == types::TY_CXXHeader && hasHeaderMode())
        InputType = CXXHeaderUnitType(CXX20HeaderType);
    } else if (A->getOption().getID() == options::OPT_U) {
      assert(A->getNumValues() == 1 && "The /U option has one value.");
      StringRef Val = A->getValue(0);
      if (Val.find_first_of("/\\") != StringRef::npos) {
        // Warn about e.g. "/Users/me/myfile.c".
        Diag(diag::warn_slash_u_filename) << Val;
        Diag(diag::note_use_dashdash);
      }
    }
    // TODO: remove when -foffload-static-lib support is dropped.
    else if (A->getOption().matches(options::OPT_offload_lib_Group)) {
      // Add the foffload-static-lib library to the command line to allow
      // processing when no source or object is supplied as well as proper
      // host link.
      Arg *InputArg = MakeInputArg(Args, Opts, A->getValue());
      Inputs.push_back(std::make_pair(types::TY_Object, InputArg));
      A->claim();
      // Use of -foffload-static-lib and -foffload-whole-static-lib are
      // deprecated with the updated functionality to scan the static libs.
      Diag(clang::diag::warn_drv_deprecated_option)
          << A->getAsString(Args) << A->getValue();
    }
  }
  if (CCCIsCPP() && Inputs.empty()) {
    // If called as standalone preprocessor, stdin is processed
    // if no other input is present.
    Arg *A = MakeInputArg(Args, Opts, "-");
    Inputs.push_back(std::make_pair(types::TY_C, A));
  }
}

static bool runBundler(const SmallVectorImpl<StringRef> &InputArgs,
                       Compilation &C) {
  // Find bundler.
  StringRef ExecPath(C.getArgs().MakeArgString(C.getDriver().Dir));
  llvm::ErrorOr<std::string> BundlerBinary =
      llvm::sys::findProgramByName("clang-offload-bundler", ExecPath);
  SmallVector<StringRef, 6> BundlerArgs;
  BundlerArgs.push_back(BundlerBinary.getError() ? "clang-offload-bundler"
                                                 : BundlerBinary.get().c_str());
  BundlerArgs.append(InputArgs);
  // Since this is run in real time and not in the toolchain, output the
  // command line if requested.
  bool OutputOnly = C.getArgs().hasArg(options::OPT__HASH_HASH_HASH);
  if (C.getArgs().hasArg(options::OPT_v) || OutputOnly) {
    for (StringRef A : BundlerArgs)
      if (OutputOnly)
        llvm::errs() << "\"" << A << "\" ";
      else
        llvm::errs() << A << " ";
    llvm::errs() << '\n';
  }
  if (BundlerBinary.getError())
    return false;

  return !llvm::sys::ExecuteAndWait(BundlerBinary.get(), BundlerArgs);
}

static bool hasFPGABinary(Compilation &C, std::string Object, types::ID Type) {
  assert(types::isFPGA(Type) && "unexpected Type for FPGA binary check");
  // Do not do the check if the file doesn't exist
  if (!llvm::sys::fs::exists(Object))
    return false;

  // Only static archives are valid FPGA Binaries for unbundling.
  if (!isStaticArchiveFile(Object))
    return false;

  // Temporary names for the output.
  llvm::Triple TT;
  TT.setArchName(types::getTypeName(Type));
  TT.setVendorName("intel");
  TT.setOS(llvm::Triple::UnknownOS);

  // Checking uses -check-section option with the input file, no output
  // file and the target triple being looked for.
  const char *Targets =
      C.getArgs().MakeArgString(Twine("-targets=sycl-") + TT.str());
  const char *Inputs = C.getArgs().MakeArgString(Twine("-input=") + Object);
  // Always use -type=ao for aocx/aocr bundle checking.  The 'bundles' are
  // actually archives.
  SmallVector<StringRef, 6> BundlerArgs = {"-type=ao", Targets, Inputs,
                                           "-check-section"};
  return runBundler(BundlerArgs, C);
}

static SmallVector<std::string, 4> getOffloadSections(Compilation &C,
                                                      const StringRef &File) {
  // Do not do the check if the file doesn't exist
  if (!llvm::sys::fs::exists(File))
    return {};

  bool IsArchive = isStaticArchiveFile(File);
  if (!(IsArchive || isObjectFile(File.str())))
    return {};

  // Use the bundler to grab the list of sections from the given archive
  // or object.
  StringRef ExecPath(C.getArgs().MakeArgString(C.getDriver().Dir));
  llvm::ErrorOr<std::string> BundlerBinary =
      llvm::sys::findProgramByName("clang-offload-bundler", ExecPath);
  const char *Input = C.getArgs().MakeArgString(Twine("-input=") + File.str());
  // Always use -type=ao for bundle checking.  The 'bundles' are
  // actually archives.
  SmallVector<StringRef, 6> BundlerArgs = {
      BundlerBinary.get(), IsArchive ? "-type=ao" : "-type=o", Input, "-list"};
  // Since this is run in real time and not in the toolchain, output the
  // command line if requested.
  bool OutputOnly = C.getArgs().hasArg(options::OPT__HASH_HASH_HASH);
  if (C.getArgs().hasArg(options::OPT_v) || OutputOnly) {
    for (StringRef A : BundlerArgs)
      if (OutputOnly)
        llvm::errs() << "\"" << A << "\" ";
      else
        llvm::errs() << A << " ";
    llvm::errs() << '\n';
  }
  if (BundlerBinary.getError())
    return {};
  llvm::SmallString<64> OutputFile(
      C.getDriver().GetTemporaryPath("bundle-list", "txt"));
  llvm::FileRemover OutputRemover(OutputFile.c_str());
  std::optional<llvm::StringRef> Redirects[] = {
      {""},
      OutputFile.str(),
      OutputFile.str(),
  };

  std::string ErrorMessage;
  if (llvm::sys::ExecuteAndWait(BundlerBinary.get(), BundlerArgs, {}, Redirects,
                                /*SecondsToWait*/ 0, /*MemoryLimit*/ 0,
                                &ErrorMessage)) {
    // Could not get the information, return false
    return {};
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> OutputBuf =
      llvm::MemoryBuffer::getFile(OutputFile.c_str());
  if (!OutputBuf) {
    // Could not capture output, return false
    return {};
  }

  SmallVector<std::string, 4> Sections;
  for (llvm::line_iterator LineIt(**OutputBuf); !LineIt.is_at_end(); ++LineIt)
    Sections.push_back(LineIt->str());
  if (Sections.empty())
    return {};

  return Sections;
}

static bool hasSYCLDefaultSection(Compilation &C, const StringRef &File) {
  // Do not do the check if the file doesn't exist
  if (!llvm::sys::fs::exists(File))
    return false;

  bool IsArchive = isStaticArchiveFile(File);
  if (!(IsArchive || isObjectFile(File.str())))
    return false;

  llvm::Triple TT(C.getDriver().MakeSYCLDeviceTriple(getDefaultSYCLArch(C)));
  // Checking uses -check-section option with the input file, no output
  // file and the target triple being looked for.
  const char *Targets =
      C.getArgs().MakeArgString(Twine("-targets=sycl-") + TT.str());
  const char *Inputs = C.getArgs().MakeArgString(Twine("-input=") + File.str());
  SmallVector<StringRef, 6> BundlerArgs = {IsArchive ? "-type=ao" : "-type=o",
                                           Targets, Inputs, "-check-section"};
  return runBundler(BundlerArgs, C);
}

static bool hasOffloadSections(Compilation &C, const StringRef &File,
                               DerivedArgList &Args) {
  SmallVector<std::string, 4> Sections(getOffloadSections(C, File));
  return !Sections.empty();
}

// Simple helper function for Linker options, where the option is valid if
// it has '-' or '--' as the designator.
static bool optionMatches(const std::string &Option,
                          const std::string &OptCheck) {
  return (Option == OptCheck || ("-" + Option) == OptCheck);
}

// Process linker inputs for use with offload static libraries.  We are only
// handling options and explicitly named static archives as these need to be
// partially linked.
static SmallVector<const char *, 16>
getLinkerArgs(Compilation &C, DerivedArgList &Args, bool IncludeObj = false) {
  SmallVector<const char *, 16> LibArgs;
  SmallVector<std::string, 8> LibPaths;
  bool IsMSVC = C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment();
  // Add search directories from LIBRARY_PATH/LIB env variable
  std::optional<std::string> LibPath =
      llvm::sys::Process::GetEnv(IsMSVC ? "LIB" : "LIBRARY_PATH");
  if (LibPath) {
    SmallVector<StringRef, 8> SplitPaths;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
    llvm::SplitString(*LibPath, SplitPaths, EnvPathSeparatorStr);
    for (StringRef Path : SplitPaths)
      LibPaths.emplace_back(Path.trim());
  }
  // Add directories from user-specified -L options
  for (std::string LibDirs : Args.getAllArgValues(options::OPT_L))
    LibPaths.emplace_back(LibDirs);

  // Do processing for any -l<arg> options passed and see if any static
  // libraries representing the name exists.  If so, convert the name and
  // use that inline with the rest of the libraries.
  // TODO: The static archive processing for SYCL is done in a different
  // manner than the OpenMP processing.  We should try and refactor this
  // to use the OpenMP flow (adding -l<name> to the llvm-link step)
  auto resolveStaticLib = [&](StringRef LibName, bool IsStatic) -> bool {
    if (!LibName.starts_with("-l"))
      return false;
    for (auto &LPath : LibPaths) {
      if (!IsStatic) {
        // Current linking state is dynamic.  We will first check for the
        // shared object and not pull in the static library if it is found.
        SmallString<128> SoLibName(LPath);
        llvm::sys::path::append(SoLibName,
                                Twine("lib" + LibName.substr(2) + ".so").str());
        if (llvm::sys::fs::exists(SoLibName))
          return false;
      }
      SmallString<128> FullName(LPath);
      llvm::sys::path::append(FullName,
                              Twine("lib" + LibName.substr(2) + ".a").str());
      if (llvm::sys::fs::exists(FullName)) {
        LibArgs.push_back(Args.MakeArgString(FullName));
        return true;
      }
    }
    return false;
  };
  for (const auto *A : Args) {
    std::string FileName = A->getAsString(Args);
    static bool IsLinkStateStatic(Args.hasArg(options::OPT_static));
    auto addLibArg = [&](StringRef LibName) -> bool {
      if (isStaticArchiveFile(LibName) ||
          (IncludeObj && isObjectFile(LibName.str()))) {
        LibArgs.push_back(Args.MakeArgString(LibName));
        return true;
      }
      return false;
    };
    if (A->getOption().getKind() == Option::InputClass) {
      if (addLibArg(FileName))
        continue;
    }
    // Evaluate any libraries passed along after /link. These are typically
    // ignored by the driver and sent directly to the linker. When performing
    // offload, we should evaluate them at the driver level.
    if (A->getOption().matches(options::OPT__SLASH_link)) {
      for (StringRef Value : A->getValues()) {
        // Add any libpath values.
        if (Value.starts_with_insensitive("-libpath:") ||
            Value.starts_with_insensitive("/libpath:"))
          LibPaths.emplace_back(Value.substr(std::string("-libpath:").size()));
        if (addLibArg(Value))
          continue;
        for (auto LPath : LibPaths) {
          SmallString<128> FullLib(LPath);
          llvm::sys::path::append(FullLib, Value);
          if (addLibArg(FullLib))
            continue;
        }
      }
    }
    if (A->getOption().matches(options::OPT_Wl_COMMA) ||
        A->getOption().matches(options::OPT_Xlinker)) {
      // Parse through additional linker arguments that are meant to go
      // directly to the linker.
      // Keep the previous arg even if it is a new argument, for example:
      //   -Xlinker -rpath -Xlinker <dir>.
      // Without this history, we do not know that <dir> was assocated with
      // -rpath and is processed incorrectly.
      static std::string PrevArg;
      for (StringRef Value : A->getValues()) {
        auto addKnownValues = [&](const StringRef &V) {
          // Only add named static libs objects and --whole-archive options.
          if (optionMatches("-whole-archive", V.str()) ||
              optionMatches("-no-whole-archive", V.str()) ||
              isStaticArchiveFile(V) || (IncludeObj && isObjectFile(V.str()))) {
            LibArgs.push_back(Args.MakeArgString(V));
            return;
          }
          // Probably not the best way to handle this, but there are options
          // that take arguments which we should not add to the known values.
          // Handle -z and -rpath for now - can be expanded if/when usage shows
          // the need.
          if (PrevArg != "-z" && PrevArg != "-rpath" && V[0] != '-' &&
              isObjectFile(V.str())) {
            LibArgs.push_back(Args.MakeArgString(V));
            return;
          }
          if (optionMatches("-Bstatic", V.str()) ||
              optionMatches("-dn", V.str()) ||
              optionMatches("-non_shared", V.str()) ||
              optionMatches("-static", V.str())) {
            IsLinkStateStatic = true;
            return;
          }
          if (optionMatches("-Bdynamic", V.str()) ||
              optionMatches("-dy", V.str()) ||
              optionMatches("-call_shared", V.str())) {
            IsLinkStateStatic = false;
            return;
          }
          resolveStaticLib(V, IsLinkStateStatic);
        };
        if (Value[0] == '@') {
          // Found a response file, we want to expand contents to try and
          // discover more libraries and options.
          SmallVector<const char *, 20> ExpandArgs;
          ExpandArgs.push_back(Value.data());

          llvm::BumpPtrAllocator A;
          llvm::StringSaver S(A);
          llvm::cl::ExpandResponseFiles(
              S,
              IsMSVC ? llvm::cl::TokenizeWindowsCommandLine
                     : llvm::cl::TokenizeGNUCommandLine,
              ExpandArgs);
          for (StringRef EA : ExpandArgs)
            addKnownValues(EA);
        } else
          addKnownValues(Value);
        PrevArg = Value;
      }
      continue;
    }
    // Use of -foffload-static-lib and -foffload-whole-static-lib is
    // considered deprecated.  Usage should move to passing in the static
    // library name on the command line, encapsulating with
    // -Wl,--whole-archive <lib> -Wl,--no-whole-archive as needed.
    if (A->getOption().matches(options::OPT_foffload_static_lib_EQ)) {
      LibArgs.push_back(Args.MakeArgString(A->getValue()));
      continue;
    }
    if (A->getOption().matches(options::OPT_foffload_whole_static_lib_EQ)) {
      // For -foffload-whole-static-lib, we add the --whole-archive wrap
      // around the library which will be used during the partial link step.
      LibArgs.push_back("--whole-archive");
      LibArgs.push_back(Args.MakeArgString(A->getValue()));
      LibArgs.push_back("--no-whole-archive");
      continue;
    }
    if (A->getOption().matches(options::OPT_l))
      resolveStaticLib(A->getAsString(Args), IsLinkStateStatic);
  }
  return LibArgs;
}

static bool IsSYCLDeviceLibObj(std::string ObjFilePath, bool isMSVCEnv) {
  StringRef ObjFileName = llvm::sys::path::filename(ObjFilePath);
  StringRef ObjSuffix = isMSVCEnv ? ".obj" : ".o";
  bool Ret =
      (ObjFileName.starts_with("libsycl-") && ObjFileName.ends_with(ObjSuffix))
          ? true
          : false;
  return Ret;
}

// Goes through all of the arguments, including inputs expected for the
// linker directly, to determine if we need to potentially add the SYCL
// default triple.
bool Driver::checkForSYCLDefaultDevice(Compilation &C,
                                       DerivedArgList &Args) const {
  // Check only if enabled with -fsycl
  if (!Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false))
    return false;

  if (Args.hasArg(options::OPT_fno_sycl_link_spirv))
    return false;

  // Do not do the check if the default device is passed in -fsycl-targets
  // or if -fsycl-targets isn't passed (that implies default device)
  if (const Arg *A = Args.getLastArg(options::OPT_fsycl_targets_EQ)) {
    for (const char *Val : A->getValues()) {
      llvm::Triple TT(C.getDriver().MakeSYCLDeviceTriple(Val));
      if (TT.isSPIR() && TT.getSubArch() == llvm::Triple::NoSubArch)
        // Default triple found
        return false;
    }
  } else if (!Args.hasArg(options::OPT_fintelfpga))
    return false;

  SmallVector<const char *, 16> AllArgs(getLinkerArgs(C, Args, true));
  for (StringRef Arg : AllArgs) {
    if (hasSYCLDefaultSection(C, Arg))
      return true;
  }
  return false;
}

// Goes through all of the arguments, including inputs expected for the
// linker directly, to determine if we need to perform additional work for
// static offload libraries.
bool Driver::checkForOffloadStaticLib(Compilation &C,
                                      DerivedArgList &Args) const {
  // Check only if enabled with -fsycl or -fopenmp-targets
  if (!Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false) &&
      !Args.hasArg(options::OPT_fopenmp_targets_EQ))
    return false;

  // Right off the bat, assume the presence of -foffload-static-lib means
  // the need to perform linking steps for fat static archive offloading.
  // TODO: remove when -foffload-static-lib support is dropped.
  if (Args.hasArg(options::OPT_offload_lib_Group))
    return true;
  SmallVector<const char *, 16> OffloadLibArgs(getLinkerArgs(C, Args));
  for (StringRef OLArg : OffloadLibArgs)
    if (isStaticArchiveFile(OLArg) && hasOffloadSections(C, OLArg, Args)) {
      // FPGA binaries with AOCX or AOCR sections are not considered fat
      // static archives.
      return !(hasFPGABinary(C, OLArg.str(), types::TY_FPGA_AOCR) ||
               hasFPGABinary(C, OLArg.str(), types::TY_FPGA_AOCR_EMU) ||
               hasFPGABinary(C, OLArg.str(), types::TY_FPGA_AOCX));
    }
  return false;
}

/// Check whether the given input tree contains any clang-offload-dependency
/// actions.
static bool ContainsOffloadDepsAction(const Action *A) {
  if (isa<OffloadDepsJobAction>(A))
    return true;
  return llvm::any_of(A->inputs(), ContainsOffloadDepsAction);
}

namespace {
/// Provides a convenient interface for different programming models to generate
/// the required device actions.
class OffloadingActionBuilder final {
  /// Flag used to trace errors in the builder.
  bool IsValid = false;

  /// The compilation that is using this builder.
  Compilation &C;

  /// Map between an input argument and the offload kinds used to process it.
  std::map<const Arg *, unsigned> InputArgToOffloadKindMap;

  /// Map between a host action and its originating input argument.
  std::map<Action *, const Arg *> HostActionToInputArgMap;

  /// Builder interface. It doesn't build anything or keep any state.
  class DeviceActionBuilder {
  public:
    typedef const llvm::SmallVectorImpl<phases::ID> PhasesTy;

    enum ActionBuilderReturnCode {
      // The builder acted successfully on the current action.
      ABRT_Success,
      // The builder didn't have to act on the current action.
      ABRT_Inactive,
      // The builder was successful and requested the host action to not be
      // generated.
      ABRT_Ignore_Host,
    };

  protected:
    /// Compilation associated with this builder.
    Compilation &C;

    /// Tool chains associated with this builder. The same programming
    /// model may have associated one or more tool chains.
    SmallVector<const ToolChain *, 2> ToolChains;

    /// The derived arguments associated with this builder.
    DerivedArgList &Args;

    /// The inputs associated with this builder.
    const Driver::InputList &Inputs;

    /// The associated offload kind.
    Action::OffloadKind AssociatedOffloadKind = Action::OFK_None;

    /// The OffloadingActionBuilder reference.
    OffloadingActionBuilder &OffloadingActionBuilderRef;

  public:
    DeviceActionBuilder(Compilation &C, DerivedArgList &Args,
                        const Driver::InputList &Inputs,
                        Action::OffloadKind AssociatedOffloadKind,
                        OffloadingActionBuilder &OAB)
        : C(C), Args(Args), Inputs(Inputs),
          AssociatedOffloadKind(AssociatedOffloadKind),
          OffloadingActionBuilderRef(OAB) {}
    virtual ~DeviceActionBuilder() {}

    /// Fill up the array \a DA with all the device dependences that should be
    /// added to the provided host action \a HostAction. By default it is
    /// inactive.
    virtual ActionBuilderReturnCode
    getDeviceDependences(OffloadAction::DeviceDependences &DA,
                         phases::ID CurPhase, phases::ID FinalPhase,
                         PhasesTy &Phases) {
      return ABRT_Inactive;
    }

    /// Update the state to include the provided host action \a HostAction as a
    /// dependency of the current device action. By default it is inactive.
    virtual ActionBuilderReturnCode addDeviceDependences(Action *HostAction) {
      return ABRT_Inactive;
    }

    /// Append top level actions generated by the builder.
    virtual void appendTopLevelActions(ActionList &AL) {}

    /// Append top level actions specific for certain link situations.
    virtual void appendTopLevelLinkAction(ActionList &AL) {}

    /// Append linker device actions generated by the builder.
    virtual void appendLinkDeviceActions(ActionList &AL) {}

    /// Append linker host action generated by the builder.
    virtual Action* appendLinkHostActions(ActionList &AL) { return nullptr; }

    /// Append linker actions generated by the builder.
    virtual void appendLinkDependences(OffloadAction::DeviceDependences &DA) {}

    /// Append linker actions generated by the builder.
    virtual void addDeviceLinkDependencies(OffloadDepsJobAction *DA) {}

    /// Initialize the builder. Return true if any initialization errors are
    /// found.
    virtual bool initialize() { return false; }

    /// Return true if the builder can use bundling/unbundling.
    virtual bool canUseBundlerUnbundler() const { return false; }

    /// Return true if this builder is valid. We have a valid builder if we have
    /// associated device tool chains.
    bool isValid() { return !ToolChains.empty(); }

    /// Return the associated offload kind.
    Action::OffloadKind getAssociatedOffloadKind() {
      return AssociatedOffloadKind;
    }

    /// Push an action from a different DeviceActionBuilder (i.e., foreign
    /// action) in the current one
    virtual void pushForeignAction(Action *A) {}
  };

  /// Base class for CUDA/HIP action builder. It injects device code in
  /// the host backend action.
  class CudaActionBuilderBase : public DeviceActionBuilder {
  protected:
    /// Flags to signal if the user requested host-only or device-only
    /// compilation.
    bool CompileHostOnly = false;
    bool CompileDeviceOnly = false;
    bool EmitLLVM = false;
    bool EmitAsm = false;

    /// ID to identify each device compilation. For CUDA it is simply the
    /// GPU arch string. For HIP it is either the GPU arch string or GPU
    /// arch string plus feature strings delimited by a plus sign, e.g.
    /// gfx906+xnack.
    struct TargetID {
      /// Target ID string which is persistent throughout the compilation.
      const char *ID;
      TargetID(CudaArch Arch) { ID = CudaArchToString(Arch); }
      TargetID(const char *ID) : ID(ID) {}
      operator const char *() { return ID; }
      operator StringRef() { return StringRef(ID); }
    };
    /// List of GPU architectures to use in this compilation.
    SmallVector<TargetID, 4> GpuArchList;

    /// The CUDA actions for the current input.
    ActionList CudaDeviceActions;

    /// The CUDA fat binary if it was generated for the current input.
    Action *CudaFatBinary = nullptr;

    /// Flag that is set to true if this builder acted on the current input.
    bool IsActive = false;

    /// Flag for -fgpu-rdc.
    bool Relocatable = false;

    /// Default GPU architecture if there's no one specified.
    CudaArch DefaultCudaArch = CudaArch::UNKNOWN;

    /// Method to generate compilation unit ID specified by option
    /// '-fuse-cuid='.
    enum UseCUIDKind { CUID_Hash, CUID_Random, CUID_None, CUID_Invalid };
    UseCUIDKind UseCUID = CUID_Hash;

    /// Compilation unit ID specified by option '-cuid='.
    StringRef FixedCUID;

  public:
    CudaActionBuilderBase(Compilation &C, DerivedArgList &Args,
                          const Driver::InputList &Inputs,
                          Action::OffloadKind OFKind,
                          OffloadingActionBuilder &OAB)
        : DeviceActionBuilder(C, Args, Inputs, OFKind, OAB) {

      CompileDeviceOnly = C.getDriver().offloadDeviceOnly();
      Relocatable = Args.hasFlag(options::OPT_fgpu_rdc,
                                 options::OPT_fno_gpu_rdc, /*Default=*/false);
    }

    ActionBuilderReturnCode addDeviceDependences(Action *HostAction) override {
      // While generating code for CUDA, we only depend on the host input action
      // to trigger the creation of all the CUDA device actions.

      // If we are dealing with an input action, replicate it for each GPU
      // architecture. If we are in host-only mode we return 'success' so that
      // the host uses the CUDA offload kind.
      if (auto *IA = dyn_cast<InputAction>(HostAction)) {
        assert(!GpuArchList.empty() &&
               "We should have at least one GPU architecture.");

        // If the host input is not CUDA or HIP, we don't need to bother about
        // this input.
        if (!(IA->getType() == types::TY_CUDA ||
              IA->getType() == types::TY_HIP ||
              IA->getType() == types::TY_PP_HIP)) {
          // The builder will ignore this input.
          IsActive = false;
          return ABRT_Inactive;
        }

        // Set the flag to true, so that the builder acts on the current input.
        IsActive = true;

        if (CompileHostOnly)
          return ABRT_Success;

        // Replicate inputs for each GPU architecture.
        auto Ty = IA->getType() == types::TY_HIP ? types::TY_HIP_DEVICE
                                                 : types::TY_CUDA_DEVICE;
        std::string CUID = FixedCUID.str();
        if (CUID.empty()) {
          if (UseCUID == CUID_Random)
            CUID = llvm::utohexstr(llvm::sys::Process::GetRandomNumber(),
                                   /*LowerCase=*/true);
          else if (UseCUID == CUID_Hash) {
            llvm::MD5 Hasher;
            llvm::MD5::MD5Result Hash;
            SmallString<256> RealPath;
            llvm::sys::fs::real_path(IA->getInputArg().getValue(), RealPath,
                                     /*expand_tilde=*/true);
            Hasher.update(RealPath);
            for (auto *A : Args) {
              if (A->getOption().matches(options::OPT_INPUT))
                continue;
              Hasher.update(A->getAsString(Args));
            }
            Hasher.final(Hash);
            CUID = llvm::utohexstr(Hash.low(), /*LowerCase=*/true);
          }
        }
        IA->setId(CUID);

        for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {
          CudaDeviceActions.push_back(
              C.MakeAction<InputAction>(IA->getInputArg(), Ty, IA->getId()));
        }

        return ABRT_Success;
      }

      // If this is an unbundling action use it as is for each CUDA toolchain.
      if (auto *UA = dyn_cast<OffloadUnbundlingJobAction>(HostAction)) {

        // If -fgpu-rdc is disabled, should not unbundle since there is no
        // device code to link.
        if (UA->getType() == types::TY_Object && !Relocatable)
          return ABRT_Inactive;

        CudaDeviceActions.clear();
        auto *IA = cast<InputAction>(UA->getInputs().back());
        std::string FileName = IA->getInputArg().getAsString(Args);
        // Check if the type of the file is the same as the action. Do not
        // unbundle it if it is not. Do not unbundle .so files, for example,
        // which are not object files. Files with extension ".lib" is classified
        // as TY_Object but they are actually archives, therefore should not be
        // unbundled here as objects. They will be handled at other places.
        const StringRef LibFileExt = ".lib";
        if (IA->getType() == types::TY_Object &&
            (!llvm::sys::path::has_extension(FileName) ||
             types::lookupTypeForExtension(
                 llvm::sys::path::extension(FileName).drop_front()) !=
                 types::TY_Object ||
             llvm::sys::path::extension(FileName) == LibFileExt))
          return ABRT_Inactive;

        for (auto Arch : GpuArchList) {
          CudaDeviceActions.push_back(UA);
          UA->registerDependentActionInfo(ToolChains[0], Arch,
                                          AssociatedOffloadKind);
        }
        IsActive = true;
        return ABRT_Success;
      }

      return IsActive ? ABRT_Success : ABRT_Inactive;
    }

    void appendTopLevelActions(ActionList &AL) override {
      // Utility to append actions to the top level list.
      auto AddTopLevel = [&](Action *A, TargetID TargetID) {
        OffloadAction::DeviceDependences Dep;
        Dep.add(*A, *ToolChains.front(), TargetID, AssociatedOffloadKind);
        AL.push_back(C.MakeAction<OffloadAction>(Dep, A->getType()));
      };

      // If we have a fat binary, add it to the list.
      if (CudaFatBinary) {
        AddTopLevel(CudaFatBinary, CudaArch::UNUSED);
        CudaDeviceActions.clear();
        CudaFatBinary = nullptr;
        return;
      }

      if (CudaDeviceActions.empty())
        return;

      // If we have CUDA actions at this point, that's because we have a have
      // partial compilation, so we should have an action for each GPU
      // architecture.
      assert(CudaDeviceActions.size() == GpuArchList.size() &&
             "Expecting one action per GPU architecture.");
      assert(ToolChains.size() == 1 &&
             "Expecting to have a single CUDA toolchain.");
      for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I)
        AddTopLevel(CudaDeviceActions[I], GpuArchList[I]);

      CudaDeviceActions.clear();
    }

    /// Get canonicalized offload arch option. \returns empty StringRef if the
    /// option is invalid.
    virtual StringRef getCanonicalOffloadArch(StringRef Arch) = 0;

    virtual std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
    getConflictOffloadArchCombination(const std::set<StringRef> &GpuArchs) = 0;

    bool initialize() override {
      assert(AssociatedOffloadKind == Action::OFK_Cuda ||
             AssociatedOffloadKind == Action::OFK_HIP);

      // We don't need to support CUDA.
      if (AssociatedOffloadKind == Action::OFK_Cuda &&
          !C.hasOffloadToolChain<Action::OFK_Cuda>())
        return false;

      // We don't need to support HIP.
      if (AssociatedOffloadKind == Action::OFK_HIP &&
          !C.hasOffloadToolChain<Action::OFK_HIP>())
        return false;

      const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
      assert(HostTC && "No toolchain for host compilation.");
      if (HostTC->getTriple().isNVPTX() ||
          HostTC->getTriple().getArch() == llvm::Triple::amdgcn) {
        // We do not support targeting NVPTX/AMDGCN for host compilation. Throw
        // an error and abort pipeline construction early so we don't trip
        // asserts that assume device-side compilation.
        C.getDriver().Diag(diag::err_drv_cuda_host_arch)
            << HostTC->getTriple().getArchName();
        return true;
      }

      ToolChains.push_back(
          AssociatedOffloadKind == Action::OFK_Cuda
              ? C.getSingleOffloadToolChain<Action::OFK_Cuda>()
              : C.getSingleOffloadToolChain<Action::OFK_HIP>());

      CompileHostOnly = C.getDriver().offloadHostOnly();
      EmitLLVM = Args.getLastArg(options::OPT_emit_llvm);
      EmitAsm = Args.getLastArg(options::OPT_S);
      FixedCUID = Args.getLastArgValue(options::OPT_cuid_EQ);
      if (Arg *A = Args.getLastArg(options::OPT_fuse_cuid_EQ)) {
        StringRef UseCUIDStr = A->getValue();
        UseCUID = llvm::StringSwitch<UseCUIDKind>(UseCUIDStr)
                      .Case("hash", CUID_Hash)
                      .Case("random", CUID_Random)
                      .Case("none", CUID_None)
                      .Default(CUID_Invalid);
        if (UseCUID == CUID_Invalid) {
          C.getDriver().Diag(diag::err_drv_invalid_value)
              << A->getAsString(Args) << UseCUIDStr;
          C.setContainsError();
          return true;
        }
      }

      // --offload and --offload-arch options are mutually exclusive.
      if (Args.hasArgNoClaim(options::OPT_offload_EQ) &&
          Args.hasArgNoClaim(options::OPT_offload_arch_EQ,
                             options::OPT_no_offload_arch_EQ)) {
        C.getDriver().Diag(diag::err_opt_not_valid_with_opt) << "--offload-arch"
                                                             << "--offload";
      }

      // Collect all offload arch parameters, removing duplicates.
      std::set<StringRef> GpuArchs;
      bool Error = false;
      for (Arg *A : Args) {
        if (!(A->getOption().matches(options::OPT_offload_arch_EQ) ||
              A->getOption().matches(options::OPT_no_offload_arch_EQ)))
          continue;
        A->claim();

        for (StringRef ArchStr : llvm::split(A->getValue(), ",")) {
          if (A->getOption().matches(options::OPT_no_offload_arch_EQ) &&
              ArchStr == "all") {
            GpuArchs.clear();
          } else if (ArchStr == "native") {
            const ToolChain &TC = *ToolChains.front();
            auto GPUsOrErr = ToolChains.front()->getSystemGPUArchs(Args);
            if (!GPUsOrErr) {
              TC.getDriver().Diag(diag::err_drv_undetermined_gpu_arch)
                  << llvm::Triple::getArchTypeName(TC.getArch())
                  << llvm::toString(GPUsOrErr.takeError()) << "--offload-arch";
              continue;
            }

            for (auto GPU : *GPUsOrErr) {
              GpuArchs.insert(Args.MakeArgString(GPU));
            }
          } else {
            ArchStr = getCanonicalOffloadArch(ArchStr);
            if (ArchStr.empty()) {
              Error = true;
            } else if (A->getOption().matches(options::OPT_offload_arch_EQ))
              GpuArchs.insert(ArchStr);
            else if (A->getOption().matches(options::OPT_no_offload_arch_EQ))
              GpuArchs.erase(ArchStr);
            else
              llvm_unreachable("Unexpected option.");
          }
        }
      }

      auto &&ConflictingArchs = getConflictOffloadArchCombination(GpuArchs);
      if (ConflictingArchs) {
        C.getDriver().Diag(clang::diag::err_drv_bad_offload_arch_combo)
            << ConflictingArchs->first << ConflictingArchs->second;
        C.setContainsError();
        return true;
      }

      // Collect list of GPUs remaining in the set.
      for (auto Arch : GpuArchs)
        GpuArchList.push_back(Arch.data());

      // Default to sm_20 which is the lowest common denominator for
      // supported GPUs.  sm_20 code should work correctly, if
      // suboptimally, on all newer GPUs.
      if (GpuArchList.empty()) {
        if (ToolChains.front()->getTriple().isSPIRV())
          GpuArchList.push_back(CudaArch::Generic);
        else
          GpuArchList.push_back(DefaultCudaArch);
      }

      return Error;
    }
  };

  /// \brief CUDA action builder. It injects device code in the host backend
  /// action.
  class CudaActionBuilder final : public CudaActionBuilderBase {
  public:
    CudaActionBuilder(Compilation &C, DerivedArgList &Args,
                      const Driver::InputList &Inputs,
                      OffloadingActionBuilder &OAB)
        : CudaActionBuilderBase(C, Args, Inputs, Action::OFK_Cuda, OAB) {
      DefaultCudaArch = CudaArch::SM_35;
    }

    StringRef getCanonicalOffloadArch(StringRef ArchStr) override {
      CudaArch Arch = StringToCudaArch(ArchStr);
      if (Arch == CudaArch::UNKNOWN || !IsNVIDIAGpuArch(Arch)) {
        C.getDriver().Diag(clang::diag::err_drv_cuda_bad_gpu_arch) << ArchStr;
        return StringRef();
      }
      return CudaArchToString(Arch);
    }

    std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
    getConflictOffloadArchCombination(
        const std::set<StringRef> &GpuArchs) override {
      return std::nullopt;
    }

    bool canUseBundlerUnbundler() const override {
      return Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false);
    }

    ActionBuilderReturnCode
    getDeviceDependences(OffloadAction::DeviceDependences &DA,
                         phases::ID CurPhase, phases::ID FinalPhase,
                         PhasesTy &Phases) override {
      if (!IsActive)
        return ABRT_Inactive;

      // If we don't have more CUDA actions, we don't have any dependences to
      // create for the host.
      if (CudaDeviceActions.empty())
        return ABRT_Success;

      assert(CudaDeviceActions.size() == GpuArchList.size() &&
             "Expecting one action per GPU architecture.");
      assert(!CompileHostOnly &&
             "Not expecting CUDA actions in host-only compilation.");

      // If we are generating code for the device or we are in a backend phase,
      // we attempt to generate the fat binary. We compile each arch to ptx and
      // assemble to cubin, then feed the cubin *and* the ptx into a device
      // "link" action, which uses fatbinary to combine these cubins into one
      // fatbin.  The fatbin is then an input to the host action if not in
      // device-only mode.
      if (CompileDeviceOnly || CurPhase == phases::Backend) {
        ActionList DeviceActions;
        for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {
          // Produce the device action from the current phase up to the assemble
          // phase.
          for (auto Ph : Phases) {
            // Skip the phases that were already dealt with.
            if (Ph < CurPhase)
              continue;
            // We have to be consistent with the host final phase.
            if (Ph > FinalPhase)
              break;

            CudaDeviceActions[I] = C.getDriver().ConstructPhaseAction(
                C, Args, Ph, CudaDeviceActions[I], Action::OFK_Cuda);

            if (Ph == phases::Assemble)
              break;
          }

          // If we didn't reach the assemble phase, we can't generate the fat
          // binary. We don't need to generate the fat binary if we are not in
          // device-only mode.
          if (!isa<AssembleJobAction>(CudaDeviceActions[I]) ||
              CompileDeviceOnly)
            continue;

          Action *AssembleAction = CudaDeviceActions[I];
          assert(AssembleAction->getType() == types::TY_Object);
          assert(AssembleAction->getInputs().size() == 1);

          Action *BackendAction = AssembleAction->getInputs()[0];
          assert(BackendAction->getType() == types::TY_PP_Asm);

          for (auto &A : {AssembleAction, BackendAction}) {
            OffloadAction::DeviceDependences DDep;
            DDep.add(*A, *ToolChains.front(), GpuArchList[I], Action::OFK_Cuda);
            DeviceActions.push_back(
                C.MakeAction<OffloadAction>(DDep, A->getType()));
          }
        }

        // We generate the fat binary if we have device input actions.
        if (!DeviceActions.empty()) {
          CudaFatBinary =
              C.MakeAction<LinkJobAction>(DeviceActions, types::TY_CUDA_FATBIN);

          if (!CompileDeviceOnly) {
            DA.add(*CudaFatBinary, *ToolChains.front(), /*BoundArch=*/nullptr,
                   Action::OFK_Cuda);
            // Clear the fat binary, it is already a dependence to an host
            // action.
            CudaFatBinary = nullptr;
          }

          // Remove the CUDA actions as they are already connected to an host
          // action or fat binary.
          CudaDeviceActions.clear();
        }

        // We avoid creating host action in device-only mode.
        return CompileDeviceOnly ? ABRT_Ignore_Host : ABRT_Success;
      } else if (CurPhase > phases::Backend) {
        // If we are past the backend phase and still have a device action, we
        // don't have to do anything as this action is already a device
        // top-level action.
        return ABRT_Success;
      }

      assert(CurPhase < phases::Backend && "Generating single CUDA "
                                           "instructions should only occur "
                                           "before the backend phase!");

      // By default, we produce an action for each device arch.
      for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {

        CudaDeviceActions[I] = C.getDriver().ConstructPhaseAction(
            C, Args, CurPhase, CudaDeviceActions[I]);

        if (CurPhase == phases::Compile) {
          OffloadAction::DeviceDependences DDep;
          DDep.add(*CudaDeviceActions[I], *ToolChains.front(), GpuArchList[I],
                   Action::OFK_Cuda);

          OffloadingActionBuilderRef.pushForeignAction(
              C.MakeAction<OffloadAction>(
                  DDep, DDep.getActions().front()->getType()));
        }
      }

      return ABRT_Success;
    }
  };
  /// \brief HIP action builder. It injects device code in the host backend
  /// action.
  class HIPActionBuilder final : public CudaActionBuilderBase {
    /// The linker inputs obtained for each device arch.
    SmallVector<ActionList, 8> DeviceLinkerInputs;
    // The default bundling behavior depends on the type of output, therefore
    // BundleOutput needs to be tri-value: None, true, or false.
    // Bundle code objects except --no-gpu-output is specified for device
    // only compilation. Bundle other type of output files only if
    // --gpu-bundle-output is specified for device only compilation.
    std::optional<bool> BundleOutput;
    std::optional<bool> EmitReloc;

  public:
    HIPActionBuilder(Compilation &C, DerivedArgList &Args,
                     const Driver::InputList &Inputs,
                     OffloadingActionBuilder &OAB)
        : CudaActionBuilderBase(C, Args, Inputs, Action::OFK_HIP, OAB) {

      DefaultCudaArch = CudaArch::GFX906;

      if (Args.hasArg(options::OPT_fhip_emit_relocatable,
                      options::OPT_fno_hip_emit_relocatable)) {
        EmitReloc = Args.hasFlag(options::OPT_fhip_emit_relocatable,
                                 options::OPT_fno_hip_emit_relocatable, false);

        if (*EmitReloc) {
          if (Relocatable) {
            C.getDriver().Diag(diag::err_opt_not_valid_with_opt)
                << "-fhip-emit-relocatable"
                << "-fgpu-rdc";
          }

          if (!CompileDeviceOnly) {
            C.getDriver().Diag(diag::err_opt_not_valid_without_opt)
                << "-fhip-emit-relocatable"
                << "--cuda-device-only";
          }
        }
      }

      if (Args.hasArg(options::OPT_gpu_bundle_output,
                      options::OPT_no_gpu_bundle_output))
        BundleOutput = Args.hasFlag(options::OPT_gpu_bundle_output,
                                    options::OPT_no_gpu_bundle_output, true) &&
                       (!EmitReloc || !*EmitReloc);
    }

    bool canUseBundlerUnbundler() const override { return true; }

    StringRef getCanonicalOffloadArch(StringRef IdStr) override {
      llvm::StringMap<bool> Features;
      // getHIPOffloadTargetTriple() is known to return valid value as it has
      // been called successfully in the CreateOffloadingDeviceToolChains().
      auto ArchStr = parseTargetID(
          *getHIPOffloadTargetTriple(C.getDriver(), C.getInputArgs()), IdStr,
          &Features);
      if (!ArchStr) {
        C.getDriver().Diag(clang::diag::err_drv_bad_target_id) << IdStr;
        C.setContainsError();
        return StringRef();
      }
      auto CanId = getCanonicalTargetID(*ArchStr, Features);
      return Args.MakeArgStringRef(CanId);
    };

    std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
    getConflictOffloadArchCombination(
        const std::set<StringRef> &GpuArchs) override {
      return getConflictTargetIDCombination(GpuArchs);
    }

    ActionBuilderReturnCode
    getDeviceDependences(OffloadAction::DeviceDependences &DA,
                         phases::ID CurPhase, phases::ID FinalPhase,
                         PhasesTy &Phases) override {
      if (!IsActive)
        return ABRT_Inactive;

      // amdgcn does not support linking of object files, therefore we skip
      // backend and assemble phases to output LLVM IR. Except for generating
      // non-relocatable device code, where we generate fat binary for device
      // code and pass to host in Backend phase.
      if (CudaDeviceActions.empty())
        return ABRT_Success;

      assert(((CurPhase == phases::Link && Relocatable) ||
              CudaDeviceActions.size() == GpuArchList.size()) &&
             "Expecting one action per GPU architecture.");
      assert(!CompileHostOnly &&
             "Not expecting HIP actions in host-only compilation.");

      bool ShouldLink = !EmitReloc || !*EmitReloc;

      if (!Relocatable && CurPhase == phases::Backend && !EmitLLVM &&
          !EmitAsm && ShouldLink) {
        // If we are in backend phase, we attempt to generate the fat binary.
        // We compile each arch to IR and use a link action to generate code
        // object containing ISA. Then we use a special "link" action to create
        // a fat binary containing all the code objects for different GPU's.
        // The fat binary is then an input to the host action.
        for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {
          if (C.getDriver().isUsingLTO(/*IsOffload=*/true)) {
            // When LTO is enabled, skip the backend and assemble phases and
            // use lld to link the bitcode.
            ActionList AL;
            AL.push_back(CudaDeviceActions[I]);
            // Create a link action to link device IR with device library
            // and generate ISA.
            CudaDeviceActions[I] =
                C.MakeAction<LinkJobAction>(AL, types::TY_Image);
          } else {
            // When LTO is not enabled, we follow the conventional
            // compiler phases, including backend and assemble phases.
            ActionList AL;
            Action *BackendAction = nullptr;
            if (ToolChains.front()->getTriple().isSPIRV()) {
              // Emit LLVM bitcode for SPIR-V targets. SPIR-V device tool chain
              // (HIPSPVToolChain) runs post-link LLVM IR passes.
              types::ID Output = Args.hasArg(options::OPT_S)
                                     ? types::TY_LLVM_IR
                                     : types::TY_LLVM_BC;
              BackendAction =
                  C.MakeAction<BackendJobAction>(CudaDeviceActions[I], Output);
            } else
              BackendAction = C.getDriver().ConstructPhaseAction(
                  C, Args, phases::Backend, CudaDeviceActions[I],
                  AssociatedOffloadKind);
            auto AssembleAction = C.getDriver().ConstructPhaseAction(
                C, Args, phases::Assemble, BackendAction,
                AssociatedOffloadKind);
            AL.push_back(AssembleAction);
            // Create a link action to link device IR with device library
            // and generate ISA.
            CudaDeviceActions[I] =
                C.MakeAction<LinkJobAction>(AL, types::TY_Image);
          }

          // OffloadingActionBuilder propagates device arch until an offload
          // action. Since the next action for creating fatbin does
          // not have device arch, whereas the above link action and its input
          // have device arch, an offload action is needed to stop the null
          // device arch of the next action being propagated to the above link
          // action.
          OffloadAction::DeviceDependences DDep;
          DDep.add(*CudaDeviceActions[I], *ToolChains.front(), GpuArchList[I],
                   AssociatedOffloadKind);
          CudaDeviceActions[I] = C.MakeAction<OffloadAction>(
              DDep, CudaDeviceActions[I]->getType());
        }

        if (!CompileDeviceOnly || !BundleOutput || *BundleOutput) {
          // Create HIP fat binary with a special "link" action.
          CudaFatBinary = C.MakeAction<LinkJobAction>(CudaDeviceActions,
                                                      types::TY_HIP_FATBIN);

          if (!CompileDeviceOnly) {
            DA.add(*CudaFatBinary, *ToolChains.front(), /*BoundArch=*/nullptr,
                   AssociatedOffloadKind);
            // Clear the fat binary, it is already a dependence to an host
            // action.
            CudaFatBinary = nullptr;
          }

          // Remove the CUDA actions as they are already connected to an host
          // action or fat binary.
          CudaDeviceActions.clear();
        }

        return CompileDeviceOnly ? ABRT_Ignore_Host : ABRT_Success;
      } else if (CurPhase == phases::Link) {
        if (!ShouldLink)
          return ABRT_Success;
        // Save CudaDeviceActions to DeviceLinkerInputs for each GPU subarch.
        // This happens to each device action originated from each input file.
        // Later on, device actions in DeviceLinkerInputs are used to create
        // device link actions in appendLinkDependences and the created device
        // link actions are passed to the offload action as device dependence.
        DeviceLinkerInputs.resize(CudaDeviceActions.size());
        auto LI = DeviceLinkerInputs.begin();
        for (auto *A : CudaDeviceActions) {
          LI->push_back(A);
          ++LI;
        }

        // We will pass the device action as a host dependence, so we don't
        // need to do anything else with them.
        CudaDeviceActions.clear();
        return CompileDeviceOnly ? ABRT_Ignore_Host : ABRT_Success;
      }

      // By default, we produce an action for each device arch.
      for (Action *&A : CudaDeviceActions)
        A = C.getDriver().ConstructPhaseAction(C, Args, CurPhase, A,
                                               AssociatedOffloadKind);

      if (CompileDeviceOnly && CurPhase == FinalPhase && BundleOutput &&
          *BundleOutput) {
        for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {
          OffloadAction::DeviceDependences DDep;
          DDep.add(*CudaDeviceActions[I], *ToolChains.front(), GpuArchList[I],
                   AssociatedOffloadKind);
          CudaDeviceActions[I] = C.MakeAction<OffloadAction>(
              DDep, CudaDeviceActions[I]->getType());
        }
        CudaFatBinary =
            C.MakeAction<OffloadBundlingJobAction>(CudaDeviceActions);
        CudaDeviceActions.clear();
      }

      return (CompileDeviceOnly &&
              (CurPhase == FinalPhase ||
               (!ShouldLink && CurPhase == phases::Assemble)))
                 ? ABRT_Ignore_Host
                 : ABRT_Success;
    }

    void appendLinkDeviceActions(ActionList &AL) override {
      if (DeviceLinkerInputs.size() == 0)
        return;

      assert(DeviceLinkerInputs.size() == GpuArchList.size() &&
             "Linker inputs and GPU arch list sizes do not match.");

      ActionList Actions;
      unsigned I = 0;
      // Append a new link action for each device.
      // Each entry in DeviceLinkerInputs corresponds to a GPU arch.
      for (auto &LI : DeviceLinkerInputs) {

        types::ID Output = Args.hasArg(options::OPT_emit_llvm)
                                   ? types::TY_LLVM_BC
                                   : types::TY_Image;

        auto *DeviceLinkAction = C.MakeAction<LinkJobAction>(LI, Output);
        // Linking all inputs for the current GPU arch.
        // LI contains all the inputs for the linker.
        OffloadAction::DeviceDependences DeviceLinkDeps;
        DeviceLinkDeps.add(*DeviceLinkAction, *ToolChains[0],
            GpuArchList[I], AssociatedOffloadKind);
        Actions.push_back(C.MakeAction<OffloadAction>(
            DeviceLinkDeps, DeviceLinkAction->getType()));
        ++I;
      }
      DeviceLinkerInputs.clear();

      // If emitting LLVM, do not generate final host/device compilation action
      if (Args.hasArg(options::OPT_emit_llvm)) {
          AL.append(Actions);
          return;
      }

      // Create a host object from all the device images by embedding them
      // in a fat binary for mixed host-device compilation. For device-only
      // compilation, creates a fat binary.
      OffloadAction::DeviceDependences DDeps;
      if (!CompileDeviceOnly || !BundleOutput || *BundleOutput) {
        auto *TopDeviceLinkAction = C.MakeAction<LinkJobAction>(
            Actions,
            CompileDeviceOnly ? types::TY_HIP_FATBIN : types::TY_Object);
        DDeps.add(*TopDeviceLinkAction, *ToolChains[0], nullptr,
                  AssociatedOffloadKind);
        // Offload the host object to the host linker.
        AL.push_back(
            C.MakeAction<OffloadAction>(DDeps, TopDeviceLinkAction->getType()));
      } else {
        AL.append(Actions);
      }
    }

    Action* appendLinkHostActions(ActionList &AL) override { return AL.back(); }

    void appendLinkDependences(OffloadAction::DeviceDependences &DA) override {}
  };

  /// OpenMP action builder. The host bitcode is passed to the device frontend
  /// and all the device linked images are passed to the host link phase.
  class OpenMPActionBuilder final : public DeviceActionBuilder {
    /// The OpenMP actions for the current input.
    ActionList OpenMPDeviceActions;

    /// The linker inputs obtained for each toolchain.
    SmallVector<ActionList, 8> DeviceLinkerInputs;

  public:
    OpenMPActionBuilder(Compilation &C, DerivedArgList &Args,
                        const Driver::InputList &Inputs,
                        OffloadingActionBuilder &OAB)
        : DeviceActionBuilder(C, Args, Inputs, Action::OFK_OpenMP, OAB) {}

    ActionBuilderReturnCode
    getDeviceDependences(OffloadAction::DeviceDependences &DA,
                         phases::ID CurPhase, phases::ID FinalPhase,
                         PhasesTy &Phases) override {
      if (OpenMPDeviceActions.empty())
        return ABRT_Inactive;

      // We should always have an action for each input.
      assert(OpenMPDeviceActions.size() == ToolChains.size() &&
             "Number of OpenMP actions and toolchains do not match.");

      // The host only depends on device action in the linking phase, when all
      // the device images have to be embedded in the host image.
      if (CurPhase == phases::Link) {
        assert(ToolChains.size() == DeviceLinkerInputs.size() &&
               "Toolchains and linker inputs sizes do not match.");
        auto LI = DeviceLinkerInputs.begin();
        for (auto *A : OpenMPDeviceActions) {
          LI->push_back(A);
          ++LI;
        }

        // We passed the device action as a host dependence, so we don't need to
        // do anything else with them.
        OpenMPDeviceActions.clear();
        return ABRT_Success;
      }

      // By default, we produce an action for each device arch.
      for (Action *&A : OpenMPDeviceActions)
        A = C.getDriver().ConstructPhaseAction(C, Args, CurPhase, A);

      return ABRT_Success;
    }

    ActionBuilderReturnCode addDeviceDependences(Action *HostAction) override {

      // If this is an input action replicate it for each OpenMP toolchain.
      if (auto *IA = dyn_cast<InputAction>(HostAction)) {
        OpenMPDeviceActions.clear();
        for (unsigned I = 0; I < ToolChains.size(); ++I)
          OpenMPDeviceActions.push_back(
              C.MakeAction<InputAction>(IA->getInputArg(), IA->getType()));
        return ABRT_Success;
      }

      // If this is an unbundling action use it as is for each OpenMP toolchain.
      if (auto *UA = dyn_cast<OffloadUnbundlingJobAction>(HostAction)) {
        OpenMPDeviceActions.clear();
        if (auto *IA = dyn_cast<InputAction>(UA->getInputs().back())) {
          std::string FileName = IA->getInputArg().getAsString(Args);
          // Check if the type of the file is the same as the action. Do not
          // unbundle it if it is not. Do not unbundle .so files, for example,
          // which are not object files.
          if (IA->getType() == types::TY_Object &&
              (!llvm::sys::path::has_extension(FileName) ||
               types::lookupTypeForExtension(
                   llvm::sys::path::extension(FileName).drop_front()) !=
                   types::TY_Object))
            return ABRT_Inactive;
        }
        for (unsigned I = 0; I < ToolChains.size(); ++I) {
          OpenMPDeviceActions.push_back(UA);
          UA->registerDependentActionInfo(
              ToolChains[I], /*BoundArch=*/StringRef(), Action::OFK_OpenMP);
        }
        return ABRT_Success;
      }

      // When generating code for OpenMP we use the host compile phase result as
      // a dependence to the device compile phase so that it can learn what
      // declarations should be emitted. However, this is not the only use for
      // the host action, so we prevent it from being collapsed.
      if (isa<CompileJobAction>(HostAction)) {
        HostAction->setCannotBeCollapsedWithNextDependentAction();
        assert(ToolChains.size() == OpenMPDeviceActions.size() &&
               "Toolchains and device action sizes do not match.");
        OffloadAction::HostDependence HDep(
            *HostAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
            /*BoundArch=*/nullptr, Action::OFK_OpenMP);
        auto TC = ToolChains.begin();
        for (Action *&A : OpenMPDeviceActions) {
          assert(isa<CompileJobAction>(A));
          OffloadAction::DeviceDependences DDep;
          DDep.add(*A, **TC, /*BoundArch=*/nullptr, Action::OFK_OpenMP);
          A = C.MakeAction<OffloadAction>(HDep, DDep);
          ++TC;
        }
      }
      return ABRT_Success;
    }

    void appendTopLevelActions(ActionList &AL) override {
      if (OpenMPDeviceActions.empty())
        return;

      // We should always have an action for each input.
      assert(OpenMPDeviceActions.size() == ToolChains.size() &&
             "Number of OpenMP actions and toolchains do not match.");

      // Append all device actions followed by the proper offload action.
      auto TI = ToolChains.begin();
      for (auto *A : OpenMPDeviceActions) {
        OffloadAction::DeviceDependences Dep;
        Dep.add(*A, **TI, /*BoundArch=*/nullptr, Action::OFK_OpenMP);
        AL.push_back(C.MakeAction<OffloadAction>(Dep, A->getType()));
        ++TI;
      }
      // We no longer need the action stored in this builder.
      OpenMPDeviceActions.clear();
    }

    void appendLinkDeviceActions(ActionList &AL) override {
      assert(ToolChains.size() == DeviceLinkerInputs.size() &&
             "Toolchains and linker inputs sizes do not match.");

      // Append a new link action for each device.
      auto TC = ToolChains.begin();
      for (auto &LI : DeviceLinkerInputs) {
        auto *DeviceLinkAction =
            C.MakeAction<LinkJobAction>(LI, types::TY_Image);
        OffloadAction::DeviceDependences DeviceLinkDeps;
        DeviceLinkDeps.add(*DeviceLinkAction, **TC, /*BoundArch=*/nullptr,
		        Action::OFK_OpenMP);
        AL.push_back(C.MakeAction<OffloadAction>(DeviceLinkDeps,
            DeviceLinkAction->getType()));
        ++TC;
      }
      DeviceLinkerInputs.clear();
    }

    Action* appendLinkHostActions(ActionList &AL) override {
      // Create wrapper bitcode from the result of device link actions and compile
      // it to an object which will be added to the host link command.
      auto *BC = C.MakeAction<OffloadWrapperJobAction>(AL, types::TY_LLVM_BC);
      auto *ASM = C.MakeAction<BackendJobAction>(BC, types::TY_PP_Asm);
      return C.MakeAction<AssembleJobAction>(ASM, types::TY_Object);
    }

    void appendLinkDependences(OffloadAction::DeviceDependences &DA) override {}

    void addDeviceLinkDependencies(OffloadDepsJobAction *DA) override {
      for (unsigned I = 0; I < ToolChains.size(); ++I) {
        // Register dependent toolchain.
        DA->registerDependentActionInfo(
            ToolChains[I], /*BoundArch=*/StringRef(), Action::OFK_OpenMP);

        if (!ToolChains[I]->getTriple().isSPIR()) {
          // Create object from the deps bitcode.
          auto *BA = C.MakeAction<BackendJobAction>(DA, types::TY_PP_Asm);
          auto *AA = C.MakeAction<AssembleJobAction>(BA, types::TY_Object);

          // Add deps object to linker inputs.
          DeviceLinkerInputs[I].push_back(AA);
        } else
          DeviceLinkerInputs[I].push_back(DA);
      }
    }

    bool initialize() override {
      // Get the OpenMP toolchains. If we don't get any, the action builder will
      // know there is nothing to do related to OpenMP offloading.
      auto OpenMPTCRange = C.getOffloadToolChains<Action::OFK_OpenMP>();
      for (auto TI = OpenMPTCRange.first, TE = OpenMPTCRange.second; TI != TE;
           ++TI)
        ToolChains.push_back(TI->second);

      DeviceLinkerInputs.resize(ToolChains.size());
      return false;
    }

    bool canUseBundlerUnbundler() const override {
      // OpenMP should use bundled files whenever possible.
      return true;
    }
  };

  /// SYCL action builder. The host bitcode is passed to the device frontend
  /// and all the device linked images are passed to the host link phase.
  /// SPIR related are wrapped before added to the fat binary
  class SYCLActionBuilder final : public DeviceActionBuilder {
    /// Flag to signal if the user requested device-only compilation.
    bool CompileDeviceOnly = false;

    /// Flag to signal if the user requested the device object to be wrapped.
    bool WrapDeviceOnlyBinary = false;

    /// Flag to signal if the user requested device code split.
    bool DeviceCodeSplit = false;

    /// List of offload device toolchain, bound arch needed to track for
    /// different binary constructions.
    /// POD to hold information about a SYCL device action.
    /// Each Action is bound to a <TC, arch> pair,
    /// we keep them together under a struct for clarity.
    struct DeviceTargetInfo {
      DeviceTargetInfo(const ToolChain *TC, const char *BA)
          : TC(TC), BoundArch(BA) {}

      const ToolChain *TC;
      const char *BoundArch;
    };
    SmallVector<DeviceTargetInfo, 4> SYCLTargetInfoList;

    /// The SYCL actions for the current input.
    /// One action per triple/boundarch.
    ActionList SYCLDeviceActions;

    /// The linker inputs obtained for each input/toolchain/arch.
    SmallVector<ActionList, 4> DeviceLinkerInputs;

    /// The SYCL link binary if it was generated for the current input.
    Action *SYCLLinkBinary = nullptr;

    /// Running list of SYCL actions specific for device linking.
    ActionList SYCLLinkBinaryList;

    /// List of SYCL Final Device binaries that should be unbundled as a final
    /// device binary and not further processed.
    SmallVector<std::pair<Action *, SmallVector<std::string, 4>>, 4>
        SYCLFinalDeviceList;

    /// SYCL ahead of time compilation inputs
    SmallVector<std::pair<llvm::Triple, const char *>, 8> SYCLAOTInputs;

    /// List of offload device triples as provided on the CLI.
    /// Does not track AOT binary inputs triples.
    SmallVector<llvm::Triple, 4> SYCLTripleList;

    /// Type of output file for FPGA device compilation.
    types::ID FPGAOutType = types::TY_FPGA_AOCX;

    /// List of objects to extract FPGA dependency info from
    ActionList FPGAObjectInputs;

    /// List of static archives to extract FPGA dependency info from
    ActionList FPGAArchiveInputs;

    /// List of AOCR based archives that contain BC members to use for
    /// providing symbols and properties.
    ActionList FPGAAOCArchives;

    // SYCLInstallation is needed in order to link SYCLDeviceLibs
    SYCLInstallationDetector SYCLInstallation;

    /// List of GPU architectures to use in this compilation with NVPTX/AMDGCN
    /// targets.
    SmallVector<std::pair<llvm::Triple, const char *>, 8> GpuArchList;

    /// Build the last steps for CUDA after all BC files have been linked.
    JobAction *finalizeNVPTXDependences(Action *Input, const llvm::Triple &TT) {
      auto *BA = C.getDriver().ConstructPhaseAction(
          C, Args, phases::Backend, Input, AssociatedOffloadKind);
      if (TT.getOS() != llvm::Triple::NVCL) {
        auto *AA = C.getDriver().ConstructPhaseAction(
            C, Args, phases::Assemble, BA, AssociatedOffloadKind);
        ActionList DeviceActions = {BA, AA};
        return C.MakeAction<LinkJobAction>(DeviceActions,
                                           types::TY_CUDA_FATBIN);
      }
      return cast<JobAction>(BA);
    }

    JobAction *finalizeAMDGCNDependences(Action *Input,
                                         const llvm::Triple &TT) {
      auto *BA = C.getDriver().ConstructPhaseAction(
          C, Args, phases::Backend, Input, AssociatedOffloadKind);

      auto *AA = C.getDriver().ConstructPhaseAction(C, Args, phases::Assemble,
                                                    BA, AssociatedOffloadKind);

      ActionList AL = {AA};
      Action *LinkAction = C.MakeAction<LinkJobAction>(AL, types::TY_Image);
      ActionList HIPActions = {LinkAction};
      JobAction *HIPFatBinary =
          C.MakeAction<LinkJobAction>(HIPActions, types::TY_HIP_FATBIN);
      return HIPFatBinary;
    }

    Action *ExternalCudaAction = nullptr;

  public:
    SYCLActionBuilder(Compilation &C, DerivedArgList &Args,
                      const Driver::InputList &Inputs,
                      OffloadingActionBuilder &OAB)
        : DeviceActionBuilder(C, Args, Inputs, Action::OFK_SYCL, OAB),
          SYCLInstallation(C.getDriver()) {}

    void withBoundArchForToolChain(const ToolChain *TC,
                                   llvm::function_ref<void(const char *)> Op) {
      for (auto &A : GpuArchList) {
        if (TC->getTriple() == A.first) {
          Op(A.second ? Args.MakeArgString(A.second) : nullptr);
          return;
        }
      }

      // no bound arch for this toolchain
      Op(nullptr);
    }

    void pushForeignAction(Action *A) override {
      // Accept a foreign action from the CudaActionBuilder for compiling CUDA
      // sources
      if (A->getOffloadingDeviceKind() == Action::OFK_Cuda)
        ExternalCudaAction = A;
    }

    ActionBuilderReturnCode
    getDeviceDependences(OffloadAction::DeviceDependences &DA,
                         phases::ID CurPhase, phases::ID FinalPhase,
                         PhasesTy &Phases) override {
      bool SYCLDeviceOnly = Args.hasArg(options::OPT_fsycl_device_only);
      if (CurPhase == phases::Preprocess) {
        // Do not perform the host compilation when doing preprocessing only
        // with -fsycl-device-only.
        bool IsPreprocessOnly =
            Args.getLastArg(options::OPT_E) ||
            Args.getLastArg(options::OPT__SLASH_EP, options::OPT__SLASH_P) ||
            Args.getLastArg(options::OPT_M, options::OPT_MM);
        if (IsPreprocessOnly) {
          for (auto TargetActionInfo :
               llvm::zip(SYCLDeviceActions, SYCLTargetInfoList)) {
            Action *&A = std::get<0>(TargetActionInfo);
            auto &TargetInfo = std::get<1>(TargetActionInfo);
            A = C.getDriver().ConstructPhaseAction(C, Args, CurPhase, A,
                                                   AssociatedOffloadKind);
            if (SYCLDeviceOnly)
              continue;
            // Add an additional compile action to generate the integration
            // header.
            Action *CompileAction =
                C.MakeAction<CompileJobAction>(A, types::TY_Nothing);
            DA.add(*CompileAction, *TargetInfo.TC, TargetInfo.BoundArch,
                   Action::OFK_SYCL);
          }
          return SYCLDeviceOnly ? ABRT_Ignore_Host : ABRT_Success;
        }
      }

      // Device compilation generates LLVM BC.
      if (CurPhase == phases::Compile && !SYCLTargetInfoList.empty()) {
        // TODO: handle stubfile handling when mix and matching programming
        // model.
        if (SYCLDeviceActions.empty())
          return ABRT_Success;

        Action *DeviceCompilerInput = nullptr;
        const DeviceTargetInfo &DevTarget = SYCLTargetInfoList.back();
        for (auto TargetActionInfo :
             llvm::zip(SYCLDeviceActions, SYCLTargetInfoList)) {
          Action *&A = std::get<0>(TargetActionInfo);
          auto &TargetInfo = std::get<1>(TargetActionInfo);
          types::ID OutputType = types::TY_LLVM_BC;
          if ((SYCLDeviceOnly || Args.hasArg(options::OPT_emit_llvm)) &&
              Args.hasArg(options::OPT_S))
            OutputType = types::TY_LLVM_IR;
          // Use of -fsycl-device-obj=spirv converts the original LLVM-IR
          // file to SPIR-V for later consumption.
          if ((SYCLDeviceOnly || FinalPhase != phases::Link) &&
              Args.getLastArgValue(options::OPT_fsycl_device_obj_EQ)
                  .equals_insensitive("spirv")) {
            auto *CompileAction =
                C.MakeAction<CompileJobAction>(A, types::TY_LLVM_BC);
            A = C.MakeAction<SPIRVTranslatorJobAction>(CompileAction,
                                                       types::TY_SPIRV);
            if (SYCLDeviceOnly)
              continue;
          } else {
            if (Args.hasArg(options::OPT_fsyntax_only))
              OutputType = types::TY_Nothing;
            A = C.MakeAction<CompileJobAction>(A, OutputType);
          }
          // Add any of the device linking steps when -fno-sycl-rdc is
          // specified. Device linking is only available for AOT at this
          // time.
          llvm::Triple TargetTriple = TargetInfo.TC->getTriple();
          if (tools::SYCL::shouldDoPerObjectFileLinking(C) &&
              TargetTriple.isSPIRAOT() && FinalPhase != phases::Link) {
            ActionList CAList;
            CAList.push_back(A);
            ActionList DeviceLinkActions;
            appendSYCLDeviceLink(CAList, TargetInfo.TC, DA, DeviceLinkActions,
                                 TargetInfo.BoundArch,
                                 /*AddOffloadAction=*/true);
            // The list of actions generated from appendSYCLDeviceLink is kept
            // in DeviceLinkActions.  Instead of adding the dependency on the
            // compiled device file, add the dependency against the compiled
            // device binary to be added to the resulting fat object.
            A = DeviceLinkActions.back();
          }
          DeviceCompilerInput = A;
        }
        DA.add(*DeviceCompilerInput, *DevTarget.TC, DevTarget.BoundArch,
               Action::OFK_SYCL);
        return SYCLDeviceOnly ? ABRT_Ignore_Host : ABRT_Success;
      }

      // Backend/Assemble actions are obsolete for the SYCL device side
      if (CurPhase == phases::Backend || CurPhase == phases::Assemble)
        return ABRT_Inactive;

      // The host only depends on device action in the linking phase, when all
      // the device images have to be embedded in the host image.
      if (CurPhase == phases::Link) {
        if (!SYCLDeviceActions.empty()) {
          assert(SYCLDeviceActions.size() == DeviceLinkerInputs.size() &&
                 "Device action and device linker inputs sizes do not match.");

          for (auto TargetAction :
               llvm::zip(DeviceLinkerInputs, SYCLDeviceActions)) {
            ActionList &LinkerList = std::get<0>(TargetAction);
            Action *A = std::get<1>(TargetAction);

            LinkerList.push_back(A);
          }
        }

        if (ExternalCudaAction) {
          assert(DeviceLinkerInputs.size() == 1 &&
                 "Number of SYCL actions and toolchains/boundarch pairs do not "
                 "match.");
          DeviceLinkerInputs[0].push_back(ExternalCudaAction);
          ExternalCudaAction = nullptr;
        }

        // With -fsycl-link-targets, we will take the unbundled binaries
        // for each device and link them together to a single binary that will
        // be used in a split compilation step.
        if (CompileDeviceOnly && !SYCLDeviceActions.empty()) {
          for (auto SDA : SYCLDeviceActions)
            SYCLLinkBinaryList.push_back(SDA);
          if (WrapDeviceOnlyBinary) {
            // -fsycl-link behavior does the following to the unbundled device
            // binaries:
            //   1) Link them together using llvm-link
            //   2) Pass the linked binary through sycl-post-link
            //   3) Translate final .bc file to .spv
            //   4) Wrap the binary with the offload wrapper which can be used
            //      by any compilation link step.
            auto *DeviceLinkAction = C.MakeAction<LinkJobAction>(
                SYCLLinkBinaryList, types::TY_Image);
            ActionList FullSYCLLinkBinaryList;
            bool SYCLDeviceLibLinked = false;
            FullSYCLLinkBinaryList.push_back(DeviceLinkAction);
            // If used without the FPGA target, -fsycl-link is used to wrap
            // device objects for future host link. Device libraries should
            // be linked by default to resolve any undefined reference.
            const auto *TC = ToolChains.front();
            llvm::Triple TT(TC->getTriple());
            if (TT.getSubArch() != llvm::Triple::SPIRSubArch_fpga) {
              SYCLDeviceLibLinked =
                  addSYCLDeviceLibs(TC, FullSYCLLinkBinaryList, true,
                                    C.getDefaultToolChain()
                                        .getTriple()
                                        .isWindowsMSVCEnvironment());
            }

            Action *FullDeviceLinkAction = nullptr;
            if (SYCLDeviceLibLinked)
              FullDeviceLinkAction = C.MakeAction<LinkJobAction>(
                  FullSYCLLinkBinaryList, types::TY_LLVM_BC);
            else
              FullDeviceLinkAction = DeviceLinkAction;
            auto *PostLinkAction = C.MakeAction<SYCLPostLinkJobAction>(
                FullDeviceLinkAction, types::TY_LLVM_BC,
                types::TY_Tempfiletable);
            PostLinkAction->setRTSetsSpecConstants(!TT.isSPIRAOT());
            auto *ExtractIRFilesAction = C.MakeAction<FileTableTformJobAction>(
                PostLinkAction, types::TY_Tempfilelist, types::TY_Tempfilelist);
            // single column w/o title fits TY_Tempfilelist format
            ExtractIRFilesAction->addExtractColumnTform(
                FileTableTformJobAction::COL_CODE, false /*drop titles*/);
            auto *TranslateAction = C.MakeAction<SPIRVTranslatorJobAction>(
                ExtractIRFilesAction, types::TY_Tempfilelist);

            ActionList TformInputs{PostLinkAction, TranslateAction};
            auto *ReplaceFilesAction = C.MakeAction<FileTableTformJobAction>(
                TformInputs, types::TY_Tempfiletable, types::TY_Tempfiletable);
            ReplaceFilesAction->addReplaceColumnTform(
                FileTableTformJobAction::COL_CODE,
                FileTableTformJobAction::COL_CODE);

            SYCLLinkBinary = C.MakeAction<OffloadWrapperJobAction>(
                ReplaceFilesAction, types::TY_Object);
          } else {
            auto *Link = C.MakeAction<LinkJobAction>(SYCLLinkBinaryList,
                                                         types::TY_Image);
            SYCLLinkBinary = C.MakeAction<SPIRVTranslatorJobAction>(
                Link, types::TY_Image);
          }

          // Remove the SYCL actions as they are already connected to an host
          // action or fat binary.
          SYCLDeviceActions.clear();
          // We avoid creating host action in device-only mode.
          return ABRT_Ignore_Host;
        }

        // We passed the device action as a host dependence, so we don't need to
        // do anything else with them.
        SYCLDeviceActions.clear();
        return ABRT_Success;
      }

      // By default, we produce an action for each device arch.
      auto TC = ToolChains.begin();
      for (Action *&A : SYCLDeviceActions) {
        if ((*TC)->getTriple().isNVPTX() && CurPhase >= phases::Backend) {
          // For CUDA, stop to emit LLVM IR so it can be linked later on.
          ++TC;
          continue;
        }

        A = C.getDriver().ConstructPhaseAction(C, Args, CurPhase, A,
                                               AssociatedOffloadKind);
        ++TC;
      }

      return ABRT_Success;
    }

    ActionBuilderReturnCode addDeviceDependences(Action *HostAction) override {

      // If this is an input action replicate it for each SYCL toolchain.
      if (auto *IA = dyn_cast<InputAction>(HostAction)) {
        SYCLDeviceActions.clear();

        // Options that are considered LinkerInput are not valid input actions
        // to the device tool chain.
        if (IA->getInputArg().getOption().hasFlag(options::LinkerInput))
          return ABRT_Inactive;

        std::string InputName = IA->getInputArg().getAsString(Args);
        // Objects will be consumed as part of the partial link step when
        // dealing with offload static libraries
        if (C.getDriver().getOffloadStaticLibSeen() &&
            IA->getType() == types::TY_Object && isObjectFile(InputName))
          return ABRT_Inactive;

        // Libraries are not processed in the SYCL toolchain
        if (IA->getType() == types::TY_Object && !isObjectFile(InputName))
          return ABRT_Inactive;

        for (auto &TargetInfo : SYCLTargetInfoList) {
          (void)TargetInfo;
          SYCLDeviceActions.push_back(
              C.MakeAction<InputAction>(IA->getInputArg(), IA->getType()));
        }
        return ABRT_Success;
      }

      // If this is an unbundling action use it as is for each SYCL toolchain.
      if (auto *UA = dyn_cast<OffloadUnbundlingJobAction>(HostAction)) {
        SYCLDeviceActions.clear();
        if (auto *IA = dyn_cast<InputAction>(UA->getInputs().back())) {
          // Options that are considered LinkerInput are not valid input actions
          // to the device tool chain.
          if (IA->getInputArg().getOption().hasFlag(options::LinkerInput))
            return ABRT_Inactive;

          std::string FileName = IA->getInputArg().getAsString(Args);
          // Check if the type of the file is the same as the action. Do not
          // unbundle it if it is not. Do not unbundle .so files, for example,
          // which are not object files.
          if (IA->getType() == types::TY_Object) {
            if (!isObjectFile(FileName))
              return ABRT_Inactive;
            // For SYCL device libraries, don't need to add them to
            // FPGAObjectInputs as there is no FPGA dep files inside.
            const auto *TC = ToolChains.front();
            if (TC->getTriple().getSubArch() ==
                    llvm::Triple::SPIRSubArch_fpga &&
                !IsSYCLDeviceLibObj(FileName, C.getDefaultToolChain()
                                                  .getTriple()
                                                  .isWindowsMSVCEnvironment()))
              FPGAObjectInputs.push_back(IA);
          }
        }
        // Create 1 device action per triple/bound arch
        for (auto &TargetInfo : SYCLTargetInfoList) {
          SYCLDeviceActions.push_back(UA);
          UA->registerDependentActionInfo(TargetInfo.TC, TargetInfo.BoundArch,
                                          Action::OFK_SYCL);
        }
        return ABRT_Success;
      }
      return ABRT_Success;
    }

    // Actions that can only be appended after all Inputs have been processed
    // occur here.  Not all offload actions are against single files.
    void appendTopLevelLinkAction(ActionList &AL) override {
      if (!SYCLLinkBinary)
        return;

      OffloadAction::DeviceDependences Dep;
      withBoundArchForToolChain(ToolChains.front(), [&](const char *BoundArch) {
        Dep.add(*SYCLLinkBinary, *ToolChains.front(), BoundArch,
                Action::OFK_SYCL);
      });
      AL.push_back(C.MakeAction<OffloadAction>(Dep, SYCLLinkBinary->getType()));
      SYCLLinkBinary = nullptr;
    }

    void appendTopLevelActions(ActionList &AL) override {
      // We should always have an action for each input.
      if (!SYCLDeviceActions.empty()) {
        assert(SYCLDeviceActions.size() == SYCLTargetInfoList.size() &&
               "Number of SYCL actions and toolchains/boundarch pairs do not "
               "match.");

        // Append all device actions followed by the proper offload action.
        for (auto TargetActionInfo :
             llvm::zip(SYCLDeviceActions, SYCLTargetInfoList)) {
          Action *A = std::get<0>(TargetActionInfo);
          DeviceTargetInfo &TargetInfo = std::get<1>(TargetActionInfo);

          OffloadAction::DeviceDependences Dep;
          Dep.add(*A, *TargetInfo.TC, TargetInfo.BoundArch, Action::OFK_SYCL);
          if (ExternalCudaAction) {
            assert(
                SYCLTargetInfoList.size() == 1 &&
                "Number of SYCL actions and toolchains/boundarch pairs do not "
                "match.");

            // Link with external CUDA action.
            ActionList LinkObjects;
            LinkObjects.push_back(
                C.MakeAction<OffloadAction>(Dep, A->getType()));
            LinkObjects.push_back(ExternalCudaAction);
            Action *DeviceLinkAction =
                C.MakeAction<LinkJobAction>(LinkObjects, types::TY_LLVM_BC);

            OffloadAction::DeviceDependences DDep;
            DDep.add(*DeviceLinkAction, *TargetInfo.TC, TargetInfo.BoundArch,
                     Action::OFK_SYCL);
            AL.push_back(C.MakeAction<OffloadAction>(DDep, A->getType()));

            ExternalCudaAction = nullptr;
          } else {
            AL.push_back(C.MakeAction<OffloadAction>(Dep, A->getType()));
          }
        }
        // We no longer need the action stored in this builder.
        SYCLDeviceActions.clear();
      }
    }

    // Performs device specific linking steps for the SYCL based toolchain.
    // This function is used for both the early AOT flow and the typical
    // offload device link flow.
    // When creating the standard offload device link flow during the link
    // phase, the ListIndex input provides an index against the
    // SYCLTargetInfoList. This is used to determine associated toolchain
    // information for the values being worked against to add the device link
    // steps. The generated device link steps are added via dependency
    // additions. For early AOT, ListIndex is the base device file that the
    // created device linking actions are performed against. The
    // DeviceLinkActions is used to hold the actions generated to be added to
    // the toolchain.
    void appendSYCLDeviceLink(const ActionList &ListIndex, const ToolChain *TC,
                              OffloadAction::DeviceDependences &DA,
                              ActionList &DeviceLinkActions,
                              const char *BoundArch,
                              bool AddOffloadAction = false) {
      auto addDeps = [&](Action *A, const ToolChain *TC,
                         const char *BoundArch) {
        if (AddOffloadAction) {
          OffloadAction::DeviceDependences Deps;
          Deps.add(*A, *TC, BoundArch, Action::OFK_SYCL);
          DeviceLinkActions.push_back(
              C.MakeAction<OffloadAction>(Deps, A->getType()));
        } else {
          DA.add(*A, *TC, BoundArch, Action::OFK_SYCL);
        }
      };

      // List of device specific libraries to be fed into llvm-link.
      ActionList SYCLDeviceLibs;

      // List of device specific library 'objects' (FPGA AOCO libraries) that
      // are fed directly to the FPGA offline compiler.
      ActionList FPGADeviceLibObjects;

      // List of device objects that go through the device link step.
      ActionList LinkObjects;

      // List of FPGA AOC specific device objects/archives.
      ActionList FPGAAOCDevices;
      auto TargetTriple = TC->getTriple();
      auto IsNVPTX = TargetTriple.isNVPTX();
      auto IsAMDGCN = TargetTriple.isAMDGCN();
      auto IsSPIR = TargetTriple.isSPIR();
      bool IsSpirvAOT = TargetTriple.isSPIRAOT();
      const bool IsSYCLNativeCPU =
          TC->getAuxTriple() &&
          driver::isSYCLNativeCPU(TargetTriple, *TC->getAuxTriple());
      for (const auto &Input : ListIndex) {
        if (TargetTriple.getSubArch() == llvm::Triple::SPIRSubArch_fpga &&
            types::isFPGA(Input->getType())) {
          assert(BoundArch == nullptr &&
                 "fpga triple bounded arch not nullptr");
          // FPGA aoco does not go through the link, everything else does.
          if (Input->getType() == types::TY_FPGA_AOCO) {
            FPGADeviceLibObjects.push_back(Input);
            continue;
          }
          // FPGA aocr/aocx does not go through the link and is passed
          // directly to the backend compilation step (aocr) or wrapper (aocx)
          if (Args.hasArg(options::OPT_fintelfpga)) {
            if (Input->getType() == types::TY_FPGA_AOCR ||
                Input->getType() == types::TY_FPGA_AOCR_EMU ||
                Input->getType() == types::TY_FPGA_AOCX)
              // Save the AOCR device items.  These will be processed along
              // with the FPGAAOCArchives.
              FPGAAOCDevices.push_back(Input);
            else
              llvm_unreachable("Unexpected FPGA input type.");
          }
          continue;
        } else if (!types::isFPGA(Input->getType())) {
          // No need for any conversion if we are coming in from the
          // clang-offload-deps or regular compilation path.
          if (IsNVPTX || IsAMDGCN || ContainsOffloadDepsAction(Input) ||
              ContainsCompileOrAssembleAction(Input)) {
            LinkObjects.push_back(Input);
            continue;
          }
          // Any objects or lists of objects that come in from the unbundling
          // step can either be LLVM-IR or SPIR-V based.  Send these through
          // the spirv-to-ir-wrapper to convert to LLVM-IR to be properly
          // processed during the device link.
          Action *ConvertSPIRVAction = C.MakeAction<SpirvToIrWrapperJobAction>(
              Input, Input->getType() == types::TY_Tempfilelist
                         ? types::TY_Tempfilelist
                         : types::TY_LLVM_BC);
          LinkObjects.push_back(ConvertSPIRVAction);
        }
      }
      // Process AOCR/AOCR_EMU
      if (FPGAAOCDevices.size()) {
        assert(FPGAAOCDevices.size() == FPGAAOCArchives.size() &&
               "Unexpected number of AOC binaries");
        // Generate AOCX/AOCR
        // Input is the unbundled device binary.  Perform an additional
        // unbundle against the input file associated to grab the wrapped
        // device binary.
        for (auto AOCRItem : llvm::zip(FPGAAOCArchives, FPGAAOCDevices)) {
          Action *Archive = std::get<0>(AOCRItem);
          Action *Device = std::get<1>(AOCRItem);

          auto *UnbundleAction = C.MakeAction<OffloadUnbundlingJobAction>(
              Archive, types::TY_Tempfilelist);
          UnbundleAction->registerDependentActionInfo(TC, /*BoundArch=*/"",
                                                      Action::OFK_SYCL);
          auto *RenameUnbundle = C.MakeAction<FileTableTformJobAction>(
              UnbundleAction, types::TY_Tempfilelist, types::TY_Tempfilelist);
          RenameUnbundle->addRenameColumnTform(
              FileTableTformJobAction::COL_ZERO,
              FileTableTformJobAction::COL_SYM_AND_PROPS);

          // Wrap the unbundled device binary along with the additional
          // .bc file that contains the Symbols and Properties
          if (Device->getType() == types::TY_FPGA_AOCX) {
            auto *RenameAction = C.MakeAction<FileTableTformJobAction>(
                Device, types::TY_Tempfilelist, types::TY_Tempfilelist);
            RenameAction->addRenameColumnTform(
                FileTableTformJobAction::COL_ZERO,
                FileTableTformJobAction::COL_CODE);
            ActionList WrapperItems({RenameAction, RenameUnbundle});
            auto *DeviceWrappingAction = C.MakeAction<OffloadWrapperJobAction>(
                WrapperItems, types::TY_Object);
            addDeps(DeviceWrappingAction, TC, BoundArch);
          } else {
            auto *FPGAAOTAction =
                C.MakeAction<BackendCompileJobAction>(Device, FPGAOutType);
            auto *RenameAction = C.MakeAction<FileTableTformJobAction>(
                FPGAAOTAction, types::TY_Tempfilelist, types::TY_Tempfilelist);
            RenameAction->addRenameColumnTform(
                FileTableTformJobAction::COL_ZERO,
                FileTableTformJobAction::COL_CODE);
            ActionList WrapperItems({RenameAction, RenameUnbundle});
            auto *DeviceWrappingAction = C.MakeAction<OffloadWrapperJobAction>(
                WrapperItems, types::TY_Object);

            Action *DeviceAction = DeviceWrappingAction;
            if (Args.hasArg(options::OPT_fsycl_link_EQ)) {
              if (auto *OWA = dyn_cast<OffloadWrapperJobAction>(DeviceAction))
                OWA->setCompileStep(false);
              ActionList BundlingActions;
              BundlingActions.push_back(DeviceWrappingAction);

              // Wrap and compile the wrapped device device binary.  This will
              // be used later when consumed as the input .bc file to retain
              // the symbols and properties associated.
              DeviceAction = C.MakeAction<OffloadWrapperJobAction>(
                  BundlingActions, types::TY_Object);
              if (auto *OWA = dyn_cast<OffloadWrapperJobAction>(DeviceAction))
                OWA->setOffloadKind(Action::OFK_Host);
              Action *CompiledDeviceAction =
                  C.MakeAction<OffloadWrapperJobAction>(WrapperItems,
                                                        types::TY_Object);
              addDeps(CompiledDeviceAction, TC, nullptr);
            }
            addDeps(DeviceAction, TC, nullptr);
          }
        }
      }
      for (const auto &A : SYCLFinalDeviceList) {
        // Given the list of archives that have final device binaries, take
        // those archives and unbundle all of the devices seen.  These will
        // be added to the final host link with no additional processing.
        // Gather the targets to unbundle.
        auto Input(A.first);
        for (StringRef TargetString : A.second) {
          // Unbundle
          types::ID InputType = Input->getType();
          if (InputType == types::TY_Archive)
            InputType = types::TY_Tempfilelist;
          auto *UA = C.MakeAction<OffloadUnbundlingJobAction>(Input, InputType);
          UA->registerDependentActionInfo(TC, /*BoundArch=*/"",
                                          Action::OFK_SYCL);
          UA->setTargetString(TargetString.str());

          // Add lists to the final link.
          addDeps(UA, TC, "");
        }
      }
      if (!LinkObjects.empty()) {
        // The linkage actions subgraph leading to the offload wrapper.
        // [cond] Means incoming/outgoing dependence is created only when cond
        //        is true. A function of:
        //   n - target is NVPTX/AMDGCN
        //   a - SPIRV AOT compilation is requested
        //   s - device code split requested
        //   r - relocatable device code is requested
        //   f - link object output type is TY_Tempfilelist (fat archive)
        //   e - Embedded IR for fusion (-fsycl-embed-ir) was requested
        //       and target is NVPTX.
        //   * - "all other cases"
        //     - no condition means output/input is "always" present
        // First symbol indicates output/input type
        //   . - single file output (TY_SPIRV, TY_LLVM_BC,...)
        //   - - TY_Tempfilelist
        //   + - TY_Tempfiletable
        //
        //                   .-----------------.
        //                   |Link(LinkObjects)|
        //                   .-----------------.
        //                ----[-!rf]   [*]
        //               [-!rf]         |
        //         .-------------.      |
        //         | llvm-foreach|      |
        //         .-------------.      |
        //               [.]            |
        //                |             |
        //                |             |
        //         .---------------------------------------.
        //         |               PostLink                |[+e]----------------
        //         .---------------------------------------.                   |
        //                           [+*]                [+]                   |
        //                             |                  |                    |
        //                             |                  |                    |
        //                             |---------         |                    |
        //                             |        |         |                    |
        //                             |        |         |                    |
        //                             |      [+!rf]      |                    |
        //                             |  .-------------. |                    |
        //                             |  | llvm-foreach| |                    |
        //                             |  .-------------. |                    |
        //                             |        |         |                    |
        //                            [+*]    [+!rf]      |                    |
        //                      .-----------------.       |                    |
        //                      | FileTableTform  |       |                    |
        //                      | (extract "Code")|       |                    |
        //                      .-----------------.       |                    |
        //                              [-]               |-----------         |
        //           --------------------|                           |         |
        //           |                   |                           |         |
        //           |                   |-----------------          |         |
        //           |                   |                |          |         |
        //           |                   |               [-!rf]      |         |
        //           |                   |         .--------------.  |         |
        //           |                   |         |FileTableTform|  |         |
        //           |                   |         |   (merge)    |  |         |
        //           |                   |         .--------------.  |         |
        //           |                   |               [-]         |-------  |
        //           |                   |                |          |      |  |
        //           |                   |                |    ------|      |  |
        //           |                   |        --------|    |            |  |
        //          [.]                 [-*]   [-!rf]        [+!rf]         |  |
        //   .---------------.  .-------------------. .--------------.      |  |
        //   | finalizeNVPTX  | |  SPIRVTranslator  | |FileTableTform|      |  |
        //   | finalizeAMDGCN | |                   | |   (merge)    |      |  |
        //   .---------------.  .-------------------. . -------------.      |  |
        //          [.]             [-as]      [-!a]         |              |  |
        //           |                |          |           |              |  |
        //           |              [-s]         |           |              |  |
        //           |       .----------------.  |           |              |  |
        //           |       | BackendCompile |  |           |              |  |
        //           |       .----------------.  |     ------|              |  |
        //           |              [-s]         |     |                    |  |
        //           |                |          |     |                    |  |
        //           |              [-a]      [-!a]  [-!rf]                 |  |
        //           |              .--------------------.                  |  |
        //           -----------[-n]|   FileTableTform   |[+*]--------------|  |
        //                          |  (replace "Code")  |                     |
        //                          .--------------------.                     |
        //                                      |      -------------------------
        //                                    [+*]     | [+e]
        //         .--------------------------------------.
        //         |            OffloadWrapper            |
        //         .--------------------------------------.
        //
        ActionList FullLinkObjects;
        bool IsRDC = !tools::SYCL::shouldDoPerObjectFileLinking(C);
        if (IsRDC) {
          Action *DeviceLinkAction =
              C.MakeAction<LinkJobAction>(LinkObjects, types::TY_LLVM_BC);
          FullLinkObjects.push_back(DeviceLinkAction);
        } else
          FullLinkObjects = LinkObjects;

        // FIXME: Link all wrapper and fallback device libraries as default,
        // When spv online link is supported by all backends, the fallback
        // device libraries are only needed when current toolchain is using
        // AOT compilation.
        bool SYCLDeviceLibLinked = false;
        if (IsSPIR || IsNVPTX) {
          bool UseJitLink =
              IsSPIR &&
              Args.hasFlag(options::OPT_fsycl_device_lib_jit_link,
                           options::OPT_fno_sycl_device_lib_jit_link, false);
          bool UseAOTLink = IsSPIR && (IsSpirvAOT || !UseJitLink);
          SYCLDeviceLibLinked = addSYCLDeviceLibs(
              TC, SYCLDeviceLibs, UseAOTLink,
              C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment());
        }
        if (IsSYCLNativeCPU) {
          SYCLDeviceLibLinked |= addSYCLNativeCPULibs(TC, SYCLDeviceLibs);
        }
        JobAction *LinkSYCLLibs =
            C.MakeAction<LinkJobAction>(SYCLDeviceLibs, types::TY_LLVM_BC);
        for (Action *FullLinkObject : FullLinkObjects) {
          if (FullLinkObject->getKind() ==
              clang::driver::Action::OffloadDepsJobClass)
            continue;
          Action *FullDeviceLinkAction = nullptr;
          ActionList WrapperInputs;

          if (SYCLDeviceLibLinked) {
            if (IsRDC) {
              // First object has to be non-DeviceLib for only-needed to be
              // passed.
              SYCLDeviceLibs.insert(SYCLDeviceLibs.begin(), FullLinkObject);
              FullDeviceLinkAction = C.MakeAction<LinkJobAction>(
                  SYCLDeviceLibs, types::TY_LLVM_BC);
            } else {
              FullDeviceLinkAction = FullLinkObject;

              ActionList DeviceCodeAndSYCLLibs{FullDeviceLinkAction,
                                               LinkSYCLLibs};
              JobAction *LinkDeviceCode = C.MakeAction<LinkJobAction>(
                  DeviceCodeAndSYCLLibs, types::TY_LLVM_BC);

              if (FullDeviceLinkAction->getType() == types::TY_Tempfilelist) {
                // If our compiler input outputs a temp file list (eg. fat
                // static archive), we need to link the device code against
                // each entry in the static archive.
                auto *ParallelLinkDeviceCode =
                    C.MakeAction<ForEachWrappingAction>(
                        cast<JobAction>(FullDeviceLinkAction), LinkDeviceCode);
                // The SYCL device library action tree should not be
                // for-eached, it only needs to happen once total. The
                // for-each action should start linking device code with the
                // device libraries.
                std::function<void(const Action *)> traverseActionTree =
                    [&](const Action *Act) {
                      ParallelLinkDeviceCode->addSerialAction(Act);
                      for (const auto &Input : Act->getInputs()) {
                        traverseActionTree(Input);
                      }
                    };
                traverseActionTree(LinkSYCLLibs);
                ActionList TformInputs{FullDeviceLinkAction,
                                       ParallelLinkDeviceCode};
                auto *ReplaceFilesAction =
                    C.MakeAction<FileTableTformJobAction>(
                        TformInputs, types::TY_Tempfilelist,
                        types::TY_Tempfilelist);
                ReplaceFilesAction->addReplaceColumnTform(
                    FileTableTformJobAction::COL_ZERO,
                    FileTableTformJobAction::COL_ZERO);
                ReplaceFilesAction->addExtractColumnTform(
                    FileTableTformJobAction::COL_ZERO, false /*drop titles*/);
                FullDeviceLinkAction = ReplaceFilesAction;
              } else {
                // If our compiler input is singular, just do a single link.
                FullDeviceLinkAction = LinkDeviceCode;
              }
            }
          } else
            FullDeviceLinkAction = FullLinkObject;

          // reflects whether current target is ahead-of-time and can't
          // support runtime setting of specialization constants
          bool IsAOT = IsNVPTX || IsAMDGCN || IsSpirvAOT || IsSYCLNativeCPU;

          // post link is not optional - even if not splitting, always need to
          // process specialization constants
          types::ID PostLinkOutType = IsSPIR || IsSYCLNativeCPU
                                          ? types::TY_Tempfiletable
                                          : types::TY_LLVM_BC;
          auto createPostLinkAction = [&]() {
            // For SPIR-V targets, force TY_Tempfiletable.
            auto TypedPostLinkAction = C.MakeAction<SYCLPostLinkJobAction>(
                FullDeviceLinkAction, PostLinkOutType, types::TY_Tempfiletable);
            TypedPostLinkAction->setRTSetsSpecConstants(!IsAOT);
            return TypedPostLinkAction;
          };
          Action *PostLinkAction = createPostLinkAction();
          if (IsSYCLNativeCPU) {
            // for SYCL Native CPU, we just take the linked device
            // modules, lower them to an object file , and link it to the host
            // object file.
            auto *BackendAct = C.MakeAction<BackendJobAction>(
                FullDeviceLinkAction, types::TY_PP_Asm);
            auto *AsmAct =
                C.MakeAction<AssembleJobAction>(BackendAct, types::TY_Object);
            DA.add(*AsmAct, *TC, BoundArch, Action::OFK_SYCL);
            auto *DeviceWrappingAction = C.MakeAction<OffloadWrapperJobAction>(
                PostLinkAction, types::TY_Object);
            DA.add(*DeviceWrappingAction, *TC, BoundArch, Action::OFK_SYCL);
            continue;
          }
          if ((IsNVPTX || IsAMDGCN) &&
              Args.hasArg(options::OPT_fsycl_embed_ir)) {
            // When compiling for Nvidia/AMD devices and the user requested the
            // IR to be embedded in the application (via option), run the output
            // of sycl-post-link (filetable referencing LLVM Bitcode + symbols)
            // through the offload wrapper and link the resulting object to the
            // application.
            auto *WrapBitcodeAction = C.MakeAction<OffloadWrapperJobAction>(
                PostLinkAction, types::TY_Object, true);
            addDeps(WrapBitcodeAction, TC, BoundArch);
          }
          bool NoRDCFatStaticArchive =
              !IsRDC &&
              FullDeviceLinkAction->getType() == types::TY_Tempfilelist;
          if (NoRDCFatStaticArchive)
            PostLinkAction = C.MakeAction<ForEachWrappingAction>(
                cast<JobAction>(FullDeviceLinkAction),
                cast<JobAction>(PostLinkAction));

          auto createExtractIRFilesAction = [&]() {
            auto *TypedExtractIRFilesAction =
                C.MakeAction<FileTableTformJobAction>(
                    PostLinkAction,
                    IsSPIR ? types::TY_Tempfilelist : PostLinkAction->getType(),
                    types::TY_Tempfilelist);
            // single column w/o title fits TY_Tempfilelist format
            TypedExtractIRFilesAction->addExtractColumnTform(
                FileTableTformJobAction::COL_CODE, false /*drop titles*/);
            return TypedExtractIRFilesAction;
          };

          Action *ExtractIRFilesAction = createExtractIRFilesAction();

          if (IsNVPTX || IsAMDGCN) {
            JobAction *FinAction =
                IsNVPTX ? finalizeNVPTXDependences(ExtractIRFilesAction,
                                                   TC->getTriple())
                        : finalizeAMDGCNDependences(ExtractIRFilesAction,
                                                    TC->getTriple());
            auto *ForEachWrapping = C.MakeAction<ForEachWrappingAction>(
                cast<JobAction>(ExtractIRFilesAction), FinAction);

            ActionList TformInputs{PostLinkAction, ForEachWrapping};
            auto *ReplaceFilesAction = C.MakeAction<FileTableTformJobAction>(
                TformInputs, types::TY_Tempfiletable, types::TY_Tempfiletable);
            ReplaceFilesAction->addReplaceColumnTform(
                FileTableTformJobAction::COL_CODE,
                FileTableTformJobAction::COL_CODE);

            WrapperInputs.push_back(ReplaceFilesAction);
          } else {
            if (NoRDCFatStaticArchive) {
              ExtractIRFilesAction = C.MakeAction<ForEachWrappingAction>(
                  cast<JobAction>(FullDeviceLinkAction),
                  cast<JobAction>(ExtractIRFilesAction));

              auto *MergeAllTablesIntoOne =
                  C.MakeAction<FileTableTformJobAction>(ExtractIRFilesAction,
                                                        types::TY_Tempfilelist,
                                                        types::TY_Tempfilelist);
              MergeAllTablesIntoOne->addMergeTform(
                  FileTableTformJobAction::COL_ZERO);
              ExtractIRFilesAction = MergeAllTablesIntoOne;
            }
            // For SPIRV-based targets - translate to SPIRV then optionally
            // compile ahead-of-time to native architecture
            Action *BuildCodeAction = C.MakeAction<SPIRVTranslatorJobAction>(
                ExtractIRFilesAction, types::TY_Tempfilelist);

            // After the Link, wrap the files before the final host link
            if (IsAOT) {
              types::ID OutType = types::TY_Tempfilelist;
              if (!DeviceCodeSplit) {
                OutType = (TargetTriple.getSubArch() ==
                           llvm::Triple::SPIRSubArch_fpga)
                              ? FPGAOutType
                              : types::TY_Image;
              }
              // Do the additional Ahead of Time compilation when the specific
              // triple calls for it (provided a valid subarch).
              ActionList BEInputs;
              BEInputs.push_back(BuildCodeAction);
              auto unbundleAdd = [&](Action *A, types::ID T) {
                ActionList AL;
                AL.push_back(A);
                Action *UnbundleAction =
                    C.MakeAction<OffloadUnbundlingJobAction>(AL, T);
                BEInputs.push_back(UnbundleAction);
              };
              // Send any known objects/archives through the unbundler to grab
              // the dependency file associated.  This is only done for
              // -fintelfpga.
              for (Action *A : FPGAObjectInputs)
                unbundleAdd(A, types::TY_FPGA_Dependencies);
              for (Action *A : FPGAArchiveInputs)
                unbundleAdd(A, types::TY_FPGA_Dependencies_List);
              for (const auto &A : FPGADeviceLibObjects)
                BEInputs.push_back(A);
              BuildCodeAction =
                  C.MakeAction<BackendCompileJobAction>(BEInputs, OutType);
            }
            if (NoRDCFatStaticArchive) {
              auto *MergeAllTablesIntoOne =
                  C.MakeAction<FileTableTformJobAction>(PostLinkAction,
                                                        types::TY_Tempfilelist,
                                                        types::TY_Tempfilelist);
              MergeAllTablesIntoOne->addMergeTform(
                  FileTableTformJobAction::COL_ZERO);
              PostLinkAction = MergeAllTablesIntoOne;
            }
            ActionList TformInputs{PostLinkAction, BuildCodeAction};
            auto *ReplaceFilesAction = C.MakeAction<FileTableTformJobAction>(
                TformInputs, types::TY_Tempfiletable, types::TY_Tempfiletable);
            ReplaceFilesAction->addReplaceColumnTform(
                FileTableTformJobAction::COL_CODE,
                FileTableTformJobAction::COL_CODE);
            WrapperInputs.push_back(ReplaceFilesAction);
          }

          // After the Link, wrap the files before the final host link
          // Add the unbundled wrapped AOC device binary to the wrapper
          // call.
          auto *DeviceWrappingAction = C.MakeAction<OffloadWrapperJobAction>(
              WrapperInputs, types::TY_Object);

          if (IsSpirvAOT) {
            // For FPGA with -fsycl-link, we need to bundle the output.
            if (TargetTriple.getSubArch() == llvm::Triple::SPIRSubArch_fpga) {
              Action *DeviceAction = DeviceWrappingAction;
              if (Args.hasArg(options::OPT_fsycl_link_EQ)) {
                // We do not want to compile the wrapped binary before the link.
                if (auto *OWA = dyn_cast<OffloadWrapperJobAction>(DeviceAction))
                  OWA->setCompileStep(false);
                ActionList BundlingActions;
                BundlingActions.push_back(DeviceWrappingAction);

                // Wrap and compile the wrapped device device binary.  This will
                // be used later when consumed as the input .bc file to retain
                // the symbols and properties associated.
                DeviceAction = C.MakeAction<OffloadWrapperJobAction>(
                    BundlingActions, types::TY_Object);
                if (auto *OWA = dyn_cast<OffloadWrapperJobAction>(DeviceAction))
                  OWA->setOffloadKind(Action::OFK_Host);
                Action *CompiledDeviceAction =
                    C.MakeAction<OffloadWrapperJobAction>(WrapperInputs,
                                                          types::TY_Object);
                addDeps(CompiledDeviceAction, TC, nullptr);
              }
              addDeps(DeviceAction, TC, nullptr);
            } else {
              bool AddBA =
                  (TargetTriple.getSubArch() == llvm::Triple::SPIRSubArch_gen &&
                   BoundArch != nullptr);
              addDeps(DeviceWrappingAction, TC, AddBA ? BoundArch : nullptr);
            }
          } else {
            withBoundArchForToolChain(TC, [&](const char *BoundArch) {
              addDeps(DeviceWrappingAction, TC, BoundArch);
            });
          }
        }
      }
    }

    bool addSYCLNativeCPULibs(const ToolChain *TC,
                              ActionList &DeviceLinkObjects) {
      std::string LibSpirvFile;
      if (Args.hasArg(options::OPT_fsycl_libspirv_path_EQ)) {
        auto ProvidedPath =
            Args.getLastArgValue(options::OPT_fsycl_libspirv_path_EQ).str();
        if (llvm::sys::fs::exists(ProvidedPath))
          LibSpirvFile = ProvidedPath;
      } else {
        SmallVector<StringRef, 8> LibraryPaths;

        // Expected path w/out install.
        SmallString<256> WithoutInstallPath(C.getDriver().ResourceDir);
        llvm::sys::path::append(WithoutInstallPath, Twine("../../clc"));
        LibraryPaths.emplace_back(WithoutInstallPath.c_str());

        // Expected path w/ install.
        SmallString<256> WithInstallPath(C.getDriver().ResourceDir);
        llvm::sys::path::append(WithInstallPath, Twine("../../../share/clc"));
        LibraryPaths.emplace_back(WithInstallPath.c_str());

        // Select libclc variant based on target triple
        std::string LibSpirvTargetName = "builtins.link.libspirv-";
        LibSpirvTargetName.append(TC->getTripleString() + ".bc");

        for (StringRef LibraryPath : LibraryPaths) {
          SmallString<128> LibSpirvTargetFile(LibraryPath);
          llvm::sys::path::append(LibSpirvTargetFile, LibSpirvTargetName);
          if (llvm::sys::fs::exists(LibSpirvTargetFile) ||
              Args.hasArg(options::OPT__HASH_HASH_HASH)) {
            LibSpirvFile = std::string(LibSpirvTargetFile.str());
            break;
          }
        }
      }

      if (!LibSpirvFile.empty()) {
        Arg *LibClcInputArg = MakeInputArg(Args, C.getDriver().getOpts(),
                                           Args.MakeArgString(LibSpirvFile));
        auto *SYCLLibClcInputAction =
            C.MakeAction<InputAction>(*LibClcInputArg, types::TY_LLVM_BC);
        DeviceLinkObjects.push_back(SYCLLibClcInputAction);
        return true;
      }
      return false;
    }

    bool addSYCLDeviceLibs(const ToolChain *TC, ActionList &DeviceLinkObjects,
                           bool isSpirvAOT, bool isMSVCEnv) {
      int NumOfDeviceLibLinked = 0;
      SmallVector<SmallString<128>, 4> LibLocCandidates;
      SYCLInstallation.getSYCLDeviceLibPath(LibLocCandidates);

      SmallVector<std::string, 8> DeviceLibraries;
      DeviceLibraries =
          tools::SYCL::getDeviceLibraries(C, TC->getTriple(), isSpirvAOT);

      for (const auto &DeviceLib : DeviceLibraries) {
        bool LibLocSelected = false;
        for (const auto &LLCandidate : LibLocCandidates) {
          if (LibLocSelected)
            break;
          SmallString<128> LibName(LLCandidate);
          llvm::sys::path::append(LibName, DeviceLib);
          if (llvm::sys::fs::exists(LibName)) {
            ++NumOfDeviceLibLinked;
            Arg *InputArg = MakeInputArg(Args, C.getDriver().getOpts(),
                                         Args.MakeArgString(LibName));
            auto *SYCLDeviceLibsInputAction =
                C.MakeAction<InputAction>(*InputArg, types::TY_Object);
            auto *SYCLDeviceLibsUnbundleAction =
                C.MakeAction<OffloadUnbundlingJobAction>(
                    SYCLDeviceLibsInputAction);

            // We are using BoundArch="" here since the NVPTX bundles in
            // the devicelib .o files do not contain any arch information
            SYCLDeviceLibsUnbundleAction->registerDependentActionInfo(
                TC, /*BoundArch=*/"", Action::OFK_SYCL);
            OffloadAction::DeviceDependences Dep;
            Dep.add(*SYCLDeviceLibsUnbundleAction, *TC, /*BoundArch=*/"",
                    Action::OFK_SYCL);
            auto *SYCLDeviceLibsDependenciesAction =
                C.MakeAction<OffloadAction>(
                    Dep, SYCLDeviceLibsUnbundleAction->getType());

            DeviceLinkObjects.push_back(SYCLDeviceLibsDependenciesAction);
            if (!LibLocSelected)
              LibLocSelected = !LibLocSelected;
          }
        }
      }

      // For NVPTX backend we need to also link libclc and CUDA libdevice
      // at the same stage that we link all of the unbundled SYCL libdevice
      // objects together.
      if (TC->getTriple().isNVPTX() && NumOfDeviceLibLinked) {
        std::string LibSpirvFile;
        if (Args.hasArg(options::OPT_fsycl_libspirv_path_EQ)) {
          auto ProvidedPath =
              Args.getLastArgValue(options::OPT_fsycl_libspirv_path_EQ).str();
          if (llvm::sys::fs::exists(ProvidedPath))
            LibSpirvFile = ProvidedPath;
        } else {
          SmallVector<StringRef, 8> LibraryPaths;

          // Expected path w/out install.
          SmallString<256> WithoutInstallPath(C.getDriver().ResourceDir);
          llvm::sys::path::append(WithoutInstallPath, Twine("../../clc"));
          LibraryPaths.emplace_back(WithoutInstallPath.c_str());

          // Expected path w/ install.
          SmallString<256> WithInstallPath(C.getDriver().ResourceDir);
          llvm::sys::path::append(WithInstallPath, Twine("../../../share/clc"));
          LibraryPaths.emplace_back(WithInstallPath.c_str());

          // Select remangled libclc variant
          std::string LibSpirvTargetName =
              (TC->getAuxTriple()->isOSWindows())
                  ? "remangled-l32-signed_char.libspirv-nvptx64-nvidia-cuda."
                    "bc"
                  : "remangled-l64-signed_char.libspirv-nvptx64-nvidia-cuda."
                    "bc";

          for (StringRef LibraryPath : LibraryPaths) {
            SmallString<128> LibSpirvTargetFile(LibraryPath);
            llvm::sys::path::append(LibSpirvTargetFile, LibSpirvTargetName);
            if (llvm::sys::fs::exists(LibSpirvTargetFile) ||
                Args.hasArg(options::OPT__HASH_HASH_HASH)) {
              LibSpirvFile = std::string(LibSpirvTargetFile.str());
              break;
            }
          }
        }

        if (!LibSpirvFile.empty()) {
          Arg *LibClcInputArg = MakeInputArg(Args, C.getDriver().getOpts(),
                                             Args.MakeArgString(LibSpirvFile));
          auto *SYCLLibClcInputAction =
              C.MakeAction<InputAction>(*LibClcInputArg, types::TY_LLVM_BC);
          DeviceLinkObjects.push_back(SYCLLibClcInputAction);
        }

        const toolchains::CudaToolChain *CudaTC =
            static_cast<const toolchains::CudaToolChain *>(TC);
        for (const auto &LinkInputEnum : enumerate(DeviceLinkerInputs)) {
          const char *BoundArch =
              SYCLTargetInfoList[LinkInputEnum.index()].BoundArch;
          std::string LibDeviceFile =
              CudaTC->CudaInstallation.getLibDeviceFile(BoundArch);
          if (!LibDeviceFile.empty()) {
            Arg *CudaDeviceLibInputArg =
                MakeInputArg(Args, C.getDriver().getOpts(),
                             Args.MakeArgString(LibDeviceFile));
            auto *SYCLDeviceLibInputAction = C.MakeAction<InputAction>(
                *CudaDeviceLibInputArg, types::TY_LLVM_BC);
            DeviceLinkObjects.push_back(SYCLDeviceLibInputAction);
          }
        }
      }
      return NumOfDeviceLibLinked != 0;
    }

    void appendLinkDependences(OffloadAction::DeviceDependences &DA) override {
      // DeviceLinkerInputs holds binaries per ToolChain (TC) / bound-arch pair
      // The following will loop link and post process for each TC / bound-arch
      // to produce a final binary.
      // They will be bundled per TC before being sent to the Offload Wrapper.
      for (const auto &LinkInputEnum : enumerate(DeviceLinkerInputs)) {
        auto &LI = LinkInputEnum.value();
        int Index = LinkInputEnum.index();
        const ToolChain *TC = SYCLTargetInfoList[Index].TC;
        const char *BoundArch = SYCLTargetInfoList[Index].BoundArch;

        auto TripleIt = llvm::find_if(SYCLTripleList, [&](auto &SYCLTriple) {
          return SYCLTriple == TC->getTriple();
        });
        if (TripleIt == SYCLTripleList.end()) {
          // If the toolchain's triple is absent in this "main" triple
          // collection, this means it was created specifically for one of
          // the SYCL AOT inputs. Those will be handled separately.
          continue;
        }
        if (LI.empty())
          // Current list is empty, nothing to process.
          continue;
        ActionList AL;
        appendSYCLDeviceLink(LI, TC, DA, AL, BoundArch);
      }
      for (auto &SAI : SYCLAOTInputs) {
        // Extract binary file name
        std::string FN(SAI.second);
        const char *FNStr = Args.MakeArgString(FN);
        Arg *myArg = Args.MakeSeparateArg(
            nullptr, C.getDriver().getOpts().getOption(options::OPT_INPUT),
            FNStr);
        auto *SYCLAdd =
            C.MakeAction<InputAction>(*myArg, types::TY_SYCL_FATBIN);
        auto *DeviceWrappingAction =
            C.MakeAction<OffloadWrapperJobAction>(SYCLAdd, types::TY_Object);

        // Extract the target triple for this binary
        llvm::Triple TT(SAI.first);
        // Extract the toolchain for this target triple
        auto SYCLDeviceTC = llvm::find_if(
            ToolChains, [&](auto &TC) { return TC->getTriple() == TT; });
        assert(SYCLDeviceTC != ToolChains.end() &&
               "No toolchain found for this AOT input");

        DA.add(*DeviceWrappingAction, **SYCLDeviceTC,
               /*BoundArch=*/nullptr, Action::OFK_SYCL);
      }
    }

    void addDeviceLinkDependencies(OffloadDepsJobAction *DA) override {
      unsigned I = 0;
      for (auto &TargetInfo : SYCLTargetInfoList) {
        DA->registerDependentActionInfo(TargetInfo.TC, TargetInfo.BoundArch,
                                        Action::OFK_SYCL);
        DeviceLinkerInputs[I++].push_back(DA);
      }
    }

    /// Initialize the GPU architecture list from arguments - this populates
    /// `GpuArchList` from `--offload-arch` flags. Only relevant if compiling to
    /// CUDA or AMDGCN. Return true if any initialization errors are found.
    /// FIXME: "offload-arch" and the BoundArch mechanism should also be
    // used in the SYCLToolChain for SPIR-V AOT to track the offload
    // architecture instead of the Triple sub-arch it currently uses.
    bool initializeGpuArchMap() {
      const OptTable &Opts = C.getDriver().getOpts();
      for (auto *A : Args) {
        unsigned Index;
        llvm::Triple *TargetBE = nullptr;

        auto GetTripleIt = [&, this](llvm::StringRef Triple) {
          llvm::Triple TargetTriple{Triple};
          auto TripleIt = llvm::find_if(SYCLTripleList, [&](auto &SYCLTriple) {
            return SYCLTriple == TargetTriple;
          });
          return TripleIt != SYCLTripleList.end() ? &*TripleIt : nullptr;
        };

        if (A->getOption().matches(options::OPT_Xsycl_backend_EQ)) {
          TargetBE = GetTripleIt(A->getValue(0));
          // Passing device args: -Xsycl-target-backend=<triple> -opt=val.
          if (TargetBE)
            Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
          else
            continue;
        } else if (A->getOption().matches(options::OPT_Xsycl_backend)) {
          if (SYCLTripleList.size() > 1) {
            C.getDriver().Diag(diag::err_drv_Xsycl_target_missing_triple)
                << A->getSpelling();
            continue;
          }
          // Passing device args: -Xsycl-target-backend -opt=val.
          TargetBE = &SYCLTripleList.front();
          Index = Args.getBaseArgs().MakeIndex(A->getValue(0));
        } else
          continue;

        auto ParsedArg = Opts.ParseOneArg(Args, Index);

        // TODO: Support --no-cuda-gpu-arch, --{,no-}cuda-gpu-arch=all.
        if (ParsedArg &&
            ParsedArg->getOption().matches(options::OPT_offload_arch_EQ)) {
          const char *ArchStr = ParsedArg->getValue(0);
          if (TargetBE->isNVPTX()) {
            // CUDA arch also applies to AMDGCN ...
            CudaArch Arch = StringToCudaArch(ArchStr);
            if (Arch == CudaArch::UNKNOWN || !IsNVIDIAGpuArch(Arch)) {
              C.getDriver().Diag(clang::diag::err_drv_cuda_bad_gpu_arch)
                  << ArchStr;
              continue;
            }
            ArchStr = CudaArchToString(Arch);
          } else if (TargetBE->isAMDGCN()) {
            llvm::StringMap<bool> Features;
            auto Arch = parseTargetID(
                *getHIPOffloadTargetTriple(C.getDriver(), C.getInputArgs()),
                ArchStr, &Features);
            if (!Arch) {
              C.getDriver().Diag(clang::diag::err_drv_bad_target_id) << ArchStr;
              continue;
            }
            auto CanId = getCanonicalTargetID(Arch.value(), Features);
            ArchStr = Args.MakeArgStringRef(CanId);
          }
          ParsedArg->claim();
          GpuArchList.emplace_back(*TargetBE, ArchStr);
          A->claim();
        }
      }

      // Handle defaults architectures
      for (auto &Triple : SYCLTripleList) {
        // For NVIDIA use SM_50 as a default
        if (Triple.isNVPTX() && llvm::none_of(GpuArchList, [&](auto &P) {
              return P.first.isNVPTX();
            })) {
          const char *DefaultArch = CudaArchToString(CudaArch::SM_50);
          GpuArchList.emplace_back(Triple, DefaultArch);
        }

        // For AMD require the architecture to be set by the user
        if (Triple.isAMDGCN() && llvm::none_of(GpuArchList, [&](auto &P) {
              return P.first.isAMDGCN();
            })) {
          C.getDriver().Diag(clang::diag::err_drv_sycl_missing_amdgpu_arch);
          return true;
        }
      }

      return false;
    }

    // Goes through all of the arguments, including inputs expected for the
    // linker directly, to determine if the targets contained in the objects and
    // archives match target expectations being performed.
    void
    checkForOffloadMismatch(Compilation &C, DerivedArgList &Args,
                            SmallVector<DeviceTargetInfo, 4> &Targets) const {
      if (Targets.empty())
        return;

      SmallVector<const char *, 16> OffloadLibArgs(
          getLinkerArgs(C, Args, true));
      // Gather all of the sections seen in the offload objects/archives
      SmallVector<std::string, 4> UniqueSections;
      for (StringRef OLArg : OffloadLibArgs) {
        SmallVector<std::string, 4> Sections(getOffloadSections(C, OLArg));
        for (auto &Section : Sections) {
          // We only care about sections that start with 'sycl-'.  Also remove
          // the prefix before adding it.
          std::string Prefix("sycl-");
          if (Section.compare(0, Prefix.length(), Prefix) != 0)
            continue;

          std::string Arch = Section.substr(Prefix.length());

          // There are a few different variants for FPGA, if we see one, just
          // use the default FPGA triple to reduce possible match confusion.
          if (Arch.compare(0, 4, "fpga") == 0)
            Arch = C.getDriver().MakeSYCLDeviceTriple("spir64_fpga").str();

          if (std::find(UniqueSections.begin(), UniqueSections.end(), Arch) ==
              UniqueSections.end())
            UniqueSections.push_back(Arch);
        }
      }

      if (!UniqueSections.size())
        return;

      for (auto &SyclTarget : Targets) {
        std::string SectionTriple = SyclTarget.TC->getTriple().str();
        if (SyclTarget.BoundArch) {
          SectionTriple += "-";
          SectionTriple += SyclTarget.BoundArch;
        }
        // If any matching section is found, we are good.
        if (std::find(UniqueSections.begin(), UniqueSections.end(),
                      SectionTriple) != UniqueSections.end())
          continue;
        // If any section found is an 'image' based object that was created
        // with the intention of not requiring the matching SYCL target, do
        // not emit the diagnostic.
        if (SyclTarget.TC->getTriple().isSPIR()) {
          bool SectionFound = false;
          for (auto Section : UniqueSections) {
            if (SectionFound)
              break;
            SmallVector<std::string, 3> ArchList = {"spir64_gen", "spir64_fpga",
                                                    "spir64_x86_64"};
            for (auto ArchStr : ArchList) {
              std::string Arch(ArchStr + "_image");
              if (Section.find(Arch) != std::string::npos) {
                SectionFound = true;
                break;
              }
            }
          }
          if (SectionFound)
            continue;
        }
        // Didn't find any matches, return the full list for the diagnostic.
        SmallString<128> ArchListStr;
        int Cnt = 0;
        for (std::string Section : UniqueSections) {
          if (Cnt)
            ArchListStr += ", ";
          ArchListStr += Section;
          Cnt++;
        }
        if (tools::SYCL::shouldDoPerObjectFileLinking(C)) {
          C.getDriver().Diag(diag::err_drv_no_rdc_sycl_target_missing)
              << SectionTriple << ArchListStr;
          C.setContainsError();
        } else {
          C.getDriver().Diag(diag::warn_drv_sycl_target_missing)
              << SectionTriple << ArchListStr;
        }
      }
    }

    // Function checks that user passed -fsycl-add-default-spec-consts-image
    // flag with at least one AOT target. If no AOT target has been passed then
    // a warning is issued.
    void checkForMisusedAddDefaultSpecConstsImageFlag(
        Compilation &C, const DerivedArgList &Args,
        const SmallVector<DeviceTargetInfo, 4> &Targets) const {
      if (!Args.hasFlag(options::OPT_fsycl_add_default_spec_consts_image,
                        options::OPT_fno_sycl_add_default_spec_consts_image,
                        false))
        return;

      bool foundAOT = std::any_of(
          Targets.begin(), Targets.end(), [](const DeviceTargetInfo &DTI) {
            llvm::Triple T = DTI.TC->getTriple();
            bool isSpirvAOT =
                T.getSubArch() == llvm::Triple::SPIRSubArch_fpga ||
                T.getSubArch() == llvm::Triple::SPIRSubArch_gen ||
                T.getSubArch() == llvm::Triple::SPIRSubArch_x86_64;

            return T.isNVPTX() || T.isAMDGCN() || isSpirvAOT;
          });

      if (!foundAOT)
        C.getDriver().Diag(
            diag::warn_drv_fsycl_add_default_spec_consts_image_flag_in_non_AOT);
    }

    // Go through the offload sections of the provided binary.  Gather all
    // all of the sections which match the expected format of the triple
    // generated when creating fat objects that contain full device binaries.
    // Expected format is sycl-<aot_arch>_image-unknown-unknown.
    //   <aot_arch> values:  spir64_gen, spir64_x86_64, spir64_fpga
    SmallVector<std::string, 4> deviceBinarySections(Compilation &C,
                                                     const StringRef &Input) {
      SmallVector<std::string, 4> Sections(getOffloadSections(C, Input));
      SmallVector<std::string, 4> FinalDeviceSections;
      for (auto S : Sections) {
        SmallVector<std::string, 3> ArchList = {"spir64_gen", "spir64_fpga",
                                                "spir64_x86_64"};
        for (auto A : ArchList) {
          std::string Arch("sycl-" + A + "_image");
          if (S.find(Arch) != std::string::npos)
            FinalDeviceSections.push_back(S);
        }
      }
      return FinalDeviceSections;
    }

    bool initialize() override {
      using namespace tools::SYCL;
      // Get the SYCL toolchains. If we don't get any, the action builder will
      // know there is nothing to do related to SYCL offloading.
      auto SYCLTCRange = C.getOffloadToolChains<Action::OFK_SYCL>();
      for (auto TI = SYCLTCRange.first, TE = SYCLTCRange.second; TI != TE;
           ++TI)
        ToolChains.push_back(TI->second);

      // Nothing to offload if no SYCL Toolchain
      if (ToolChains.empty())
        return false;

      Arg *SYCLLinkTargets = Args.getLastArg(
                                  options::OPT_fsycl_link_targets_EQ);
      WrapDeviceOnlyBinary = Args.hasArg(options::OPT_fsycl_link_EQ);
      auto *DeviceCodeSplitArg =
          Args.getLastArg(options::OPT_fsycl_device_code_split_EQ);
      // -fsycl-device-code-split is an alias to
      // -fsycl-device-code-split=auto
      DeviceCodeSplit = DeviceCodeSplitArg &&
                        DeviceCodeSplitArg->getValue() != StringRef("off");
      // Gather information about the SYCL Ahead of Time targets.  The targets
      // are determined on the SubArch values passed along in the triple.
      Arg *SYCLTargets =
              C.getInputArgs().getLastArg(options::OPT_fsycl_targets_EQ);
      Arg *SYCLAddTargets = Args.getLastArg(options::OPT_fsycl_add_targets_EQ);
      Arg *SYCLfpga = C.getInputArgs().getLastArg(options::OPT_fintelfpga);
      bool HasValidSYCLRuntime = C.getInputArgs().hasFlag(
          options::OPT_fsycl, options::OPT_fno_sycl, false);
      bool SYCLfpgaTriple = false;
      bool ShouldAddDefaultTriple = true;
      bool GpuInitHasErrors = false;
      bool HasSYCLTargetsOption =
          SYCLAddTargets || SYCLTargets || SYCLLinkTargets;

      // Make -fintelfpga flag imply -fsycl.
      if (SYCLfpga && !HasValidSYCLRuntime)
        HasValidSYCLRuntime = true;

      if (HasSYCLTargetsOption) {
        if (SYCLTargets || SYCLLinkTargets) {
          Arg *SYCLTargetsValues = SYCLTargets ? SYCLTargets : SYCLLinkTargets;
          // Fill SYCLTripleList
          llvm::StringMap<StringRef> FoundNormalizedTriples;
          for (StringRef Val : SYCLTargetsValues->getValues()) {
            StringRef UserTargetName(Val);
            if (auto ValidDevice = gen::isGPUTarget<gen::IntelGPU>(Val)) {
              if (ValidDevice->empty())
                // Unrecognized, we have already diagnosed this earlier; skip.
                continue;
              // Add the proper -device value to the list.
              GpuArchList.emplace_back(C.getDriver().MakeSYCLDeviceTriple(
                                       "spir64_gen"), ValidDevice->data());
              UserTargetName = "spir64_gen";
            } else if (auto ValidDevice =
                           gen::isGPUTarget<gen::NvidiaGPU>(Val)) {
              if (ValidDevice->empty())
                // Unrecognized, we have already diagnosed this earlier; skip.
                continue;
              // Add the proper -device value to the list.
              GpuArchList.emplace_back(
                  C.getDriver().MakeSYCLDeviceTriple("nvptx64-nvidia-cuda"),
                  ValidDevice->data());
              UserTargetName = "nvptx64-nvidia-cuda";
            } else if (auto ValidDevice = gen::isGPUTarget<gen::AmdGPU>(Val)) {
              if (ValidDevice->empty())
                // Unrecognized, we have already diagnosed this earlier; skip.
                continue;
              // Add the proper -device value to the list.
              GpuArchList.emplace_back(
                  C.getDriver().MakeSYCLDeviceTriple("amdgcn-amd-amdhsa"),
                  ValidDevice->data());
              UserTargetName = "amdgcn-amd-amdhsa";
            } else if (Val == "native_cpu") {
              const ToolChain *HostTC =
                  C.getSingleOffloadToolChain<Action::OFK_Host>();
              llvm::Triple TT = HostTC->getTriple();
              SYCLTripleList.push_back(TT);
              continue;
            }

            llvm::Triple TT(C.getDriver().MakeSYCLDeviceTriple(Val));
            std::string NormalizedName = TT.normalize();

            // Make sure we don't have a duplicate triple.
            auto Duplicate = FoundNormalizedTriples.find(NormalizedName);
            if (Duplicate != FoundNormalizedTriples.end())
              continue;

            // Store the current triple so that we can check for duplicates in
            // the following iterations.
            FoundNormalizedTriples[NormalizedName] = Val;

            SYCLTripleList.push_back(
                C.getDriver().MakeSYCLDeviceTriple(UserTargetName));
            if (TT.getSubArch() == llvm::Triple::SPIRSubArch_fpga)
              SYCLfpgaTriple = true;
            // For user specified spir64_gen, add an empty device value as a
            // placeholder.
            if (TT.getSubArch() == llvm::Triple::SPIRSubArch_gen)
              GpuArchList.emplace_back(TT, nullptr);
          }

          // Fill GpuArchList, end if there are issues in initializingGpuArchMap
          GpuInitHasErrors = initializeGpuArchMap();
          if (GpuInitHasErrors)
            return true;

          int I = 0;
          // Fill SYCLTargetInfoList
          for (auto &TT : SYCLTripleList) {
            auto TCIt = llvm::find_if(
                ToolChains, [&](auto &TC) { return TT == TC->getTriple(); });
            assert(TCIt != ToolChains.end() &&
                   "Toolchain was not created for this platform");
            if (!TT.isNVPTX() && !TT.isAMDGCN()) {
              // When users specify the target as 'intel_gpu_*', the proper
              // triple is 'spir64_gen'.  The given string from intel_gpu_*
              // is the target device.
              if (TT.isSPIR() &&
                  TT.getSubArch() == llvm::Triple::SPIRSubArch_gen) {
                StringRef Device(GpuArchList[I].second);
                SYCLTargetInfoList.emplace_back(
                    *TCIt, Device.empty() ? nullptr : Device.data());
                ++I;
                continue;
              }
              SYCLTargetInfoList.emplace_back(*TCIt, nullptr);
            } else {
              const char *OffloadArch = nullptr;
              for (auto &A : GpuArchList) {
                if (TT == A.first) {
                  OffloadArch = A.second;
                  break;
                }
              }
              assert(OffloadArch && "Failed to find matching arch.");
              SYCLTargetInfoList.emplace_back(*TCIt, OffloadArch);
              ++I;
            }
          }
        }
        if (SYCLAddTargets) {
          for (StringRef Val : SYCLAddTargets->getValues()) {
            // Parse out the Triple and Input (triple:binary). At this point,
            // the format has already been validated at the Driver level.
            // Populate the pairs. Each of these will be wrapped and fed
            // into the final binary.
            std::pair<StringRef, StringRef> I = Val.split(':');
            llvm::Triple TT(I.first);

            auto TCIt = llvm::find_if(
                ToolChains, [&](auto &TC) { return TT == TC->getTriple(); });
            assert(TCIt != ToolChains.end() &&
                   "Toolchain was not created for this platform");

            const char *TF = C.getArgs().MakeArgString(I.second);
            // populate the AOT binary inputs vector.
            SYCLAOTInputs.push_back(std::make_pair(TT, TF));
            ShouldAddDefaultTriple = false;
            // Add an empty entry to the Device list
            if (TT.getSubArch() == llvm::Triple::SPIRSubArch_gen)
              GpuArchList.emplace_back(TT, nullptr);
          }
        }
      } else if (HasValidSYCLRuntime) {
        // -fsycl is provided without -fsycl-*targets.
        bool SYCLfpga = C.getInputArgs().hasArg(options::OPT_fintelfpga);
        // -fsycl -fintelfpga implies spir64_fpga
        const char *SYCLTargetArch =
            SYCLfpga ? "spir64_fpga" : getDefaultSYCLArch(C);
        llvm::Triple TT = C.getDriver().MakeSYCLDeviceTriple(SYCLTargetArch);
        auto TCIt = llvm::find_if(
            ToolChains, [&](auto &TC) { return TT == TC->getTriple(); });
        assert(TCIt != ToolChains.end() &&
               "Toolchain was not created for this platform");
        SYCLTripleList.push_back(TT);
        SYCLTargetInfoList.emplace_back(*TCIt, nullptr);
        if (SYCLfpga)
          SYCLfpgaTriple = true;
      }

      // Device only compilation for -fsycl-link (no FPGA) and
      // -fsycl-link-targets
      CompileDeviceOnly =
          (SYCLLinkTargets || (WrapDeviceOnlyBinary && !SYCLfpgaTriple));

      // Set the FPGA output type based on command line (-fsycl-link).
      if (auto *A = C.getInputArgs().getLastArg(options::OPT_fsycl_link_EQ)) {
        FPGAOutType = (A->getValue() == StringRef("early"))
                          ? types::TY_FPGA_AOCR
                          : types::TY_FPGA_AOCX;
        if (C.getDriver().IsFPGAEmulationMode())
          FPGAOutType = (A->getValue() == StringRef("early"))
                            ? types::TY_FPGA_AOCR_EMU
                            : types::TY_FPGA_AOCX;
      }

      auto makeInputAction = [&](const StringRef Name,
                                 types::ID Type) -> Action * {
        const llvm::opt::OptTable &Opts = C.getDriver().getOpts();
        Arg *InputArg = MakeInputArg(Args, Opts, Args.MakeArgString(Name));
        Action *Current = C.MakeAction<InputAction>(*InputArg, Type);
        return Current;
      };
      // Populate FPGA archives that could contain dep files to be
      // incorporated into the aoc compilation.  Consider AOCR type archives
      // as well for tracking symbols and properties information.
      if (SYCLfpgaTriple && Args.hasArg(options::OPT_fintelfpga)) {
        SmallVector<const char *, 16> LinkArgs(getLinkerArgs(C, Args));
        for (StringRef LA : LinkArgs) {
          if (isStaticArchiveFile(LA) && hasOffloadSections(C, LA, Args)) {
            FPGAArchiveInputs.push_back(makeInputAction(LA, types::TY_Archive));
            for (types::ID Type : {types::TY_FPGA_AOCR, types::TY_FPGA_AOCR_EMU,
                                   types::TY_FPGA_AOCX}) {
              if (hasFPGABinary(C, LA.str(), Type)) {
                FPGAAOCArchives.push_back(makeInputAction(LA, Type));
                break;
              }
            }
          }
        }
      }
      // Discover any objects and archives that contain final device binaries.
      if (HasValidSYCLRuntime) {
        SmallVector<const char *, 16> LinkArgs(getLinkerArgs(C, Args, true));
        for (StringRef LA : LinkArgs) {
          SmallVector<std::string, 4> DeviceTargets(
              deviceBinarySections(C, LA));
          if (!DeviceTargets.empty()) {
            bool IsArchive = isStaticArchiveFile(LA);
            types::ID FileType =
                IsArchive ? types::TY_Archive : types::TY_Object;
            SYCLFinalDeviceList.push_back(
                std::make_pair(makeInputAction(LA, FileType), DeviceTargets));
          }
        }
      }

      if (ShouldAddDefaultTriple && addSYCLDefaultTriple(C, SYCLTripleList)) {
        // If a SYCLDefaultTriple is added to SYCLTripleList,
        // add new target to SYCLTargetInfoList
        llvm::Triple TT = SYCLTripleList.front();
        auto TCIt = llvm::find_if(
            ToolChains, [&](auto &TC) { return TT == TC->getTriple(); });
        SYCLTargetInfoList.emplace_back(*TCIt, nullptr);
      }
      if (SYCLTargetInfoList.empty()) {
        // If there are no SYCL Targets add the front toolchain, this is for
        // `-fsycl-device-only` is provided with no `fsycl` or when all dummy
        // targets are given
        const auto *TC = ToolChains.front();
        SYCLTargetInfoList.emplace_back(TC, nullptr);
      }

      checkForOffloadMismatch(C, Args, SYCLTargetInfoList);
      checkForMisusedAddDefaultSpecConstsImageFlag(C, Args, SYCLTargetInfoList);

      DeviceLinkerInputs.resize(SYCLTargetInfoList.size());
      return false;
    }

    bool canUseBundlerUnbundler() const override {
      // SYCL should use bundled files whenever possible.
      return true;
    }
  };

  ///
  /// TODO: Add the implementation for other specialized builders here.
  ///

  /// Specialized builders being used by this offloading action builder.
  SmallVector<DeviceActionBuilder *, 4> SpecializedBuilders;

  /// Flag set to true if all valid builders allow file bundling/unbundling.
  bool CanUseBundler;

public:
  OffloadingActionBuilder(Compilation &C, DerivedArgList &Args,
                          const Driver::InputList &Inputs)
      : C(C) {
    // Create a specialized builder for each device toolchain.

    IsValid = true;

    // Create a specialized builder for CUDA.
    SpecializedBuilders.push_back(
        new CudaActionBuilder(C, Args, Inputs, *this));

    // Create a specialized builder for HIP.
    SpecializedBuilders.push_back(new HIPActionBuilder(C, Args, Inputs, *this));

    // Create a specialized builder for OpenMP.
    SpecializedBuilders.push_back(
        new OpenMPActionBuilder(C, Args, Inputs, *this));

    // Create a specialized builder for SYCL.
    SpecializedBuilders.push_back(
        new SYCLActionBuilder(C, Args, Inputs, *this));

    //
    // TODO: Build other specialized builders here.
    //

    // Initialize all the builders, keeping track of errors. If all valid
    // builders agree that we can use bundling, set the flag to true.
    unsigned ValidBuilders = 0u;
    unsigned ValidBuildersSupportingBundling = 0u;
    for (auto *SB : SpecializedBuilders) {
      IsValid = IsValid && !SB->initialize();

      // Update the counters if the builder is valid.
      if (SB->isValid()) {
        ++ValidBuilders;
        if (SB->canUseBundlerUnbundler())
          ++ValidBuildersSupportingBundling;
      }
    }
    CanUseBundler =
        ValidBuilders && ValidBuilders == ValidBuildersSupportingBundling;
  }

  ~OffloadingActionBuilder() {
    for (auto *SB : SpecializedBuilders)
      delete SB;
  }

  /// Push an action coming from a specialized DeviceActionBuilder (i.e.,
  /// foreign action) to the other ones
  void pushForeignAction(Action *A) {
    for (auto *SB : SpecializedBuilders) {
      if (SB->isValid())
        SB->pushForeignAction(A);
    }
  }

  /// Record a host action and its originating input argument.
  void recordHostAction(Action *HostAction, const Arg *InputArg) {
    assert(HostAction && "Invalid host action");
    assert(InputArg && "Invalid input argument");
    auto Loc = HostActionToInputArgMap.find(HostAction);
    if (Loc == HostActionToInputArgMap.end())
      HostActionToInputArgMap[HostAction] = InputArg;
    assert(HostActionToInputArgMap[HostAction] == InputArg &&
           "host action mapped to multiple input arguments");
  }

  /// Generate an action that adds device dependences (if any) to a host action.
  /// If no device dependence actions exist, just return the host action \a
  /// HostAction. If an error is found or if no builder requires the host action
  /// to be generated, return nullptr.
  Action *
  addDeviceDependencesToHostAction(Action *HostAction, const Arg *InputArg,
                                   phases::ID CurPhase, phases::ID FinalPhase,
                                   DeviceActionBuilder::PhasesTy &Phases) {
    if (!IsValid)
      return nullptr;

    if (SpecializedBuilders.empty())
      return HostAction;

    assert(HostAction && "Invalid host action!");
    recordHostAction(HostAction, InputArg);

    OffloadAction::DeviceDependences DDeps;
    // Check if all the programming models agree we should not emit the host
    // action. Also, keep track of the offloading kinds employed.
    auto &OffloadKind = InputArgToOffloadKindMap[InputArg];
    unsigned InactiveBuilders = 0u;
    unsigned IgnoringBuilders = 0u;
    for (auto *SB : SpecializedBuilders) {
      if (!SB->isValid()) {
        ++InactiveBuilders;
        continue;
      }
      auto RetCode =
          SB->getDeviceDependences(DDeps, CurPhase, FinalPhase, Phases);

      // If the builder explicitly says the host action should be ignored,
      // we need to increment the variable that tracks the builders that request
      // the host object to be ignored.
      if (RetCode == DeviceActionBuilder::ABRT_Ignore_Host)
        ++IgnoringBuilders;

      // Unless the builder was inactive for this action, we have to record the
      // offload kind because the host will have to use it.
      if (RetCode != DeviceActionBuilder::ABRT_Inactive)
        OffloadKind |= SB->getAssociatedOffloadKind();
    }

    // If all builders agree that the host object should be ignored, just return
    // nullptr.
    if (IgnoringBuilders &&
        SpecializedBuilders.size() == (InactiveBuilders + IgnoringBuilders))
      return nullptr;

    if (DDeps.getActions().empty())
      return HostAction;

    // Add host-cuda-sycl offload kind for the SYCL compilation of .cu files
    if (OffloadKind == (Action::OFK_Cuda | Action::OFK_SYCL)) {
      OffloadAction::HostDependence HDep(
          *HostAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
          /*BoundArch=*/nullptr, Action::OFK_SYCL | Action::OFK_Cuda);
      return C.MakeAction<OffloadAction>(HDep, DDeps);
    }

    // We have dependences we need to bundle together. We use an offload action
    // for that.
    OffloadAction::HostDependence HDep(
        *HostAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
        /*BoundArch=*/nullptr, DDeps);
    return C.MakeAction<OffloadAction>(HDep, DDeps);
  }

  // Update Input action to reflect FPGA device archive specifics based
  // on archive contents.
  bool updateInputForFPGA(Action *&A, const Arg *InputArg,
                          DerivedArgList &Args) {
    std::string InputName = InputArg->getAsString(Args);
    const Driver &D = C.getDriver();
    // Only check for FPGA device information when using fpga SubArch.
    if (A->getType() == types::TY_Object && isObjectFile(InputName))
      return true;

    auto ArchiveTypeMismatch = [&D, &InputName](bool EmitDiag) {
      if (EmitDiag)
        D.Diag(clang::diag::warn_drv_mismatch_fpga_archive) << InputName;
    };
    // Type FPGA aoco is a special case for static archives
    if (A->getType() == types::TY_FPGA_AOCO) {
      if (!hasFPGABinary(C, InputName, types::TY_FPGA_AOCO))
        return false;
      A = C.MakeAction<InputAction>(*InputArg, types::TY_FPGA_AOCO);
      return true;
    }

    // Type FPGA aocx is considered the same way for Hardware and Emulation.
    if (hasFPGABinary(C, InputName, types::TY_FPGA_AOCX)) {
      A = C.MakeAction<InputAction>(*InputArg, types::TY_FPGA_AOCX);
      return true;
    }

    SmallVector<std::pair<types::ID, bool>, 4> FPGAAOCTypes = {
        {types::TY_FPGA_AOCR, false},
        {types::TY_FPGA_AOCR_EMU, true}};
    for (const auto &ArchiveType : FPGAAOCTypes) {
      bool BinaryFound = hasFPGABinary(C, InputName, ArchiveType.first);
      if (BinaryFound && ArchiveType.second == D.IsFPGAEmulationMode()) {
        // Binary matches check and emulation type, we keep this one.
        A = C.MakeAction<InputAction>(*InputArg, ArchiveType.first);
        return true;
      }
      ArchiveTypeMismatch(BinaryFound &&
                          ArchiveType.second == D.IsFPGAHWMode());
    }
    return true;
  }

  /// Generate an action that adds a host dependence to a device action. The
  /// results will be kept in this action builder. Return true if an error was
  /// found.
  bool addHostDependenceToDeviceActions(Action *&HostAction,
                                        const Arg *InputArg,
                                        DerivedArgList &Args) {
    if (!IsValid)
      return true;

    // An FPGA AOCX input does not have a host dependence to the unbundler
    if (HostAction->getType() == types::TY_FPGA_AOCX)
      return false;
    recordHostAction(HostAction, InputArg);

    // If we are supporting bundling/unbundling and the current action is an
    // input action of non-source file, we replace the host action by the
    // unbundling action. The bundler tool has the logic to detect if an input
    // is a bundle or not and if the input is not a bundle it assumes it is a
    // host file. Therefore it is safe to create an unbundling action even if
    // the input is not a bundle.
    bool HasFPGATarget = false;
    if (CanUseBundler && isa<InputAction>(HostAction) &&
        InputArg->getOption().getKind() == llvm::opt::Option::InputClass &&
        !InputArg->getOption().hasFlag(options::LinkerInput) &&
        (!types::isSrcFile(HostAction->getType()) ||
         HostAction->getType() == types::TY_PP_HIP)) {
      ActionList HostActionList;
      Action *A(HostAction);
      bool HasSPIRTarget = false;
      // Only check for FPGA device information when using fpga SubArch.
      auto SYCLTCRange = C.getOffloadToolChains<Action::OFK_SYCL>();
      for (auto TI = SYCLTCRange.first, TE = SYCLTCRange.second; TI != TE;
           ++TI) {
        HasFPGATarget |= TI->second->getTriple().getSubArch() ==
                         llvm::Triple::SPIRSubArch_fpga;
        HasSPIRTarget |= TI->second->getTriple().isSPIR();
      }
      bool isArchive = !(HostAction->getType() == types::TY_Object &&
                         isObjectFile(InputArg->getAsString(Args)));
      if (!HasFPGATarget && isArchive &&
          HostAction->getType() == types::TY_FPGA_AOCO)
        // Archive with Non-FPGA target with AOCO type should not be unbundled.
        return false;
      if (HasFPGATarget && !updateInputForFPGA(A, InputArg, Args))
        return false;
      auto UnbundlingHostAction = C.MakeAction<OffloadUnbundlingJobAction>(
          A, (HasSPIRTarget && HostAction->getType() == types::TY_Archive)
                 ? types::TY_Tempfilelist
                 : A->getType());
      UnbundlingHostAction->registerDependentActionInfo(
          C.getSingleOffloadToolChain<Action::OFK_Host>(),
          /*BoundArch=*/StringRef(), Action::OFK_Host);
      HostAction = UnbundlingHostAction;
      recordHostAction(HostAction, InputArg);
    }

    assert(HostAction && "Invalid host action!");

    // Register the offload kinds that are used.
    auto &OffloadKind = InputArgToOffloadKindMap[InputArg];
    for (auto *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;

      auto RetCode = SB->addDeviceDependences(HostAction);

      // Host dependences for device actions are not compatible with that same
      // action being ignored.
      assert(RetCode != DeviceActionBuilder::ABRT_Ignore_Host &&
             "Host dependence not expected to be ignored.!");

      // Unless the builder was inactive for this action, we have to record the
      // offload kind because the host will have to use it.
      if (RetCode != DeviceActionBuilder::ABRT_Inactive)
        OffloadKind |= SB->getAssociatedOffloadKind();
    }

    // Do not use unbundler if the Host does not depend on device action.
    // Now that we have unbundled the object, when doing -fsycl-link we
    // want to continue the host link with the input object.
    // For unbundling of an FPGA AOCX binary, we want to link with the original
    // FPGA device archive.
    if ((OffloadKind == Action::OFK_None && CanUseBundler) ||
        (HasFPGATarget && ((Args.hasArg(options::OPT_fsycl_link_EQ) &&
                            HostAction->getType() == types::TY_Object) ||
                           HostAction->getType() == types::TY_FPGA_AOCX)))
      if (auto *UA = dyn_cast<OffloadUnbundlingJobAction>(HostAction))
        HostAction = UA->getInputs().back();

    return false;
  }

  /// Add the offloading top level actions that are specific for unique
  /// linking situations where objects are used at only the device link
  /// with no intermedate steps.
  bool appendTopLevelLinkAction(ActionList &AL) {
    // Get the device actions to be appended.
    ActionList OffloadAL;
    for (auto *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;
      SB->appendTopLevelLinkAction(OffloadAL);
    }
    // Append the device actions.
    AL.append(OffloadAL.begin(), OffloadAL.end());
    return false;
  }

  /// Add the offloading top level actions to the provided action list. This
  /// function can replace the host action by a bundling action if the
  /// programming models allow it.
  bool appendTopLevelActions(ActionList &AL, Action *HostAction,
                             const Arg *InputArg) {
    if (HostAction)
      recordHostAction(HostAction, InputArg);

    // Get the device actions to be appended.
    ActionList OffloadAL;
    for (auto *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;
      SB->appendTopLevelActions(OffloadAL);
    }

    // If we can use the bundler, replace the host action by the bundling one in
    // the resulting list. Otherwise, just append the device actions. For
    // device only compilation, HostAction is a null pointer, therefore only do
    // this when HostAction is not a null pointer.
    if (CanUseBundler && HostAction &&
        HostAction->getType() != types::TY_Nothing && !OffloadAL.empty()) {
      // Add the host action to the list in order to create the bundling action.
      OffloadAL.push_back(HostAction);

      // We expect that the host action was just appended to the action list
      // before this method was called.
      assert(HostAction == AL.back() && "Host action not in the list??");
      HostAction = C.MakeAction<OffloadBundlingJobAction>(OffloadAL);
      recordHostAction(HostAction, InputArg);
      AL.back() = HostAction;
    } else
      AL.append(OffloadAL.begin(), OffloadAL.end());

    // Propagate to the current host action (if any) the offload information
    // associated with the current input.
    if (HostAction)
      HostAction->propagateHostOffloadInfo(InputArgToOffloadKindMap[InputArg],
                                           /*BoundArch=*/nullptr);
    return false;
  }

  /// Create link job from the given host inputs and feed the result to offload
  /// deps job which fetches device dependencies from the linked host image.
  /// Offload deps output is then forwarded to active device action builders so
  /// they can add it to the device linker inputs.
  void addDeviceLinkDependenciesFromHost(ActionList &LinkerInputs) {
    // Link image for reading dependencies from it.
    auto *LA = C.MakeAction<LinkJobAction>(LinkerInputs,
                                           types::TY_Host_Dependencies_Image);

    // Calculate all the offload kinds used in the current compilation.
    unsigned ActiveOffloadKinds = 0u;
    for (auto &I : InputArgToOffloadKindMap)
      ActiveOffloadKinds |= I.second;

    OffloadAction::HostDependence HDep(
        *LA, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
        /*BoundArch*/ nullptr, ActiveOffloadKinds);

    auto *DA = C.MakeAction<OffloadDepsJobAction>(HDep, types::TY_LLVM_BC);

    for (auto *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;
      SB->addDeviceLinkDependencies(DA);
    }
  }

  void appendDeviceLinkActions(ActionList &AL) {
    for (DeviceActionBuilder *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;
      SB->appendLinkDeviceActions(AL);
    }
  }

  void makeHostLinkAction(ActionList &LinkerInputs) {

    bool IsCUinSYCL = false;
    for (auto &I : InputArgToOffloadKindMap) {
      if (I.second == (Action::OFK_Cuda | Action::OFK_SYCL)) {
        IsCUinSYCL = true;
      }
    }

    // Add offload action for the SYCL compilation of .cu files
    if (IsCUinSYCL) {
      for (size_t i = 0; i < LinkerInputs.size(); ++i) {
        OffloadAction::HostDependence HDep(
            *LinkerInputs[i], *C.getSingleOffloadToolChain<Action::OFK_Host>(),
            nullptr,
            InputArgToOffloadKindMap[HostActionToInputArgMap[LinkerInputs[i]]]);
        LinkerInputs[i] = C.MakeAction<OffloadAction>(HDep);
      }
    }

    // Build a list of device linking actions.
    ActionList DeviceAL;
    appendDeviceLinkActions(DeviceAL);
    if (DeviceAL.empty())
      return;

    // Let builders add host linking actions.
    Action* HA = nullptr;
    for (DeviceActionBuilder *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;
      HA = SB->appendLinkHostActions(DeviceAL);
      // This created host action has no originating input argument, therefore
      // needs to set its offloading kind directly.
      if (HA) {
        HA->propagateHostOffloadInfo(SB->getAssociatedOffloadKind(),
                                     /*BoundArch=*/nullptr);
        LinkerInputs.push_back(HA);
      }
    }
  }

  /// Processes the host linker action. This currently consists of replacing it
  /// with an offload action if there are device link objects and propagate to
  /// the host action all the offload kinds used in the current compilation. The
  /// resulting action is returned.
  Action *processHostLinkAction(Action *HostAction) {
    // Add all the dependences from the device linking actions.
    OffloadAction::DeviceDependences DDeps;
    for (auto *SB : SpecializedBuilders) {
      if (!SB->isValid())
        continue;

      SB->appendLinkDependences(DDeps);
    }

    // Calculate all the offload kinds used in the current compilation.
    unsigned ActiveOffloadKinds = 0u;
    for (auto &I : InputArgToOffloadKindMap)
      ActiveOffloadKinds |= I.second;

    // If we don't have device dependencies, we don't have to create an offload
    // action.
    if (DDeps.getActions().empty()) {
      // Set all the active offloading kinds to the link action. Given that it
      // is a link action it is assumed to depend on all actions generated so
      // far.
      HostAction->setHostOffloadInfo(ActiveOffloadKinds,
                                     /*BoundArch=*/nullptr);
      // Propagate active offloading kinds for each input to the link action.
      // Each input may have different active offloading kind.
      for (auto *A : HostAction->inputs()) {
        auto ArgLoc = HostActionToInputArgMap.find(A);
        if (ArgLoc == HostActionToInputArgMap.end())
          continue;
        auto OFKLoc = InputArgToOffloadKindMap.find(ArgLoc->second);
        if (OFKLoc == InputArgToOffloadKindMap.end())
          continue;
        A->propagateHostOffloadInfo(OFKLoc->second, /*BoundArch=*/nullptr);
      }
      return HostAction;
    }

    // Create the offload action with all dependences. When an offload action
    // is created the kinds are propagated to the host action, so we don't have
    // to do that explicitly here.
    OffloadAction::HostDependence HDep(
        *HostAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
        /*BoundArch*/ nullptr, ActiveOffloadKinds);
    return C.MakeAction<OffloadAction>(HDep, DDeps);
  }

  void unbundleStaticArchives(Compilation &C, DerivedArgList &Args) {
    if (!Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false))
      return;

    // Go through all of the args, and create a Linker specific argument list.
    // When dealing with fat static archives each archive is individually
    // unbundled.
    SmallVector<const char *, 16> LinkArgs(getLinkerArgs(C, Args));
    const llvm::opt::OptTable &Opts = C.getDriver().getOpts();
    auto unbundleStaticLib = [&](types::ID T, const StringRef &A) {
      Arg *InputArg = MakeInputArg(Args, Opts, Args.MakeArgString(A));
      Action *Current = C.MakeAction<InputAction>(*InputArg, T);
      addHostDependenceToDeviceActions(Current, InputArg, Args);
      auto PL = types::getCompilationPhases(T);
      addDeviceDependencesToHostAction(Current, InputArg, phases::Link,
                                       PL.back(), PL);
    };
    for (StringRef LA : LinkArgs) {
      // At this point, we will process the archives for FPGA AOCO and
      // individual archive unbundling for Windows.
      if (!isStaticArchiveFile(LA))
        continue;
      // FPGA AOCX/AOCR files are archives, but we do not want to unbundle them
      // here as they have already been unbundled and processed for linking.
      // TODO: The multiple binary checks for FPGA types getting a little out
      // of hand. Improve this by doing a single scan of the args and holding
      // that in a data structure for reference.
      if (hasFPGABinary(C, LA.str(), types::TY_FPGA_AOCX) ||
          hasFPGABinary(C, LA.str(), types::TY_FPGA_AOCR) ||
          hasFPGABinary(C, LA.str(), types::TY_FPGA_AOCR_EMU))
        continue;
      // For offload-static-libs we add an unbundling action for each static
      // archive which produces list files with extracted objects. Device lists
      // are then added to the appropriate device link actions and host list is
      // ignored since we are adding offload-static-libs as normal libraries to
      // the host link command.
      if (hasOffloadSections(C, LA, Args)) {
        // Pass along the static libraries to check if we need to add them for
        // unbundling for FPGA AOT static lib usage.  Uses FPGA aoco type to
        // differentiate if aoco unbundling is needed.  Unbundling of aoco is
        // not needed for emulation, as these are treated as regular archives.
        if (C.getDriver().IsFPGAHWMode())
          unbundleStaticLib(types::TY_FPGA_AOCO, LA);
        unbundleStaticLib(types::TY_Archive, LA);
      }
    }
  }
};
} // anonymous namespace.

void Driver::handleArguments(Compilation &C, DerivedArgList &Args,
                             const InputList &Inputs,
                             ActionList &Actions) const {

  // Ignore /Yc/Yu if both /Yc and /Yu passed but with different filenames.
  Arg *YcArg = Args.getLastArg(options::OPT__SLASH_Yc);
  Arg *YuArg = Args.getLastArg(options::OPT__SLASH_Yu);
  if (YcArg && YuArg && strcmp(YcArg->getValue(), YuArg->getValue()) != 0) {
    Diag(clang::diag::warn_drv_ycyu_different_arg_clang_cl);
    Args.eraseArg(options::OPT__SLASH_Yc);
    Args.eraseArg(options::OPT__SLASH_Yu);
    YcArg = YuArg = nullptr;
  }
  if (YcArg && Inputs.size() > 1) {
    Diag(clang::diag::warn_drv_yc_multiple_inputs_clang_cl);
    Args.eraseArg(options::OPT__SLASH_Yc);
    YcArg = nullptr;
  }

  Arg *FinalPhaseArg;
  phases::ID FinalPhase = getFinalPhase(Args, &FinalPhaseArg);

  if (FinalPhase == phases::Link) {
    if (Args.hasArgNoClaim(options::OPT_hipstdpar)) {
      Args.AddFlagArg(nullptr, getOpts().getOption(options::OPT_hip_link));
      Args.AddFlagArg(nullptr,
                      getOpts().getOption(options::OPT_frtlib_add_rpath));
    }
    // Emitting LLVM while linking disabled except in HIPAMD Toolchain
    if (Args.hasArg(options::OPT_emit_llvm) && !Args.hasArg(options::OPT_hip_link))
      Diag(clang::diag::err_drv_emit_llvm_link);
    if (IsCLMode() && LTOMode != LTOK_None &&
        !Args.getLastArgValue(options::OPT_fuse_ld_EQ)
             .equals_insensitive("lld"))
      Diag(clang::diag::err_drv_lto_without_lld);

    // If -dumpdir is not specified, give a default prefix derived from the link
    // output filename. For example, `clang -g -gsplit-dwarf a.c -o x` passes
    // `-dumpdir x-` to cc1. If -o is unspecified, use
    // stem(getDefaultImageName()) (usually stem("a.out") = "a").
    if (!Args.hasArg(options::OPT_dumpdir)) {
      Arg *FinalOutput = Args.getLastArg(options::OPT_o, options::OPT__SLASH_o);
      Arg *Arg = Args.MakeSeparateArg(
          nullptr, getOpts().getOption(options::OPT_dumpdir),
          Args.MakeArgString(
              (FinalOutput ? FinalOutput->getValue()
                           : llvm::sys::path::stem(getDefaultImageName())) +
              "-"));
      Arg->claim();
      Args.append(Arg);
    }
  }

  if (FinalPhase == phases::Preprocess || Args.hasArg(options::OPT__SLASH_Y_)) {
    // If only preprocessing or /Y- is used, all pch handling is disabled.
    // Rather than check for it everywhere, just remove clang-cl pch-related
    // flags here.
    Args.eraseArg(options::OPT__SLASH_Fp);
    Args.eraseArg(options::OPT__SLASH_Yc);
    Args.eraseArg(options::OPT__SLASH_Yu);
    YcArg = YuArg = nullptr;
  }

  unsigned LastPLSize = 0;
  for (auto &I : Inputs) {
    types::ID InputType = I.first;
    const Arg *InputArg = I.second;

    auto PL = types::getCompilationPhases(InputType);
    LastPLSize = PL.size();

    // If the first step comes after the final phase we are doing as part of
    // this compilation, warn the user about it.
    phases::ID InitialPhase = PL[0];
    if (InitialPhase > FinalPhase) {
      if (InputArg->isClaimed())
        continue;

      // Claim here to avoid the more general unused warning.
      InputArg->claim();

      // Suppress all unused style warnings with -Qunused-arguments
      if (Args.hasArg(options::OPT_Qunused_arguments))
        continue;

      // Special case when final phase determined by binary name, rather than
      // by a command-line argument with a corresponding Arg.
      if (CCCIsCPP())
        Diag(clang::diag::warn_drv_input_file_unused_by_cpp)
            << InputArg->getAsString(Args) << getPhaseName(InitialPhase);
      // Special case '-E' warning on a previously preprocessed file to make
      // more sense.
      else if (InitialPhase == phases::Compile &&
               (Args.getLastArg(options::OPT__SLASH_EP,
                                options::OPT__SLASH_P) ||
                Args.getLastArg(options::OPT_E) ||
                Args.getLastArg(options::OPT_M, options::OPT_MM)) &&
               getPreprocessedType(InputType) == types::TY_INVALID)
        Diag(clang::diag::warn_drv_preprocessed_input_file_unused)
            << InputArg->getAsString(Args) << !!FinalPhaseArg
            << (FinalPhaseArg ? FinalPhaseArg->getOption().getName() : "");
      else
        Diag(clang::diag::warn_drv_input_file_unused)
            << InputArg->getAsString(Args) << getPhaseName(InitialPhase)
            << !!FinalPhaseArg
            << (FinalPhaseArg ? FinalPhaseArg->getOption().getName() : "");
      continue;
    }

    if (YcArg) {
      // Add a separate precompile phase for the compile phase.
      if (FinalPhase >= phases::Compile) {
        const types::ID HeaderType = lookupHeaderTypeForSourceType(InputType);
        // Build the pipeline for the pch file.
        Action *ClangClPch = C.MakeAction<InputAction>(*InputArg, HeaderType);
        for (phases::ID Phase : types::getCompilationPhases(HeaderType))
          ClangClPch = ConstructPhaseAction(C, Args, Phase, ClangClPch);
        assert(ClangClPch);
        Actions.push_back(ClangClPch);
        // The driver currently exits after the first failed command.  This
        // relies on that behavior, to make sure if the pch generation fails,
        // the main compilation won't run.
        // FIXME: If the main compilation fails, the PCH generation should
        // probably not be considered successful either.
      }
    }
  }

  // If we are linking, claim any options which are obviously only used for
  // compilation.
  // FIXME: Understand why the last Phase List length is used here.
  if (FinalPhase == phases::Link && LastPLSize == 1) {
    Args.ClaimAllArgs(options::OPT_CompileOnly_Group);
    Args.ClaimAllArgs(options::OPT_cl_compile_Group);
  }
}

void Driver::BuildActions(Compilation &C, DerivedArgList &Args,
                          const InputList &Inputs, ActionList &Actions) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation actions");

  if (!SuppressMissingInputWarning && Inputs.empty()) {
    Diag(clang::diag::err_drv_no_input_files);
    return;
  }

  // Diagnose misuse of /Fo.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_Fo)) {
    StringRef V = A->getValue();
    if (Inputs.size() > 1 && !V.empty() &&
        !llvm::sys::path::is_separator(V.back())) {
      // Check whether /Fo tries to name an output file for multiple inputs.
      Diag(clang::diag::err_drv_out_file_argument_with_multiple_sources)
          << A->getSpelling() << V;
      Args.eraseArg(options::OPT__SLASH_Fo);
    }
  }

  // Diagnose misuse of /Fa.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_Fa)) {
    StringRef V = A->getValue();
    if (Inputs.size() > 1 && !V.empty() &&
        !llvm::sys::path::is_separator(V.back())) {
      // Check whether /Fa tries to name an asm file for multiple inputs.
      Diag(clang::diag::err_drv_out_file_argument_with_multiple_sources)
          << A->getSpelling() << V;
      Args.eraseArg(options::OPT__SLASH_Fa);
    }
  }

  // Diagnose misuse of /o.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_o)) {
    if (A->getValue()[0] == '\0') {
      // It has to have a value.
      Diag(clang::diag::err_drv_missing_argument) << A->getSpelling() << 1;
      Args.eraseArg(options::OPT__SLASH_o);
    }
  }

  handleArguments(C, Args, Inputs, Actions);

  // If '-fintelfpga' is passed, add '-fsycl' to the list of arguments
  const llvm::opt::OptTable &Opts = getOpts();
  Arg *SYCLFpgaArg = C.getInputArgs().getLastArg(options::OPT_fintelfpga);
  if (SYCLFpgaArg &&
      !Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false))
    Args.AddFlagArg(0, Opts.getOption(options::OPT_fsycl));

  // When compiling for -fsycl, generate the integration header files and the
  // Unique ID that will be used during the compilation.
  if (Args.hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false)) {
    const bool IsSaveTemps = isSaveTempsEnabled();
    SmallString<128> OutFileDir;
    if (IsSaveTemps) {
      if (SaveTemps == SaveTempsObj) {
        auto *OptO = C.getArgs().getLastArg(options::OPT_o);
        OutFileDir = (OptO ? OptO->getValues()[0] : "");
        llvm::sys::path::remove_filename(OutFileDir);
        if (!OutFileDir.empty())
          OutFileDir.append(llvm::sys::path::get_separator());
      }
    }
    for (auto &I : Inputs) {
      std::string SrcFileName(I.second->getAsString(Args));
      if ((I.first == types::TY_PP_C || I.first == types::TY_PP_CXX ||
           types::isSrcFile(I.first))) {
        // Unique ID is generated for source files and preprocessed files.
        SmallString<128> ResultID;
        llvm::sys::fs::createUniquePath("uid%%%%%%%%%%%%%%%%", ResultID, false);
        addSYCLUniqueID(Args.MakeArgString(ResultID.str()), SrcFileName);
      }
      if (!types::isSrcFile(I.first))
        continue;

      std::string TmpFileNameHeader;
      std::string TmpFileNameFooter;
      auto StemmedSrcFileName = llvm::sys::path::stem(SrcFileName).str();
      if (IsSaveTemps) {
        TmpFileNameHeader.append(C.getDriver().GetUniquePath(
            OutFileDir.c_str() + StemmedSrcFileName + "-header", "h"));
        TmpFileNameFooter.append(C.getDriver().GetUniquePath(
            OutFileDir.c_str() + StemmedSrcFileName + "-footer", "h"));
      } else {
        TmpFileNameHeader.assign(C.getDriver().GetTemporaryPath(
            StemmedSrcFileName + "-header", "h"));
        TmpFileNameFooter =
            C.getDriver().GetTemporaryPath(StemmedSrcFileName + "-footer", "h");
      }
      StringRef TmpFileHeader =
          C.addTempFile(C.getArgs().MakeArgString(TmpFileNameHeader));
      StringRef TmpFileFooter =
          C.addTempFile(C.getArgs().MakeArgString(TmpFileNameFooter));
      // Use of -fsycl-footer-path puts the integration footer into that
      // specified location.
      if (Arg *A = C.getArgs().getLastArg(options::OPT_fsycl_footer_path_EQ)) {
        SmallString<128> OutName(A->getValue());
        llvm::sys::path::append(OutName,
                                llvm::sys::path::filename(TmpFileNameFooter));
        TmpFileFooter = C.addTempFile(C.getArgs().MakeArgString(OutName));
      }
      addIntegrationFiles(TmpFileHeader, TmpFileFooter, SrcFileName);
    }
  }

  bool UseNewOffloadingDriver =
      C.isOffloadingHostKind(Action::OFK_OpenMP) ||
      Args.hasFlag(options::OPT_offload_new_driver,
                   options::OPT_no_offload_new_driver, false);

  // Builder to be used to build offloading actions.
  std::unique_ptr<OffloadingActionBuilder> OffloadBuilder =
      !UseNewOffloadingDriver
          ? std::make_unique<OffloadingActionBuilder>(C, Args, Inputs)
          : nullptr;

  // Construct the actions to perform.
  ExtractAPIJobAction *ExtractAPIAction = nullptr;
  ActionList LinkerInputs;
  ActionList MergerInputs;
  ActionList DeviceAOTLinkerInputs;
  ActionList HostActions;
  llvm::SmallVector<const Arg *, 6> LinkerInputArgs;
  llvm::SmallVector<phases::ID, phases::MaxNumberOfPhases> PL;

  for (auto &I : Inputs) {
    types::ID InputType = I.first;
    const Arg *InputArg = I.second;

    PL = types::getCompilationPhases(*this, Args, InputType);
    if (PL.empty())
      continue;

    auto FullPL = types::getCompilationPhases(InputType);

    // Build the pipeline for this file.
    Action *Current = C.MakeAction<InputAction>(*InputArg, InputType);

    // Use the current host action in any of the offloading actions, if
    // required.
    if (!UseNewOffloadingDriver)
      if (OffloadBuilder->addHostDependenceToDeviceActions(Current, InputArg, Args))
        break;

    for (phases::ID Phase : PL) {

      // Add any offload action the host action depends on.
      if (!UseNewOffloadingDriver)
        Current = OffloadBuilder->addDeviceDependencesToHostAction(
            Current, InputArg, Phase, PL.back(), FullPL);
      if (!Current)
        break;

      // Queue linker inputs.
      if (Phase == phases::Link) {
        assert(Phase == PL.back() && "linking must be final compilation step.");
        // We don't need to generate additional link commands if emitting AMD
        // bitcode or compiling only for the offload device
        if (!(C.getInputArgs().hasArg(options::OPT_hip_link) &&
              (C.getInputArgs().hasArg(options::OPT_emit_llvm))) &&
            !offloadDeviceOnly())
          LinkerInputs.push_back(Current);
        Current = nullptr;
        break;
      }

      // TODO: Consider removing this because the merged may not end up being
      // the final Phase in the pipeline. Perhaps the merged could just merge
      // and then pass an artifact of some sort to the Link Phase.
      // Queue merger inputs.
      if (Phase == phases::IfsMerge) {
        assert(Phase == PL.back() && "merging must be final compilation step.");
        MergerInputs.push_back(Current);
        Current = nullptr;
        break;
      }

      // When performing -fsycl based compilations and generating dependency
      // information, perform a specific dependency generation compilation which
      // is not based on the source + footer compilation.
      if (Phase == phases::Preprocess && Args.hasArg(options::OPT_fsycl) &&
          Args.hasArg(options::OPT_M_Group) &&
          !Args.hasArg(options::OPT_fno_sycl_use_footer)) {
        Action *PreprocessAction =
            C.MakeAction<PreprocessJobAction>(Current, types::TY_Dependencies);
        PreprocessAction->propagateHostOffloadInfo(Action::OFK_SYCL,
                                                   /*BoundArch=*/nullptr);
        Actions.push_back(PreprocessAction);
      }

      if (Phase == phases::Precompile && ExtractAPIAction) {
        ExtractAPIAction->addHeaderInput(Current);
        Current = nullptr;
        break;
      }

      // FIXME: Should we include any prior module file outputs as inputs of
      // later actions in the same command line?

      // Otherwise construct the appropriate action.
      Action *NewCurrent = ConstructPhaseAction(C, Args, Phase, Current);

      // We didn't create a new action, so we will just move to the next phase.
      if (NewCurrent == Current)
        continue;

      if (auto *EAA = dyn_cast<ExtractAPIJobAction>(NewCurrent))
        ExtractAPIAction = EAA;

      Current = NewCurrent;

      // Try to build the offloading actions and add the result as a dependency
      // to the host.
      if (UseNewOffloadingDriver)
        Current = BuildOffloadingActions(C, Args, I, Current);
      // Use the current host action in any of the offloading actions, if
      // required.
      else if (OffloadBuilder->addHostDependenceToDeviceActions(Current,
                                                                InputArg,
                                                                Args))
        break;

      if (Current->getType() == types::TY_Nothing)
        break;
    }

    // If we ended with something, add to the output list.
    if (Current)
      Actions.push_back(Current);

    // Add any top level actions generated for offloading.
    if (!UseNewOffloadingDriver)
      OffloadBuilder->appendTopLevelActions(Actions, Current, InputArg);
    else if (Current)
      Current->propagateHostOffloadInfo(C.getActiveOffloadKinds(),
                                        /*BoundArch=*/nullptr);
  }

  if (!UseNewOffloadingDriver) {
    OffloadBuilder->appendTopLevelLinkAction(Actions);

    // With static fat archives we need to create additional steps for
    // generating dependence objects for device link actions.
    if (!LinkerInputs.empty() && C.getDriver().getOffloadStaticLibSeen())
      OffloadBuilder->addDeviceLinkDependenciesFromHost(LinkerInputs);

    OffloadBuilder->unbundleStaticArchives(C, Args);
  }

  // For an FPGA archive, we add the unbundling step above to take care of
  // the device side, but also unbundle here to extract the host side
  bool EarlyLink = false;
  if (const Arg *A = Args.getLastArg(options::OPT_fsycl_link_EQ))
    EarlyLink = A->getValue() == StringRef("early");
  for (auto &LI : LinkerInputs) {
    Action *UnbundlerInput = nullptr;
    auto wrapObject = [&] {
      if (EarlyLink && Args.hasArg(options::OPT_fintelfpga)) {
        // Only wrap the object with -fsycl-link=early
        auto *BC = C.MakeAction<OffloadWrapperJobAction>(LI, types::TY_LLVM_BC);
        auto *ASM = C.MakeAction<BackendJobAction>(BC, types::TY_PP_Asm);
        LI = C.MakeAction<AssembleJobAction>(ASM, types::TY_Object);
      }
    };
    if (auto *IA = dyn_cast<InputAction>(LI)) {
      if (IA->getType() == types::TY_FPGA_AOCR ||
          IA->getType() == types::TY_FPGA_AOCX ||
          IA->getType() == types::TY_FPGA_AOCR_EMU) {
        // Add to unbundler.
        UnbundlerInput = LI;
      } else {
        std::string FileName = IA->getInputArg().getAsString(Args);
        if ((IA->getType() == types::TY_Object && !isObjectFile(FileName)) ||
            IA->getInputArg().getOption().hasFlag(options::LinkerInput))
          continue;
        wrapObject();
      }
    } else {
      wrapObject();
    }
    if (UnbundlerInput && !PL.empty()) {
      if (auto *IA = dyn_cast<InputAction>(UnbundlerInput)) {
        std::string FileName = IA->getInputArg().getAsString(Args);
        Arg *InputArg = MakeInputArg(Args, getOpts(), FileName);
        if (!UseNewOffloadingDriver)
          OffloadBuilder->addHostDependenceToDeviceActions(UnbundlerInput,
                                                           InputArg, Args);
      }
    }
  }

  // Add a link action if necessary.

  if (LinkerInputs.empty()) {
    Arg *FinalPhaseArg;
    if (getFinalPhase(Args, &FinalPhaseArg) == phases::Link)
      if (!UseNewOffloadingDriver)
        OffloadBuilder->appendDeviceLinkActions(Actions);
  }

  if (!LinkerInputs.empty()) {
    if (!UseNewOffloadingDriver)
      OffloadBuilder->makeHostLinkAction(LinkerInputs);
    types::ID LinkType(types::TY_Image);
    if (Args.hasArg(options::OPT_fsycl_link_EQ))
      LinkType = types::TY_Archive;
    Action *LA;
    // Check if this Linker Job should emit a static library.
    if (ShouldEmitStaticLibrary(Args)) {
      LA = C.MakeAction<StaticLibJobAction>(LinkerInputs, LinkType);
    } else if (UseNewOffloadingDriver ||
               Args.hasArg(options::OPT_offload_link)) {
      LA = C.MakeAction<LinkerWrapperJobAction>(LinkerInputs, types::TY_Image);
      LA->propagateHostOffloadInfo(C.getActiveOffloadKinds(),
                                   /*BoundArch=*/nullptr);
    } else {
      LA = C.MakeAction<LinkJobAction>(LinkerInputs, LinkType);
    }
    if (!UseNewOffloadingDriver)
      LA = OffloadBuilder->processHostLinkAction(LA);
    Actions.push_back(LA);
  }

  // Add an interface stubs merge action if necessary.
  if (!MergerInputs.empty())
    Actions.push_back(
        C.MakeAction<IfsMergeJobAction>(MergerInputs, types::TY_Image));

  if (Args.hasArg(options::OPT_emit_interface_stubs)) {
    auto PhaseList = types::getCompilationPhases(
        types::TY_IFS_CPP,
        Args.hasArg(options::OPT_c) ? phases::Compile : phases::IfsMerge);

    ActionList MergerInputs;

    for (auto &I : Inputs) {
      types::ID InputType = I.first;
      const Arg *InputArg = I.second;

      // Currently clang and the llvm assembler do not support generating symbol
      // stubs from assembly, so we skip the input on asm files. For ifs files
      // we rely on the normal pipeline setup in the pipeline setup code above.
      if (InputType == types::TY_IFS || InputType == types::TY_PP_Asm ||
          InputType == types::TY_Asm)
        continue;

      Action *Current = C.MakeAction<InputAction>(*InputArg, InputType);

      for (auto Phase : PhaseList) {
        switch (Phase) {
        default:
          llvm_unreachable(
              "IFS Pipeline can only consist of Compile followed by IfsMerge.");
        case phases::Compile: {
          // Only IfsMerge (llvm-ifs) can handle .o files by looking for ifs
          // files where the .o file is located. The compile action can not
          // handle this.
          if (InputType == types::TY_Object)
            break;

          Current = C.MakeAction<CompileJobAction>(Current, types::TY_IFS_CPP);
          break;
        }
        case phases::IfsMerge: {
          assert(Phase == PhaseList.back() &&
                 "merging must be final compilation step.");
          MergerInputs.push_back(Current);
          Current = nullptr;
          break;
        }
        }
      }

      // If we ended with something, add to the output list.
      if (Current)
        Actions.push_back(Current);
    }

    // Add an interface stubs merge action if necessary.
    if (!MergerInputs.empty())
      Actions.push_back(
          C.MakeAction<IfsMergeJobAction>(MergerInputs, types::TY_Image));
  }

  for (auto Opt : {options::OPT_print_supported_cpus,
                   options::OPT_print_supported_extensions}) {
    // If --print-supported-cpus, -mcpu=? or -mtune=? is specified, build a
    // custom Compile phase that prints out supported cpu models and quits.
    //
    // If --print-supported-extensions is specified, call the helper function
    // RISCVMarchHelp in RISCVISAInfo.cpp that prints out supported extensions
    // and quits.
    if (Arg *A = Args.getLastArg(Opt)) {
      if (Opt == options::OPT_print_supported_extensions &&
          !C.getDefaultToolChain().getTriple().isRISCV() &&
          !C.getDefaultToolChain().getTriple().isAArch64() &&
          !C.getDefaultToolChain().getTriple().isARM()) {
        C.getDriver().Diag(diag::err_opt_not_valid_on_target)
            << "--print-supported-extensions";
        return;
      }

      // Use the -mcpu=? flag as the dummy input to cc1.
      Actions.clear();
      Action *InputAc = C.MakeAction<InputAction>(*A, types::TY_C);
      Actions.push_back(
          C.MakeAction<PrecompileJobAction>(InputAc, types::TY_Nothing));
      for (auto &I : Inputs)
        I.second->claim();
    }
  }

  // Call validator for dxil when -Vd not in Args.
  if (C.getDefaultToolChain().getTriple().isDXIL()) {
    // Only add action when needValidation.
    const auto &TC =
        static_cast<const toolchains::HLSLToolChain &>(C.getDefaultToolChain());
    if (TC.requiresValidation(Args)) {
      Action *LastAction = Actions.back();
      Actions.push_back(C.MakeAction<BinaryAnalyzeJobAction>(
          LastAction, types::TY_DX_CONTAINER));
    }
  }

  // Claim ignored clang-cl options.
  Args.ClaimAllArgs(options::OPT_cl_ignored_Group);
}

/// Returns the canonical name for the offloading architecture when using a HIP
/// or CUDA architecture.
static StringRef getCanonicalArchString(Compilation &C,
                                        const llvm::opt::DerivedArgList &Args,
                                        StringRef ArchStr,
                                        const llvm::Triple &Triple,
                                        bool SuppressError = false) {
  // Lookup the CUDA / HIP architecture string. Only report an error if we were
  // expecting the triple to be only NVPTX / AMDGPU.
  CudaArch Arch = StringToCudaArch(getProcessorFromTargetID(Triple, ArchStr));
  if (!SuppressError && Triple.isNVPTX() &&
      (Arch == CudaArch::UNKNOWN || !IsNVIDIAGpuArch(Arch))) {
    C.getDriver().Diag(clang::diag::err_drv_offload_bad_gpu_arch)
        << "CUDA" << ArchStr;
    return StringRef();
  } else if (!SuppressError && Triple.isAMDGPU() &&
             (Arch == CudaArch::UNKNOWN || !IsAMDGpuArch(Arch))) {
    C.getDriver().Diag(clang::diag::err_drv_offload_bad_gpu_arch)
        << "HIP" << ArchStr;
    return StringRef();
  }

  if (IsNVIDIAGpuArch(Arch))
    return Args.MakeArgStringRef(CudaArchToString(Arch));

  if (IsAMDGpuArch(Arch)) {
    llvm::StringMap<bool> Features;
    auto HIPTriple = getHIPOffloadTargetTriple(C.getDriver(), C.getInputArgs());
    if (!HIPTriple)
      return StringRef();
    auto Arch = parseTargetID(*HIPTriple, ArchStr, &Features);
    if (!Arch) {
      C.getDriver().Diag(clang::diag::err_drv_bad_target_id) << ArchStr;
      C.setContainsError();
      return StringRef();
    }
    return Args.MakeArgStringRef(getCanonicalTargetID(*Arch, Features));
  }

  // If the input isn't CUDA or HIP just return the architecture.
  return ArchStr;
}

/// Checks if the set offloading architectures does not conflict. Returns the
/// incompatible pair if a conflict occurs.
static std::optional<std::pair<llvm::StringRef, llvm::StringRef>>
getConflictOffloadArchCombination(const llvm::DenseSet<StringRef> &Archs,
                                  llvm::Triple Triple) {
  if (!Triple.isAMDGPU())
    return std::nullopt;

  std::set<StringRef> ArchSet;
  llvm::copy(Archs, std::inserter(ArchSet, ArchSet.begin()));
  return getConflictTargetIDCombination(ArchSet);
}

llvm::DenseSet<StringRef>
Driver::getOffloadArchs(Compilation &C, const llvm::opt::DerivedArgList &Args,
                        Action::OffloadKind Kind, const ToolChain *TC,
                        bool SuppressError) const {
  if (!TC)
    TC = &C.getDefaultToolChain();

  // --offload and --offload-arch options are mutually exclusive.
  if (Args.hasArgNoClaim(options::OPT_offload_EQ) &&
      Args.hasArgNoClaim(options::OPT_offload_arch_EQ,
                         options::OPT_no_offload_arch_EQ)) {
    C.getDriver().Diag(diag::err_opt_not_valid_with_opt)
        << "--offload"
        << (Args.hasArgNoClaim(options::OPT_offload_arch_EQ)
                ? "--offload-arch"
                : "--no-offload-arch");
  }

  if (KnownArchs.contains(TC))
    return KnownArchs.lookup(TC);

  llvm::DenseSet<StringRef> Archs;
  for (auto *Arg : Args) {
    // Extract any '--[no-]offload-arch' arguments intended for this toolchain.
    std::unique_ptr<llvm::opt::Arg> ExtractedArg = nullptr;
    if (Arg->getOption().matches(options::OPT_Xopenmp_target_EQ) &&
        ToolChain::getOpenMPTriple(Arg->getValue(0)) == TC->getTriple()) {
      Arg->claim();
      unsigned Index = Args.getBaseArgs().MakeIndex(Arg->getValue(1));
      ExtractedArg = getOpts().ParseOneArg(Args, Index);
      Arg = ExtractedArg.get();
    }

    // Add or remove the seen architectures in order of appearance. If an
    // invalid architecture is given we simply exit.
    if (Arg->getOption().matches(options::OPT_offload_arch_EQ)) {
      for (StringRef Arch : llvm::split(Arg->getValue(), ",")) {
        if (Arch == "native" || Arch.empty()) {
          auto GPUsOrErr = TC->getSystemGPUArchs(Args);
          if (!GPUsOrErr) {
            if (SuppressError)
              llvm::consumeError(GPUsOrErr.takeError());
            else
              TC->getDriver().Diag(diag::err_drv_undetermined_gpu_arch)
                  << llvm::Triple::getArchTypeName(TC->getArch())
                  << llvm::toString(GPUsOrErr.takeError()) << "--offload-arch";
            continue;
          }

          for (auto ArchStr : *GPUsOrErr) {
            Archs.insert(
                getCanonicalArchString(C, Args, Args.MakeArgString(ArchStr),
                                       TC->getTriple(), SuppressError));
          }
        } else {
          StringRef ArchStr = getCanonicalArchString(
              C, Args, Arch, TC->getTriple(), SuppressError);
          if (ArchStr.empty())
            return Archs;
          Archs.insert(ArchStr);
        }
      }
    } else if (Arg->getOption().matches(options::OPT_no_offload_arch_EQ)) {
      for (StringRef Arch : llvm::split(Arg->getValue(), ",")) {
        if (Arch == "all") {
          Archs.clear();
        } else {
          StringRef ArchStr = getCanonicalArchString(
              C, Args, Arch, TC->getTriple(), SuppressError);
          if (ArchStr.empty())
            return Archs;
          Archs.erase(ArchStr);
        }
      }
    }
  }

  if (auto ConflictingArchs =
          getConflictOffloadArchCombination(Archs, TC->getTriple())) {
    C.getDriver().Diag(clang::diag::err_drv_bad_offload_arch_combo)
        << ConflictingArchs->first << ConflictingArchs->second;
    C.setContainsError();
  }

  // Skip filling defaults if we're just querying what is availible.
  if (SuppressError)
    return Archs;

  if (Archs.empty()) {
    if (Kind == Action::OFK_Cuda)
      Archs.insert(CudaArchToString(CudaArch::CudaDefault));
    else if (Kind == Action::OFK_HIP)
      Archs.insert(CudaArchToString(CudaArch::HIPDefault));
    else if (Kind == Action::OFK_OpenMP)
      Archs.insert(StringRef());
    else if (Kind == Action::OFK_SYCL)
      Archs.insert(StringRef());
  } else {
    Args.ClaimAllArgs(options::OPT_offload_arch_EQ);
    Args.ClaimAllArgs(options::OPT_no_offload_arch_EQ);
  }

  return Archs;
}

Action *Driver::BuildOffloadingActions(Compilation &C,
                                       llvm::opt::DerivedArgList &Args,
                                       const InputTy &Input,
                                       Action *HostAction) const {
  // Don't build offloading actions if explicitly disabled or we do not have a
  // valid source input and compile action to embed it in. If preprocessing only
  // ignore embedding.
  if (offloadHostOnly() || !types::isSrcFile(Input.first) ||
      !(isa<CompileJobAction>(HostAction) ||
        getFinalPhase(Args) == phases::Preprocess))
    return HostAction;

  ActionList OffloadActions;
  OffloadAction::DeviceDependences DDeps;

  const Action::OffloadKind OffloadKinds[] = {
      Action::OFK_OpenMP, Action::OFK_Cuda, Action::OFK_HIP, Action::OFK_SYCL};

  for (Action::OffloadKind Kind : OffloadKinds) {
    SmallVector<const ToolChain *, 2> ToolChains;
    ActionList DeviceActions;

    auto TCRange = C.getOffloadToolChains(Kind);
    for (auto TI = TCRange.first, TE = TCRange.second; TI != TE; ++TI)
      ToolChains.push_back(TI->second);

    if (ToolChains.empty())
      continue;

    types::ID InputType = Input.first;
    const Arg *InputArg = Input.second;

    // The toolchain can be active for unsupported file types.
    if ((Kind == Action::OFK_Cuda && !types::isCuda(InputType)) ||
        (Kind == Action::OFK_HIP && !types::isHIP(InputType)))
      continue;

    // Get the product of all bound architectures and toolchains.
    SmallVector<std::pair<const ToolChain *, StringRef>> TCAndArchs;
    for (const ToolChain *TC : ToolChains)
      for (StringRef Arch : getOffloadArchs(C, Args, Kind, TC))
        TCAndArchs.push_back(std::make_pair(TC, Arch));

    for (unsigned I = 0, E = TCAndArchs.size(); I != E; ++I)
      DeviceActions.push_back(C.MakeAction<InputAction>(*InputArg, InputType));

    if (DeviceActions.empty())
      return HostAction;

    auto PL = types::getCompilationPhases(*this, Args, InputType);

    for (phases::ID Phase : PL) {
      if (Phase == phases::Link) {
        assert(Phase == PL.back() && "linking must be final compilation step.");
        break;
      }

      auto TCAndArch = TCAndArchs.begin();
      for (Action *&A : DeviceActions) {
        if (A->getType() == types::TY_Nothing)
          continue;

        // Propagate the ToolChain so we can use it in ConstructPhaseAction.
        A->propagateDeviceOffloadInfo(Kind, TCAndArch->second.data(),
                                      TCAndArch->first);
        A = ConstructPhaseAction(C, Args, Phase, A, Kind);

        if (isa<CompileJobAction>(A) && isa<CompileJobAction>(HostAction) &&
            Kind == Action::OFK_OpenMP &&
            HostAction->getType() != types::TY_Nothing) {
          // OpenMP offloading has a dependency on the host compile action to
          // identify which declarations need to be emitted. This shouldn't be
          // collapsed with any other actions so we can use it in the device.
          HostAction->setCannotBeCollapsedWithNextDependentAction();
          OffloadAction::HostDependence HDep(
              *HostAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
              TCAndArch->second.data(), Kind);
          OffloadAction::DeviceDependences DDep;
          DDep.add(*A, *TCAndArch->first, TCAndArch->second.data(), Kind);
          A = C.MakeAction<OffloadAction>(HDep, DDep);
        }

        ++TCAndArch;
      }
    }

    // Compiling HIP in non-RDC mode requires linking each action individually.
    for (Action *&A : DeviceActions) {
      if ((A->getType() != types::TY_Object &&
           A->getType() != types::TY_LTO_BC) ||
          Kind != Action::OFK_HIP ||
          Args.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc, false))
        continue;
      ActionList LinkerInput = {A};
      A = C.MakeAction<LinkJobAction>(LinkerInput, types::TY_Image);
    }

    auto TCAndArch = TCAndArchs.begin();
    for (Action *A : DeviceActions) {
      DDeps.add(*A, *TCAndArch->first, TCAndArch->second.data(), Kind);
      OffloadAction::DeviceDependences DDep;
      DDep.add(*A, *TCAndArch->first, TCAndArch->second.data(), Kind);
      OffloadActions.push_back(C.MakeAction<OffloadAction>(DDep, A->getType()));
      ++TCAndArch;
    }
  }

  if (offloadDeviceOnly())
    return C.MakeAction<OffloadAction>(DDeps, types::TY_Nothing);

  if (OffloadActions.empty())
    return HostAction;

  OffloadAction::DeviceDependences DDep;
  if (C.isOffloadingHostKind(Action::OFK_Cuda) &&
      !Args.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc, false)) {
    // If we are not in RDC-mode we just emit the final CUDA fatbinary for
    // each translation unit without requiring any linking.
    Action *FatbinAction =
        C.MakeAction<LinkJobAction>(OffloadActions, types::TY_CUDA_FATBIN);
    DDep.add(*FatbinAction, *C.getSingleOffloadToolChain<Action::OFK_Cuda>(),
             nullptr, Action::OFK_Cuda);
  } else if (C.isOffloadingHostKind(Action::OFK_HIP) &&
             !Args.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                           false)) {
    // If we are not in RDC-mode we just emit the final HIP fatbinary for each
    // translation unit, linking each input individually.
    Action *FatbinAction =
        C.MakeAction<LinkJobAction>(OffloadActions, types::TY_HIP_FATBIN);
    DDep.add(*FatbinAction, *C.getSingleOffloadToolChain<Action::OFK_HIP>(),
             nullptr, Action::OFK_HIP);
  } else {
    // Package all the offloading actions into a single output that can be
    // embedded in the host and linked.
    Action *PackagerAction =
        C.MakeAction<OffloadPackagerJobAction>(OffloadActions, types::TY_Image);
    DDep.add(*PackagerAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
             nullptr, C.getActiveOffloadKinds());
  }

  // If we are unable to embed a single device output into the host, we need to
  // add each device output as a host dependency to ensure they are still built.
  bool SingleDeviceOutput = !llvm::any_of(OffloadActions, [](Action *A) {
    return A->getType() == types::TY_Nothing;
  }) && isa<CompileJobAction>(HostAction);
  OffloadAction::HostDependence HDep(
      *HostAction, *C.getSingleOffloadToolChain<Action::OFK_Host>(),
      /*BoundArch=*/nullptr, SingleDeviceOutput ? DDep : DDeps);
  return C.MakeAction<OffloadAction>(HDep, SingleDeviceOutput ? DDep : DDeps);
}

Action *Driver::ConstructPhaseAction(
    Compilation &C, const ArgList &Args, phases::ID Phase, Action *Input,
    Action::OffloadKind TargetDeviceOffloadKind) const {
  llvm::PrettyStackTraceString CrashInfo("Constructing phase actions");

  // Some types skip the assembler phase (e.g., llvm-bc), but we can't
  // encode this in the steps because the intermediate type depends on
  // arguments. Just special case here.
  if (Phase == phases::Assemble && Input->getType() != types::TY_PP_Asm)
    return Input;

  // Build the appropriate action.
  switch (Phase) {
  case phases::Link:
    llvm_unreachable("link action invalid here.");
  case phases::IfsMerge:
    llvm_unreachable("ifsmerge action invalid here.");
  case phases::Preprocess: {
    types::ID OutputTy;
    // -M and -MM specify the dependency file name by altering the output type,
    // -if -MD and -MMD are not specified.
    if (Args.hasArg(options::OPT_M, options::OPT_MM) &&
        !Args.hasArg(options::OPT_MD, options::OPT_MMD)) {
      OutputTy = types::TY_Dependencies;
    } else {
      OutputTy = Input->getType();
      // For these cases, the preprocessor is only translating forms, the Output
      // still needs preprocessing.
      if (!Args.hasFlag(options::OPT_frewrite_includes,
                        options::OPT_fno_rewrite_includes, false) &&
          !Args.hasFlag(options::OPT_frewrite_imports,
                        options::OPT_fno_rewrite_imports, false) &&
          !Args.hasFlag(options::OPT_fdirectives_only,
                        options::OPT_fno_directives_only, false) &&
          !CCGenDiagnostics)
        OutputTy = types::getPreprocessedType(OutputTy);
      assert(OutputTy != types::TY_INVALID &&
             "Cannot preprocess this input type!");
    }
    types::ID HostPPType = types::getPreprocessedType(Input->getType());
    if (Args.hasArg(options::OPT_fsycl) && HostPPType != types::TY_INVALID &&
        !Args.hasArg(options::OPT_fno_sycl_use_footer) &&
        TargetDeviceOffloadKind == Action::OFK_None &&
        Input->getType() != types::TY_CUDA_DEVICE) {
      // Performing a host compilation with -fsycl.  Append the integration
      // footer to the source file.
      auto *AppendFooter =
          C.MakeAction<AppendFooterJobAction>(Input, Input->getType());
      // FIXME: There are 2 issues with dependency generation in regards to
      // the integration footer that need to be addressed.
      // 1) Input file referenced on the RHS of a dependency is based on the
      //    input src, which is a temporary.  We want this to be the true
      //    user input src file.
      // 2) When generating dependencies against a preprocessed file, header
      //    file information (using -MD or-MMD) is not provided.
      return C.MakeAction<PreprocessJobAction>(AppendFooter, OutputTy);
    }
    return C.MakeAction<PreprocessJobAction>(Input, OutputTy);
  }
  case phases::Precompile: {
    // API extraction should not generate an actual precompilation action.
    if (Args.hasArg(options::OPT_extract_api))
      return C.MakeAction<ExtractAPIJobAction>(Input, types::TY_API_INFO);

    types::ID OutputTy = getPrecompiledType(Input->getType());
    assert(OutputTy != types::TY_INVALID &&
           "Cannot precompile this input type!");

    // If we're given a module name, precompile header file inputs as a
    // module, not as a precompiled header.
    const char *ModName = nullptr;
    if (OutputTy == types::TY_PCH) {
      if (Arg *A = Args.getLastArg(options::OPT_fmodule_name_EQ))
        ModName = A->getValue();
      if (ModName)
        OutputTy = types::TY_ModuleFile;
    }

    if (Args.hasArg(options::OPT_fsyntax_only)) {
      // Syntax checks should not emit a PCH file
      OutputTy = types::TY_Nothing;
    }

    return C.MakeAction<PrecompileJobAction>(Input, OutputTy);
  }
  case phases::Compile: {
    if (Args.hasArg(options::OPT_fsyntax_only))
      return C.MakeAction<CompileJobAction>(Input, types::TY_Nothing);
    if (Args.hasArg(options::OPT_rewrite_objc))
      return C.MakeAction<CompileJobAction>(Input, types::TY_RewrittenObjC);
    if (Args.hasArg(options::OPT_rewrite_legacy_objc))
      return C.MakeAction<CompileJobAction>(Input,
                                            types::TY_RewrittenLegacyObjC);
    if (Args.hasArg(options::OPT__analyze))
      return C.MakeAction<AnalyzeJobAction>(Input, types::TY_Plist);
    if (Args.hasArg(options::OPT__migrate))
      return C.MakeAction<MigrateJobAction>(Input, types::TY_Remap);
    if (Args.hasArg(options::OPT_emit_ast))
      return C.MakeAction<CompileJobAction>(Input, types::TY_AST);
    if (Args.hasArg(options::OPT_module_file_info))
      return C.MakeAction<CompileJobAction>(Input, types::TY_ModuleFile);
    if (Args.hasArg(options::OPT_verify_pch))
      return C.MakeAction<VerifyPCHJobAction>(Input, types::TY_Nothing);
    if (Args.hasArg(options::OPT_extract_api))
      return C.MakeAction<ExtractAPIJobAction>(Input, types::TY_API_INFO);
    return C.MakeAction<CompileJobAction>(Input, types::TY_LLVM_BC);
  }
  case phases::Backend: {
    if (isUsingLTO() && TargetDeviceOffloadKind == Action::OFK_None) {
      types::ID Output;
      if (Args.hasArg(options::OPT_ffat_lto_objects) &&
          !Args.hasArg(options::OPT_emit_llvm))
        Output = types::TY_PP_Asm;
      else if (Args.hasArg(options::OPT_S))
        Output = types::TY_LTO_IR;
      else
        Output = types::TY_LTO_BC;
      return C.MakeAction<BackendJobAction>(Input, Output);
    }
    if (isUsingLTO(/* IsOffload */ true) &&
        TargetDeviceOffloadKind != Action::OFK_None) {
      types::ID Output =
          Args.hasArg(options::OPT_S) ? types::TY_LTO_IR : types::TY_LTO_BC;
      return C.MakeAction<BackendJobAction>(Input, Output);
    }
    if (Args.hasArg(options::OPT_emit_llvm) ||
        (((Input->getOffloadingToolChain() &&
           Input->getOffloadingToolChain()->getTriple().isAMDGPU()) ||
          TargetDeviceOffloadKind == Action::OFK_HIP) &&
         (Args.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                       false) ||
          TargetDeviceOffloadKind == Action::OFK_OpenMP))) {
      types::ID Output =
          Args.hasArg(options::OPT_S) &&
                  (TargetDeviceOffloadKind == Action::OFK_None ||
                   offloadDeviceOnly() ||
                   (TargetDeviceOffloadKind == Action::OFK_HIP &&
                    !Args.hasFlag(options::OPT_offload_new_driver,
                                  options::OPT_no_offload_new_driver, false)))
              ? types::TY_LLVM_IR
              : types::TY_LLVM_BC;
      return C.MakeAction<BackendJobAction>(Input, Output);
    }
    return C.MakeAction<BackendJobAction>(Input, types::TY_PP_Asm);
  }
  case phases::Assemble:
    return C.MakeAction<AssembleJobAction>(std::move(Input), types::TY_Object);
  }

  llvm_unreachable("invalid phase in ConstructPhaseAction");
}

void Driver::BuildJobs(Compilation &C) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");

  Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);

  // It is an error to provide a -o option if we are making multiple output
  // files. There are exceptions:
  //
  // IfsMergeJob: when generating interface stubs enabled we want to be able to
  // generate the stub file at the same time that we generate the real
  // library/a.out. So when a .o, .so, etc are the output, with clang interface
  // stubs there will also be a .ifs and .ifso at the same location.
  //
  // CompileJob of type TY_IFS_CPP: when generating interface stubs is enabled
  // and -c is passed, we still want to be able to generate a .ifs file while
  // we are also generating .o files. So we allow more than one output file in
  // this case as well.
  //
  // Preprocessing job performed for -fsycl enabled compilation specifically
  // for dependency generation (TY_Dependencies)
  //
  // OffloadClass of type TY_Nothing: device-only output will place many outputs
  // into a single offloading action. We should count all inputs to the action
  // as outputs. Also ignore device-only outputs if we're compiling with
  // -fsyntax-only.
  if (FinalOutput) {
    unsigned NumOutputs = 0;
    unsigned NumIfsOutputs = 0;
    for (const Action *A : C.getActions()) {
      if (A->getType() != types::TY_Nothing &&
          A->getType() != types::TY_DX_CONTAINER &&
          !(A->getKind() == Action::IfsMergeJobClass ||
            (A->getType() == clang::driver::types::TY_IFS_CPP &&
             A->getKind() == clang::driver::Action::CompileJobClass &&
             0 == NumIfsOutputs++) ||
            (A->getKind() == Action::BindArchClass && A->getInputs().size() &&
             A->getInputs().front()->getKind() == Action::IfsMergeJobClass) ||
            (A->getKind() == Action::PreprocessJobClass &&
             A->getType() == types::TY_Dependencies &&
             C.getArgs().hasArg(options::OPT_fsycl))))
        ++NumOutputs;
      else if (A->getKind() == Action::OffloadClass &&
               A->getType() == types::TY_Nothing &&
               !C.getArgs().hasArg(options::OPT_fsyntax_only))
        NumOutputs += A->size();
    }

    if (NumOutputs > 1) {
      Diag(clang::diag::err_drv_output_argument_with_multiple_files);
      FinalOutput = nullptr;
    }
  }

  const llvm::Triple &RawTriple = C.getDefaultToolChain().getTriple();

  // Collect the list of architectures.
  llvm::StringSet<> ArchNames;
  if (RawTriple.isOSBinFormatMachO())
    for (const Arg *A : C.getArgs())
      if (A->getOption().matches(options::OPT_arch))
        ArchNames.insert(A->getValue());

  // Set of (Action, canonical ToolChain triple) pairs we've built jobs for.
  std::map<std::pair<const Action *, std::string>, InputInfoList> CachedResults;
  for (Action *A : C.getActions()) {
    // If we are linking an image for multiple archs then the linker wants
    // -arch_multiple and -final_output <final image name>. Unfortunately, this
    // doesn't fit in cleanly because we have to pass this information down.
    //
    // FIXME: This is a hack; find a cleaner way to integrate this into the
    // process.
    const char *LinkingOutput = nullptr;
    if (isa<LipoJobAction>(A)) {
      if (FinalOutput)
        LinkingOutput = FinalOutput->getValue();
      else
        LinkingOutput = getDefaultImageName();
    }

    BuildJobsForAction(C, A, &C.getDefaultToolChain(),
                       /*BoundArch*/ StringRef(),
                       /*AtTopLevel*/ true,
                       /*MultipleArchs*/ ArchNames.size() > 1,
                       /*LinkingOutput*/ LinkingOutput, CachedResults,
                       /*TargetDeviceOffloadKind*/ Action::OFK_None);
  }

  // If we have more than one job, then disable integrated-cc1 for now. Do this
  // also when we need to report process execution statistics.
  if (C.getJobs().size() > 1 || CCPrintProcessStats)
    for (auto &J : C.getJobs())
      J.InProcess = false;

  if (CCPrintProcessStats) {
    C.setPostCallback([=](const Command &Cmd, int Res) {
      std::optional<llvm::sys::ProcessStatistics> ProcStat =
          Cmd.getProcessStatistics();
      if (!ProcStat)
        return;

      const char *LinkingOutput = nullptr;
      if (FinalOutput)
        LinkingOutput = FinalOutput->getValue();
      else if (!Cmd.getOutputFilenames().empty())
        LinkingOutput = Cmd.getOutputFilenames().front().c_str();
      else
        LinkingOutput = getDefaultImageName();

      if (CCPrintStatReportFilename.empty()) {
        using namespace llvm;
        // Human readable output.
        outs() << sys::path::filename(Cmd.getExecutable()) << ": "
               << "output=" << LinkingOutput;
        outs() << ", total="
               << format("%.3f", ProcStat->TotalTime.count() / 1000.) << " ms"
               << ", user="
               << format("%.3f", ProcStat->UserTime.count() / 1000.) << " ms"
               << ", mem=" << ProcStat->PeakMemory << " Kb\n";
      } else {
        // CSV format.
        std::string Buffer;
        llvm::raw_string_ostream Out(Buffer);
        llvm::sys::printArg(Out, llvm::sys::path::filename(Cmd.getExecutable()),
                            /*Quote*/ true);
        Out << ',';
        llvm::sys::printArg(Out, LinkingOutput, true);
        Out << ',' << ProcStat->TotalTime.count() << ','
            << ProcStat->UserTime.count() << ',' << ProcStat->PeakMemory
            << '\n';
        Out.flush();
        std::error_code EC;
        llvm::raw_fd_ostream OS(CCPrintStatReportFilename, EC,
                                llvm::sys::fs::OF_Append |
                                    llvm::sys::fs::OF_Text);
        if (EC)
          return;
        auto L = OS.lock();
        if (!L) {
          llvm::errs() << "ERROR: Cannot lock file "
                       << CCPrintStatReportFilename << ": "
                       << toString(L.takeError()) << "\n";
          return;
        }
        OS << Buffer;
        OS.flush();
      }
    });
  }

  // If the user passed -Qunused-arguments or there were errors, don't warn
  // about any unused arguments.
  if (Diags.hasErrorOccurred() ||
      C.getArgs().hasArg(options::OPT_Qunused_arguments))
    return;

  // Claim -fdriver-only here.
  (void)C.getArgs().hasArg(options::OPT_fdriver_only);
  // Claim -### here.
  (void)C.getArgs().hasArg(options::OPT__HASH_HASH_HASH);

  // Claim --driver-mode, --rsp-quoting, it was handled earlier.
  (void)C.getArgs().hasArg(options::OPT_driver_mode);
  (void)C.getArgs().hasArg(options::OPT_rsp_quoting);

  bool HasAssembleJob = llvm::any_of(C.getJobs(), [](auto &J) {
    // Match ClangAs and other derived assemblers of Tool. ClangAs uses a
    // longer ShortName "clang integrated assembler" while other assemblers just
    // use "assembler".
    return strstr(J.getCreator().getShortName(), "assembler");
  });
  for (Arg *A : C.getArgs()) {
    // FIXME: It would be nice to be able to send the argument to the
    // DiagnosticsEngine, so that extra values, position, and so on could be
    // printed.
    if (!A->isClaimed()) {
      if (A->getOption().hasFlag(options::NoArgumentUnused))
        continue;

      // Suppress the warning automatically if this is just a flag, and it is an
      // instance of an argument we already claimed.
      const Option &Opt = A->getOption();
      if (Opt.getKind() == Option::FlagClass) {
        bool DuplicateClaimed = false;

        for (const Arg *AA : C.getArgs().filtered(&Opt)) {
          if (AA->isClaimed()) {
            DuplicateClaimed = true;
            break;
          }
        }

        if (DuplicateClaimed)
          continue;
      }

      // In clang-cl, don't mention unknown arguments here since they have
      // already been warned about.
      if (!IsCLMode() || !A->getOption().matches(options::OPT_UNKNOWN)) {
        if (A->getOption().hasFlag(options::TargetSpecific) &&
            !A->isIgnoredTargetSpecific() && !HasAssembleJob &&
            // When for example -### or -v is used
            // without a file, target specific options are not
            // consumed/validated.
            // Instead emitting an error emit a warning instead.
            !C.getActions().empty()) {
          Diag(diag::err_drv_unsupported_opt_for_target)
              << A->getSpelling() << getTargetTriple();
        } else {
          Diag(clang::diag::warn_drv_unused_argument)
              << A->getAsString(C.getArgs());
        }
      }
    }
  }
}

namespace {
/// Utility class to control the collapse of dependent actions and select the
/// tools accordingly.
class ToolSelector final {
  /// The tool chain this selector refers to.
  const ToolChain &TC;

  /// The compilation this selector refers to.
  const Compilation &C;

  /// The base action this selector refers to.
  const JobAction *BaseAction;

  /// Set to true if the current toolchain refers to host actions.
  bool IsHostSelector;

  /// Set to true if save-temps and embed-bitcode functionalities are active.
  bool SaveTemps;
  bool EmbedBitcode;

  /// Get previous dependent action or null if that does not exist. If
  /// \a CanBeCollapsed is false, that action must be legal to collapse or
  /// null will be returned.
  const JobAction *getPrevDependentAction(const ActionList &Inputs,
                                          ActionList &SavedOffloadAction,
                                          bool CanBeCollapsed = true) {
    // An option can be collapsed only if it has a single input.
    if (Inputs.size() != 1)
      return nullptr;

    Action *CurAction = *Inputs.begin();
    if (CanBeCollapsed &&
        !CurAction->isCollapsingWithNextDependentActionLegal())
      return nullptr;

    // If the input action is an offload action. Look through it and save any
    // offload action that can be dropped in the event of a collapse.
    if (auto *OA = dyn_cast<OffloadAction>(CurAction)) {
      // If the dependent action is a device action, we will attempt to collapse
      // only with other device actions. Otherwise, we would do the same but
      // with host actions only.
      if (!IsHostSelector) {
        if (OA->hasSingleDeviceDependence(/*DoNotConsiderHostActions=*/true)) {
          CurAction =
              OA->getSingleDeviceDependence(/*DoNotConsiderHostActions=*/true);
          if (CanBeCollapsed &&
              !CurAction->isCollapsingWithNextDependentActionLegal())
            return nullptr;
          SavedOffloadAction.push_back(OA);
          return dyn_cast<JobAction>(CurAction);
        }
      } else if (OA->hasHostDependence()) {
        CurAction = OA->getHostDependence();
        if (CanBeCollapsed &&
            !CurAction->isCollapsingWithNextDependentActionLegal())
          return nullptr;
        SavedOffloadAction.push_back(OA);
        return dyn_cast<JobAction>(CurAction);
      }
      return nullptr;
    }

    return dyn_cast<JobAction>(CurAction);
  }

  /// Return true if an assemble action can be collapsed.
  bool canCollapseAssembleAction() const {
    return TC.useIntegratedAs() && !SaveTemps &&
           !C.getArgs().hasArg(options::OPT_via_file_asm) &&
           !C.getArgs().hasArg(options::OPT__SLASH_FA) &&
           !C.getArgs().hasArg(options::OPT__SLASH_Fa) &&
           !C.getArgs().hasArg(options::OPT_dxc_Fc);
  }

  /// Return true if a preprocessor action can be collapsed.
  bool canCollapsePreprocessorAction() const {
    return !C.getArgs().hasArg(options::OPT_no_integrated_cpp) &&
           !C.getArgs().hasArg(options::OPT_traditional_cpp) && !SaveTemps &&
           !C.getArgs().hasArg(options::OPT_rewrite_objc);
  }

  /// Struct that relates an action with the offload actions that would be
  /// collapsed with it.
  struct JobActionInfo final {
    /// The action this info refers to.
    const JobAction *JA = nullptr;
    /// The offload actions we need to take care off if this action is
    /// collapsed.
    ActionList SavedOffloadAction;
  };

  /// Append collapsed offload actions from the give nnumber of elements in the
  /// action info array.
  static void AppendCollapsedOffloadAction(ActionList &CollapsedOffloadAction,
                                           ArrayRef<JobActionInfo> &ActionInfo,
                                           unsigned ElementNum) {
    assert(ElementNum <= ActionInfo.size() && "Invalid number of elements.");
    for (unsigned I = 0; I < ElementNum; ++I)
      CollapsedOffloadAction.append(ActionInfo[I].SavedOffloadAction.begin(),
                                    ActionInfo[I].SavedOffloadAction.end());
  }

  /// Functions that attempt to perform the combining. They detect if that is
  /// legal, and if so they update the inputs \a Inputs and the offload action
  /// that were collapsed in \a CollapsedOffloadAction. A tool that deals with
  /// the combined action is returned. If the combining is not legal or if the
  /// tool does not exist, null is returned.
  /// Currently three kinds of collapsing are supported:
  ///  - Assemble + Backend + Compile;
  ///  - Assemble + Backend ;
  ///  - Backend + Compile.
  const Tool *
  combineAssembleBackendCompile(ArrayRef<JobActionInfo> ActionInfo,
                                ActionList &Inputs,
                                ActionList &CollapsedOffloadAction) {
    if (ActionInfo.size() < 3 || !canCollapseAssembleAction())
      return nullptr;
    auto *AJ = dyn_cast<AssembleJobAction>(ActionInfo[0].JA);
    auto *BJ = dyn_cast<BackendJobAction>(ActionInfo[1].JA);
    auto *CJ = dyn_cast<CompileJobAction>(ActionInfo[2].JA);
    if (!AJ || !BJ || !CJ)
      return nullptr;

    // Get compiler tool.
    const Tool *T = TC.SelectTool(*CJ);
    if (!T)
      return nullptr;

    // Can't collapse if we don't have codegen support unless we are
    // emitting LLVM IR.
    bool OutputIsLLVM = types::isLLVMIR(ActionInfo[0].JA->getType());
    if (!T->hasIntegratedBackend() && !(OutputIsLLVM && T->canEmitIR()))
      return nullptr;

    // When using -fembed-bitcode, it is required to have the same tool (clang)
    // for both CompilerJA and BackendJA. Otherwise, combine two stages.
    if (EmbedBitcode) {
      const Tool *BT = TC.SelectTool(*BJ);
      if (BT == T)
        return nullptr;
    }

    if (!T->hasIntegratedAssembler())
      return nullptr;

    Inputs = CJ->getInputs();
    AppendCollapsedOffloadAction(CollapsedOffloadAction, ActionInfo,
                                 /*NumElements=*/3);
    return T;
  }
  const Tool *combineAssembleBackend(ArrayRef<JobActionInfo> ActionInfo,
                                     ActionList &Inputs,
                                     ActionList &CollapsedOffloadAction) {
    if (ActionInfo.size() < 2 || !canCollapseAssembleAction())
      return nullptr;
    auto *AJ = dyn_cast<AssembleJobAction>(ActionInfo[0].JA);
    auto *BJ = dyn_cast<BackendJobAction>(ActionInfo[1].JA);
    if (!AJ || !BJ)
      return nullptr;

    // Get backend tool.
    const Tool *T = TC.SelectTool(*BJ);
    if (!T)
      return nullptr;

    if (!T->hasIntegratedAssembler())
      return nullptr;

    Inputs = BJ->getInputs();
    AppendCollapsedOffloadAction(CollapsedOffloadAction, ActionInfo,
                                 /*NumElements=*/2);
    return T;
  }
  const Tool *combineBackendCompile(ArrayRef<JobActionInfo> ActionInfo,
                                    ActionList &Inputs,
                                    ActionList &CollapsedOffloadAction) {
    if (ActionInfo.size() < 2)
      return nullptr;
    auto *BJ = dyn_cast<BackendJobAction>(ActionInfo[0].JA);
    auto *CJ = dyn_cast<CompileJobAction>(ActionInfo[1].JA);
    if (!BJ || !CJ)
      return nullptr;

    // Check if the initial input (to the compile job or its predessor if one
    // exists) is LLVM bitcode. In that case, no preprocessor step is required
    // and we can still collapse the compile and backend jobs when we have
    // -save-temps. I.e. there is no need for a separate compile job just to
    // emit unoptimized bitcode.
    bool InputIsBitcode = true;
    for (size_t i = 1; i < ActionInfo.size(); i++)
      if (ActionInfo[i].JA->getType() != types::TY_LLVM_BC &&
          ActionInfo[i].JA->getType() != types::TY_LTO_BC) {
        InputIsBitcode = false;
        break;
      }
    if (!InputIsBitcode && !canCollapsePreprocessorAction())
      return nullptr;

    // Get compiler tool.
    const Tool *T = TC.SelectTool(*CJ);
    if (!T)
      return nullptr;

    // Can't collapse if we don't have codegen support unless we are
    // emitting LLVM IR.
    bool OutputIsLLVM = types::isLLVMIR(ActionInfo[0].JA->getType());
    if (!T->hasIntegratedBackend() && !(OutputIsLLVM && T->canEmitIR()))
      return nullptr;

    if (T->canEmitIR() && ((SaveTemps && !InputIsBitcode) || EmbedBitcode))
      return nullptr;

    Inputs = CJ->getInputs();
    AppendCollapsedOffloadAction(CollapsedOffloadAction, ActionInfo,
                                 /*NumElements=*/2);
    return T;
  }

  /// Updates the inputs if the obtained tool supports combining with
  /// preprocessor action, and the current input is indeed a preprocessor
  /// action. If combining results in the collapse of offloading actions, those
  /// are appended to \a CollapsedOffloadAction.
  void combineWithPreprocessor(const Tool *T, ActionList &Inputs,
                               ActionList &CollapsedOffloadAction) {
    if (!T || !canCollapsePreprocessorAction() || !T->hasIntegratedCPP())
      return;

    // Attempt to get a preprocessor action dependence.
    ActionList PreprocessJobOffloadActions;
    ActionList NewInputs;
    for (Action *A : Inputs) {
      auto *PJ = getPrevDependentAction({A}, PreprocessJobOffloadActions);
      if (!PJ || !isa<PreprocessJobAction>(PJ)) {
        NewInputs.push_back(A);
        continue;
      }

      // This is legal to combine. Append any offload action we found and add the
      // current input to preprocessor inputs.
      CollapsedOffloadAction.append(PreprocessJobOffloadActions.begin(),
                                    PreprocessJobOffloadActions.end());
      NewInputs.append(PJ->input_begin(), PJ->input_end());
    }
    Inputs = NewInputs;
  }

public:
  ToolSelector(const JobAction *BaseAction, const ToolChain &TC,
               const Compilation &C, bool SaveTemps, bool EmbedBitcode)
      : TC(TC), C(C), BaseAction(BaseAction), SaveTemps(SaveTemps),
        EmbedBitcode(EmbedBitcode) {
    assert(BaseAction && "Invalid base action.");
    IsHostSelector = BaseAction->getOffloadingDeviceKind() == Action::OFK_None;
  }

  /// Check if a chain of actions can be combined and return the tool that can
  /// handle the combination of actions. The pointer to the current inputs \a
  /// Inputs and the list of offload actions \a CollapsedOffloadActions
  /// connected to collapsed actions are updated accordingly. The latter enables
  /// the caller of the selector to process them afterwards instead of just
  /// dropping them. If no suitable tool is found, null will be returned.
  const Tool *getTool(ActionList &Inputs,
                      ActionList &CollapsedOffloadAction) {
    //
    // Get the largest chain of actions that we could combine.
    //

    SmallVector<JobActionInfo, 5> ActionChain(1);
    ActionChain.back().JA = BaseAction;
    while (ActionChain.back().JA) {
      const Action *CurAction = ActionChain.back().JA;

      // Grow the chain by one element.
      ActionChain.resize(ActionChain.size() + 1);
      JobActionInfo &AI = ActionChain.back();

      // Attempt to fill it with the
      AI.JA =
          getPrevDependentAction(CurAction->getInputs(), AI.SavedOffloadAction);
    }

    // Pop the last action info as it could not be filled.
    ActionChain.pop_back();

    //
    // Attempt to combine actions. If all combining attempts failed, just return
    // the tool of the provided action. At the end we attempt to combine the
    // action with any preprocessor action it may depend on.
    //

    const Tool *T = combineAssembleBackendCompile(ActionChain, Inputs,
                                                  CollapsedOffloadAction);
    if (!T)
      T = combineAssembleBackend(ActionChain, Inputs, CollapsedOffloadAction);
    if (!T)
      T = combineBackendCompile(ActionChain, Inputs, CollapsedOffloadAction);
    if (!T) {
      Inputs = BaseAction->getInputs();
      T = TC.SelectTool(*BaseAction);
    }

    combineWithPreprocessor(T, Inputs, CollapsedOffloadAction);
    return T;
  }
};
}

/// Return a string that uniquely identifies the result of a job. The bound arch
/// is not necessarily represented in the toolchain's triple -- for example,
/// armv7 and armv7s both map to the same triple -- so we need both in our map.
/// Also, we need to add the offloading device kind, as the same tool chain can
/// be used for host and device for some programming models, e.g. OpenMP.
static std::string GetTriplePlusArchString(const ToolChain *TC,
                                           StringRef BoundArch,
                                           Action::OffloadKind OffloadKind) {
  std::string TriplePlusArch = TC->getTriple().normalize();
  if (!BoundArch.empty()) {
    TriplePlusArch += "-";
    TriplePlusArch += BoundArch;
  }
  TriplePlusArch += "-";
  TriplePlusArch += Action::GetOffloadKindName(OffloadKind);
  return TriplePlusArch;
}

static void CollectForEachInputs(
    InputInfoList &InputInfos, const Action *SourceAction, const ToolChain *TC,
    StringRef BoundArch, Action::OffloadKind TargetDeviceOffloadKind,
    const std::map<std::pair<const Action *, std::string>, InputInfoList>
        &CachedResults,
    const ForEachWrappingAction *FEA) {
  for (const Action *Input : SourceAction->getInputs()) {
    // Search for the Input, if not in the cache assume actions were collapsed
    // so recurse.
    auto Lookup = CachedResults.find(
        {Input,
         GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)});
    if (Lookup != CachedResults.end()) {
      if (!FEA->getSerialActions().count(Input)) {
        InputInfos.append(Lookup->second);
      }
    } else {
      CollectForEachInputs(InputInfos, Input, TC, BoundArch,
                           TargetDeviceOffloadKind, CachedResults, FEA);
    }
  }
}

InputInfoList Driver::BuildJobsForAction(
    Compilation &C, const Action *A, const ToolChain *TC, StringRef BoundArch,
    bool AtTopLevel, bool MultipleArchs, const char *LinkingOutput,
    std::map<std::pair<const Action *, std::string>, InputInfoList>
        &CachedResults,
    Action::OffloadKind TargetDeviceOffloadKind) const {
  std::pair<const Action *, std::string> ActionTC = {
      A, GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)};
  auto CachedResult = CachedResults.find(ActionTC);
  if (CachedResult != CachedResults.end()) {
    return CachedResult->second;
  }
  InputInfoList Result = BuildJobsForActionNoCache(
      C, A, TC, BoundArch, AtTopLevel, MultipleArchs, LinkingOutput,
      CachedResults, TargetDeviceOffloadKind);
  CachedResults[ActionTC] = Result;
  return Result;
}

static void handleTimeTrace(Compilation &C, const ArgList &Args,
                            const JobAction *JA, const char *BaseInput,
                            const InputInfo &Result) {
  Arg *A =
      Args.getLastArg(options::OPT_ftime_trace, options::OPT_ftime_trace_EQ);
  if (!A)
    return;
  SmallString<128> Path;
  if (A->getOption().matches(options::OPT_ftime_trace_EQ)) {
    Path = A->getValue();
    if (llvm::sys::fs::is_directory(Path)) {
      SmallString<128> Tmp(Result.getFilename());
      llvm::sys::path::replace_extension(Tmp, "json");
      llvm::sys::path::append(Path, llvm::sys::path::filename(Tmp));
    }
  } else {
    if (Arg *DumpDir = Args.getLastArgNoClaim(options::OPT_dumpdir)) {
      // The trace file is ${dumpdir}${basename}.json. Note that dumpdir may not
      // end with a path separator.
      Path = DumpDir->getValue();
      Path += llvm::sys::path::filename(BaseInput);
    } else {
      Path = Result.getFilename();
    }
    llvm::sys::path::replace_extension(Path, "json");
  }
  const char *ResultFile = C.getArgs().MakeArgString(Path);
  C.addTimeTraceFile(ResultFile, JA);
  C.addResultFile(ResultFile, JA);
}

InputInfoList Driver::BuildJobsForActionNoCache(
    Compilation &C, const Action *A, const ToolChain *TC, StringRef BoundArch,
    bool AtTopLevel, bool MultipleArchs, const char *LinkingOutput,
    std::map<std::pair<const Action *, std::string>, InputInfoList>
        &CachedResults,
    Action::OffloadKind TargetDeviceOffloadKind) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");

  InputInfoList OffloadDependencesInputInfo;
  bool BuildingForOffloadDevice = TargetDeviceOffloadKind != Action::OFK_None;
  if (const OffloadAction *OA = dyn_cast<OffloadAction>(A)) {
    // The 'Darwin' toolchain is initialized only when its arguments are
    // computed. Get the default arguments for OFK_None to ensure that
    // initialization is performed before processing the offload action.
    // FIXME: Remove when darwin's toolchain is initialized during construction.
    C.getArgsForToolChain(TC, BoundArch, Action::OFK_None);

    // The offload action is expected to be used in four different situations.
    //
    // a) Set a toolchain/architecture/kind for a host action:
    //    Host Action 1 -> OffloadAction -> Host Action 2
    //
    // b) Set a toolchain/architecture/kind for a device action;
    //    Device Action 1 -> OffloadAction -> Device Action 2
    //
    // c) Specify a device dependence to a host action;
    //    Device Action 1  _
    //                      \
    //      Host Action 1  ---> OffloadAction -> Host Action 2
    //
    // d) Specify a host dependence to a device action.
    //      Host Action 1  _
    //                      \
    //    Device Action 1  ---> OffloadAction -> Device Action 2
    //
    // For a) and b), we just return the job generated for the dependences. For
    // c) and d) we override the current action with the host/device dependence
    // if the current toolchain is host/device and set the offload dependences
    // info with the jobs obtained from the device/host dependence(s).

    // If there is a single device option or has no host action, just generate
    // the job for it.
    if (OA->hasSingleDeviceDependence() || !OA->hasHostDependence()) {
      InputInfoList DevA;
      OA->doOnEachDeviceDependence([&](Action *DepA, const ToolChain *DepTC,
                                       const char *DepBoundArch) {
        DevA.append(BuildJobsForAction(C, DepA, DepTC, DepBoundArch, AtTopLevel,
                                       /*MultipleArchs*/ !!DepBoundArch,
                                       LinkingOutput, CachedResults,
                                       DepA->getOffloadingDeviceKind()));
      });
      return DevA;
    }

    // If 'Action 2' is host, we generate jobs for the device dependences and
    // override the current action with the host dependence. Otherwise, we
    // generate the host dependences and override the action with the device
    // dependence. The dependences can't therefore be a top-level action.
    OA->doOnEachDependence(
        /*IsHostDependence=*/BuildingForOffloadDevice,
        [&](Action *DepA, const ToolChain *DepTC, const char *DepBoundArch) {
          OffloadDependencesInputInfo.append(BuildJobsForAction(
              C, DepA, DepTC, DepBoundArch, /*AtTopLevel=*/false,
              /*MultipleArchs*/ !!DepBoundArch, LinkingOutput, CachedResults,
              DepA->getOffloadingDeviceKind()));
        });

    A = BuildingForOffloadDevice
            ? OA->getSingleDeviceDependence(/*DoNotConsiderHostActions=*/true)
            : OA->getHostDependence();

    // We may have already built this action as a part of the offloading
    // toolchain, return the cached input if so.
    std::pair<const Action *, std::string> ActionTC = {
        OA->getHostDependence(),
        GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)};
    if (CachedResults.find(ActionTC) != CachedResults.end()) {
      InputInfoList Inputs = CachedResults[ActionTC];
      Inputs.append(OffloadDependencesInputInfo);
      return Inputs;
    }
  }

  if (const InputAction *IA = dyn_cast<InputAction>(A)) {
    // FIXME: It would be nice to not claim this here; maybe the old scheme of
    // just using Args was better?
    const Arg &Input = IA->getInputArg();
    Input.claim();
    if (Input.getOption().matches(options::OPT_INPUT)) {
      const char *Name = Input.getValue();
      return {InputInfo(A, Name, /* _BaseInput = */ Name)};
    }
    return {InputInfo(A, &Input, /* _BaseInput = */ "")};
  }
  if (const BindArchAction *BAA = dyn_cast<BindArchAction>(A)) {
    const ToolChain *TC;
    StringRef ArchName = BAA->getArchName();

    if (!ArchName.empty())
      TC = &getToolChain(C.getArgs(),
                         computeTargetTriple(*this, TargetTriple,
                                             C.getArgs(), ArchName));
    else
      TC = &C.getDefaultToolChain();

    return BuildJobsForAction(C, *BAA->input_begin(), TC, ArchName, AtTopLevel,
                              MultipleArchs, LinkingOutput, CachedResults,
                              TargetDeviceOffloadKind);
  }

  if (const ForEachWrappingAction *FEA = dyn_cast<ForEachWrappingAction>(A)) {
    // Check that the main action wasn't already processed.
    auto MainActionOutput = CachedResults.find(
        {FEA->getJobAction(),
         GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)});
    if (MainActionOutput != CachedResults.end()) {
      // The input was processed on behalf of another foreach.
      // Add entry in cache and return.
      CachedResults[{FEA, GetTriplePlusArchString(TC, BoundArch,
                                                  TargetDeviceOffloadKind)}] =
          MainActionOutput->second;
      return MainActionOutput->second;
    }

    // Build commands for the TFormInput then take any command added after as
    // needing a llvm-foreach wrapping.
    BuildJobsForAction(C, FEA->getTFormInput(), TC, BoundArch,
                       /*AtTopLevel=*/false, MultipleArchs, LinkingOutput,
                       CachedResults, TargetDeviceOffloadKind);
    unsigned OffsetIdx = C.getJobs().size();
    BuildJobsForAction(C, FEA->getJobAction(), TC, BoundArch,
                       /*AtTopLevel=*/false, MultipleArchs, LinkingOutput,
                       CachedResults, TargetDeviceOffloadKind);

    auto begin = C.getJobs().getJobsForOverride().begin() + OffsetIdx;
    auto end = C.getJobs().getJobsForOverride().end();

    // Steal the commands.
    llvm::SmallVector<std::unique_ptr<Command>, 4> JobsToWrap(
        std::make_move_iterator(begin), std::make_move_iterator(end));
    C.getJobs().getJobsForOverride().erase(begin, end);

    InputInfo ActionResult;
    for (std::unique_ptr<Command> Cmd :
         llvm::make_range(std::make_move_iterator(JobsToWrap.begin()),
                          std::make_move_iterator(JobsToWrap.end()))) {
      const JobAction *SourceAction = cast<JobAction>(&Cmd->getSource());
      if (FEA->getSerialActions().count(SourceAction)) {
        C.addCommand(std::move(Cmd));
        continue;
      }
      ActionResult = CachedResults.at(
          {SourceAction,
           GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)}).front();
      InputInfoList InputInfos;
      CollectForEachInputs(InputInfos, SourceAction, TC, BoundArch,
                           TargetDeviceOffloadKind, CachedResults, FEA);
      const Tool *Creator = &Cmd->getCreator();
      StringRef ParallelJobs;
      if (TargetDeviceOffloadKind == Action::OFK_SYCL)
        ParallelJobs = C.getArgs().getLastArgValue(
            options::OPT_fsycl_max_parallel_jobs_EQ);

      tools::SYCL::constructLLVMForeachCommand(
          C, *SourceAction, std::move(Cmd), InputInfos, ActionResult, Creator,
          "", types::getTypeTempSuffix(ActionResult.getType()), ParallelJobs);
    }
    return { ActionResult };
  }

  ActionList Inputs = A->getInputs();

  const JobAction *JA = cast<JobAction>(A);
  ActionList CollapsedOffloadActions;

  auto *DA = dyn_cast<OffloadDepsJobAction>(JA);
  const ToolChain *JATC = DA ? DA->getHostTC() : TC;

  ToolSelector TS(JA, *JATC, C, isSaveTempsEnabled(),
                  embedBitcodeInObject() && !isUsingLTO());
  const Tool *T = TS.getTool(Inputs, CollapsedOffloadActions);

  if (!T)
    return {InputInfo()};

  // If we've collapsed action list that contained OffloadAction we
  // need to build jobs for host/device-side inputs it may have held.
  for (const auto *OA : CollapsedOffloadActions)
    cast<OffloadAction>(OA)->doOnEachDependence(
        /*IsHostDependence=*/BuildingForOffloadDevice,
        [&](Action *DepA, const ToolChain *DepTC, const char *DepBoundArch) {
          OffloadDependencesInputInfo.append(BuildJobsForAction(
              C, DepA, DepTC, DepBoundArch, /* AtTopLevel */ false,
              /*MultipleArchs=*/!!DepBoundArch, LinkingOutput, CachedResults,
              DepA->getOffloadingDeviceKind()));
        });

  // Only use pipes when there is exactly one input.
  InputInfoList InputInfos;
  for (const Action *Input : Inputs) {
    // Treat dsymutil and verify sub-jobs as being at the top-level too, they
    // shouldn't get temporary output names.
    // FIXME: Clean this up.
    bool SubJobAtTopLevel =
        AtTopLevel && (isa<DsymutilJobAction>(A) || isa<VerifyJobAction>(A));
    InputInfos.append(BuildJobsForAction(
        C, Input, JATC, DA ? DA->getOffloadingArch() : BoundArch,
        SubJobAtTopLevel, MultipleArchs, LinkingOutput, CachedResults,
        A->getOffloadingDeviceKind()));
  }

  // Always use the first file input as the base input.
  const char *BaseInput = InputInfos[0].getBaseInput();
  for (auto &Info : InputInfos) {
    if (Info.isFilename()) {
      BaseInput = Info.getBaseInput();
      break;
    }
  }

  // ... except dsymutil actions, which use their actual input as the base
  // input.
  if (JA->getType() == types::TY_dSYM)
    BaseInput = InputInfos[0].getFilename();

  // Append outputs of offload device jobs to the input list
  if (!OffloadDependencesInputInfo.empty())
    InputInfos.append(OffloadDependencesInputInfo.begin(),
                      OffloadDependencesInputInfo.end());

  // Set the effective triple of the toolchain for the duration of this job.
  llvm::Triple EffectiveTriple;
  const ToolChain &ToolTC = T->getToolChain();
  const ArgList &Args =
      C.getArgsForToolChain(TC, BoundArch, A->getOffloadingDeviceKind());
  if (InputInfos.size() != 1) {
    EffectiveTriple = llvm::Triple(ToolTC.ComputeEffectiveClangTriple(Args));
  } else {
    // Pass along the input type if it can be unambiguously determined.
    EffectiveTriple = llvm::Triple(
        ToolTC.ComputeEffectiveClangTriple(Args, InputInfos[0].getType()));
  }
  RegisterEffectiveTriple TripleRAII(ToolTC, EffectiveTriple);

  // Determine the place to write output to, if any.
  InputInfo Result;
  InputInfoList UnbundlingResults;
  if (auto *UA = dyn_cast<OffloadUnbundlingJobAction>(JA)) {
    // If we have an unbundling job, we need to create results for all the
    // outputs. We also update the results cache so that other actions using
    // this unbundling action can get the right results.
    for (auto &UI : UA->getDependentActionsInfo()) {
      assert(UI.DependentOffloadKind != Action::OFK_None &&
             "Unbundling with no offloading??");

      // Unbundling actions are never at the top level. When we generate the
      // offloading prefix, we also do that for the host file because the
      // unbundling action does not change the type of the output which can
      // cause a overwrite.
      InputInfo CurI;
      bool IsFPGAObjLink =
          (JA->getType() == types::TY_Object &&
           EffectiveTriple.getSubArch() == llvm::Triple::SPIRSubArch_fpga &&
           C.getInputArgs().hasArg(options::OPT_fsycl_link_EQ));
      if (C.getDriver().getOffloadStaticLibSeen() &&
          (JA->getType() == types::TY_Archive ||
           JA->getType() == types::TY_Tempfilelist)) {
        // Host part of the unbundled static archive is not used.
        if (UI.DependentOffloadKind == Action::OFK_Host)
          continue;
        // Host part of the unbundled object is not used when using the
        // FPGA target and -fsycl-link is enabled.
        if (UI.DependentOffloadKind == Action::OFK_Host && IsFPGAObjLink)
          continue;
        std::string TmpFileName = C.getDriver().GetTemporaryPath(
            llvm::sys::path::stem(BaseInput),
            JA->getType() == types::TY_Archive ? "a" : "txt");
        const char *TmpFile = C.addTempFile(
            C.getArgs().MakeArgString(TmpFileName), JA->getType());
        CurI = InputInfo(JA->getType(), TmpFile, TmpFile);
      } else if (types::isFPGA(JA->getType())) {
        std::string Ext(types::getTypeTempSuffix(JA->getType()));
        types::ID TI = types::TY_Object;
        if (EffectiveTriple.isSPIR()) {
          if (!UI.DependentToolChain->getTriple().isSPIR())
            continue;
          // Output file from unbundle is FPGA device. Name the file
          // accordingly.
          if (UI.DependentOffloadKind == Action::OFK_Host) {
            // Do not add the current info for Host with FPGA device.  The host
            // side isn't used
            continue;
          }
          if (JA->getType() == types::TY_FPGA_AOCO) {
            TI = types::TY_TempAOCOfilelist;
            Ext = "txt";
          }
          if (JA->getType() == types::TY_FPGA_AOCR ||
              JA->getType() == types::TY_FPGA_AOCX ||
              JA->getType() == types::TY_FPGA_AOCR_EMU) {
            if (IsFPGAObjLink)
              continue;
            // AOCR files are always unbundled into a list file.
            TI = types::TY_Tempfilelist;
          }
        } else {
          if (UI.DependentOffloadKind == Action::OFK_SYCL)
            // Do not add the current info for device with FPGA device.  The
            // device side isn't used
            continue;
          TI = types::TY_Tempfilelist;
          Ext = "txt";
        }
        std::string TmpFileName = C.getDriver().GetTemporaryPath(
            llvm::sys::path::stem(BaseInput), Ext);
        const char *TmpFile =
            C.addTempFile(C.getArgs().MakeArgString(TmpFileName), TI);
        CurI = InputInfo(TI, TmpFile, TmpFile);
      } else {
        // Host part of the unbundled object is not used when -fsycl-link is
        // enabled with FPGA target
        if (UI.DependentOffloadKind == Action::OFK_Host && IsFPGAObjLink)
          continue;
        std::string OffloadingPrefix = Action::GetOffloadingFileNamePrefix(
          UI.DependentOffloadKind,
          UI.DependentToolChain->getTriple().normalize(),
          /*CreatePrefixForHost=*/true);
        CurI = InputInfo(
          UA,
          GetNamedOutputPath(C, *UA, BaseInput, UI.DependentBoundArch,
                             /*AtTopLevel=*/false,
                             MultipleArchs ||
                                 UI.DependentOffloadKind == Action::OFK_HIP,
                             OffloadingPrefix),
          BaseInput);
      }
      // Save the unbundling result.
      UnbundlingResults.push_back(CurI);

      // Get the unique string identifier for this dependence and cache the
      // result.
      StringRef Arch;
      if (TargetDeviceOffloadKind == Action::OFK_HIP ||
          TargetDeviceOffloadKind == Action::OFK_SYCL) {
        if (UI.DependentOffloadKind == Action::OFK_Host)
          Arch = StringRef();
        else
          Arch = UI.DependentBoundArch;
      } else
        Arch = BoundArch;
      // When unbundling for SYCL and there is no Target offload, assume
      // Host as the dependent offload, as the host path has been stripped
      // in this instance
      Action::OffloadKind DependentOffloadKind;
      if (UI.DependentOffloadKind == Action::OFK_SYCL &&
          TargetDeviceOffloadKind == Action::OFK_None)
        DependentOffloadKind = Action::OFK_Host;
      else
        DependentOffloadKind = UI.DependentOffloadKind;

      CachedResults[{A, GetTriplePlusArchString(UI.DependentToolChain, Arch,
                                                DependentOffloadKind)}] = {
          CurI};
    }
    // Do a check for a dependency file unbundle for FPGA.  This is out of line
    // from a regular unbundle, so just create and return the name of the
    // unbundled file.
    if (JA->getType() == types::TY_FPGA_Dependencies ||
        JA->getType() == types::TY_FPGA_Dependencies_List) {
      std::string Ext(types::getTypeTempSuffix(JA->getType()));
      std::string TmpFileName =
          C.getDriver().GetTemporaryPath(llvm::sys::path::stem(BaseInput), Ext);
      const char *TmpFile =
          C.addTempFile(C.getArgs().MakeArgString(TmpFileName), JA->getType());
      Result = InputInfo(JA->getType(), TmpFile, TmpFile);
      UnbundlingResults.push_back(Result);
    } else {
      // Now that we have all the results generated, select the one that should
      // be returned for the current depending action.
      std::pair<const Action *, std::string> ActionTC = {
          A, GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)};
      assert(CachedResults.find(ActionTC) != CachedResults.end() &&
             "Result does not exist??");
      Result = CachedResults[ActionTC].front();
    }
  } else if (auto *DA = dyn_cast<OffloadDepsJobAction>(JA)) {
    for (auto &DI : DA->getDependentActionsInfo()) {
      assert(DI.DependentOffloadKind != Action::OFK_None &&
             "Deps job with no offloading");

      std::string OffloadingPrefix = Action::GetOffloadingFileNamePrefix(
          DI.DependentOffloadKind,
          DI.DependentToolChain->getTriple().normalize(),
          /*CreatePrefixForHost=*/true);
      auto CurI = InputInfo(
          DA,
          GetNamedOutputPath(C, *DA, BaseInput, DI.DependentBoundArch,
                             /*AtTopLevel=*/false,
                             MultipleArchs ||
                                 DI.DependentOffloadKind == Action::OFK_HIP,
                             OffloadingPrefix),
          BaseInput);
      // Save the result.
      UnbundlingResults.push_back(CurI);

      // Get the unique string identifier for this dependence and cache the
      // result.
      StringRef Arch = TargetDeviceOffloadKind == Action::OFK_HIP
                           ? DI.DependentOffloadKind == Action::OFK_Host
                                 ? StringRef()
                                 : DI.DependentBoundArch
                           : BoundArch;

      CachedResults[{A, GetTriplePlusArchString(DI.DependentToolChain, Arch,
                                                DI.DependentOffloadKind)}] = {
          CurI};
    }

    // Now that we have all the results generated, select the one that should be
    // returned for the current depending action.
    std::pair<const Action *, std::string> ActionTC = {
        A, GetTriplePlusArchString(TC, BoundArch, TargetDeviceOffloadKind)};
    auto It = CachedResults.find(ActionTC);
    assert(It != CachedResults.end() && "Result does not exist??");
    Result = It->second.front();
  } else if (JA->getType() == types::TY_Nothing)
    Result = {InputInfo(A, BaseInput)};
  else {
    std::string OffloadingPrefix;
    // When generating binaries with -fsycl-link-target or -fsycl-link, the
    // output file prefix is the triple arch only.  Do not add the arch when
    // compiling for host.
    if (!A->getOffloadingHostActiveKinds() &&
        (Args.getLastArg(options::OPT_fsycl_link_targets_EQ) ||
         Args.hasArg(options::OPT_fsycl_link_EQ))) {
      OffloadingPrefix = "-";
      OffloadingPrefix += TC->getTriple().getArchName();
    } else {
      // We only have to generate a prefix for the host if this is not a
      // top-level action.
      OffloadingPrefix = Action::GetOffloadingFileNamePrefix(
        A->getOffloadingDeviceKind(), TC->getTriple().normalize(),
        /*CreatePrefixForHost=*/isa<OffloadPackagerJobAction>(A) ||
            !(A->getOffloadingHostActiveKinds() == Action::OFK_None ||
              AtTopLevel));
    }
    if (isa<OffloadWrapperJobAction>(JA)) {
      if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o))
        BaseInput = FinalOutput->getValue();
      // Do not use the default image name when using -fno-sycl-rdc
      else if (!tools::SYCL::shouldDoPerObjectFileLinking(C))
        BaseInput = getDefaultImageName();
      BaseInput =
          C.getArgs().MakeArgString(std::string(BaseInput) + "-wrapper");
    }
    Result = InputInfo(A, GetNamedOutputPath(C, *JA, BaseInput, BoundArch,
                                             AtTopLevel, MultipleArchs,
                                             OffloadingPrefix),
                       BaseInput);
    if (T->canEmitIR() && OffloadingPrefix.empty())
      handleTimeTrace(C, Args, JA, BaseInput, Result);
  }

  if (CCCPrintBindings && !CCGenDiagnostics) {
    llvm::errs() << "# \"" << T->getToolChain().getTripleString() << '"'
                 << " - \"" << T->getName() << "\", inputs: [";
    for (unsigned i = 0, e = InputInfos.size(); i != e; ++i) {
      llvm::errs() << InputInfos[i].getAsString();
      if (i + 1 != e)
        llvm::errs() << ", ";
    }
    if (UnbundlingResults.empty())
      llvm::errs() << "], output: " << Result.getAsString() << "\n";
    else {
      llvm::errs() << "], outputs: [";
      for (unsigned i = 0, e = UnbundlingResults.size(); i != e; ++i) {
        llvm::errs() << UnbundlingResults[i].getAsString();
        if (i + 1 != e)
          llvm::errs() << ", ";
      }
      llvm::errs() << "] \n";
    }
  } else {
    if (UnbundlingResults.empty())
      T->ConstructJob(
          C, *JA, Result, InputInfos,
          C.getArgsForToolChain(TC, BoundArch, JA->getOffloadingDeviceKind()),
          LinkingOutput);
    else
      T->ConstructJobMultipleOutputs(
          C, *JA, UnbundlingResults, InputInfos,
          C.getArgsForToolChain(TC, BoundArch, JA->getOffloadingDeviceKind()),
          LinkingOutput);
  }
  return {Result};
}

const char *Driver::getDefaultImageName() const {
  llvm::Triple Target(llvm::Triple::normalize(TargetTriple));
  return Target.isOSWindows() ? "a.exe" : "a.out";
}

/// Create output filename based on ArgValue, which could either be a
/// full filename, filename without extension, or a directory. If ArgValue
/// does not provide a filename, then use BaseName, and use the extension
/// suitable for FileType.
static const char *MakeCLOutputFilename(const ArgList &Args, StringRef ArgValue,
                                        StringRef BaseName,
                                        types::ID FileType) {
  SmallString<128> Filename = ArgValue;

  if (ArgValue.empty()) {
    // If the argument is empty, output to BaseName in the current dir.
    Filename = BaseName;
  } else if (llvm::sys::path::is_separator(Filename.back())) {
    // If the argument is a directory, output to BaseName in that dir.
    llvm::sys::path::append(Filename, BaseName);
  }

  if (!llvm::sys::path::has_extension(ArgValue)) {
    // If the argument didn't provide an extension, then set it.
    const char *Extension = types::getTypeTempSuffix(FileType, true);

    if (FileType == types::TY_Image &&
        Args.hasArg(options::OPT__SLASH_LD, options::OPT__SLASH_LDd)) {
      // The output file is a dll.
      Extension = "dll";
    }

    llvm::sys::path::replace_extension(Filename, Extension);
  }

  return Args.MakeArgString(Filename.c_str());
}

static bool HasPreprocessOutput(const Action &JA) {
  if (isa<PreprocessJobAction>(JA))
    return true;
  if (isa<OffloadAction>(JA) && isa<PreprocessJobAction>(JA.getInputs()[0]))
    return true;
  if (isa<OffloadBundlingJobAction>(JA) &&
      HasPreprocessOutput(*(JA.getInputs()[0])))
    return true;
  return false;
}

const char *Driver::CreateTempFile(Compilation &C, StringRef Prefix,
                                   StringRef Suffix, bool MultipleArchs,
                                   StringRef BoundArch,
                                   types::ID Type,
                                   bool NeedUniqueDirectory) const {
  SmallString<128> TmpName;
  Arg *A = C.getArgs().getLastArg(options::OPT_fcrash_diagnostics_dir);
  std::optional<std::string> CrashDirectory =
      CCGenDiagnostics && A
          ? std::string(A->getValue())
          : llvm::sys::Process::GetEnv("CLANG_CRASH_DIAGNOSTICS_DIR");
  if (CrashDirectory) {
    if (!getVFS().exists(*CrashDirectory))
      llvm::sys::fs::create_directories(*CrashDirectory);
    SmallString<128> Path(*CrashDirectory);
    llvm::sys::path::append(Path, Prefix);
    const char *Middle = !Suffix.empty() ? "-%%%%%%." : "-%%%%%%";
    if (std::error_code EC =
            llvm::sys::fs::createUniqueFile(Path + Middle + Suffix, TmpName)) {
      Diag(clang::diag::err_unable_to_make_temp) << EC.message();
      return "";
    }
  } else {
    if (MultipleArchs && !BoundArch.empty()) {
      if (NeedUniqueDirectory) {
        TmpName = GetTemporaryDirectory(Prefix);
        llvm::sys::path::append(TmpName,
                                Twine(Prefix) + "-" + BoundArch + "." + Suffix);
      } else {
        TmpName =
            GetTemporaryPath((Twine(Prefix) + "-" + BoundArch).str(), Suffix);
      }

    } else {
      TmpName = GetTemporaryPath(Prefix, Suffix);
    }
  }
  return C.addTempFile(C.getArgs().MakeArgString(TmpName), Type);
}

// Calculate the output path of the module file when compiling a module unit
// with the `-fmodule-output` option or `-fmodule-output=` option specified.
// The behavior is:
// - If `-fmodule-output=` is specfied, then the module file is
//   writing to the value.
// - Otherwise if the output object file of the module unit is specified, the
// output path
//   of the module file should be the same with the output object file except
//   the corresponding suffix. This requires both `-o` and `-c` are specified.
// - Otherwise, the output path of the module file will be the same with the
//   input with the corresponding suffix.
static const char *GetModuleOutputPath(Compilation &C, const JobAction &JA,
                                       const char *BaseInput) {
  assert(isa<PrecompileJobAction>(JA) && JA.getType() == types::TY_ModuleFile &&
         (C.getArgs().hasArg(options::OPT_fmodule_output) ||
          C.getArgs().hasArg(options::OPT_fmodule_output_EQ)));

  if (Arg *ModuleOutputEQ =
          C.getArgs().getLastArg(options::OPT_fmodule_output_EQ))
    return C.addResultFile(ModuleOutputEQ->getValue(), &JA);

  SmallString<64> OutputPath;
  Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);
  if (FinalOutput && C.getArgs().hasArg(options::OPT_c))
    OutputPath = FinalOutput->getValue();
  else
    OutputPath = BaseInput;

  const char *Extension = types::getTypeTempSuffix(JA.getType());
  llvm::sys::path::replace_extension(OutputPath, Extension);
  return C.addResultFile(C.getArgs().MakeArgString(OutputPath.c_str()), &JA);
}

const char *Driver::GetNamedOutputPath(Compilation &C, const JobAction &JA,
                                       const char *BaseInput,
                                       StringRef OrigBoundArch, bool AtTopLevel,
                                       bool MultipleArchs,
                                       StringRef OffloadingPrefix) const {
  std::string BoundArch = OrigBoundArch.str();
  if (is_style_windows(llvm::sys::path::Style::native)) {
    // BoundArch may contains ':', which is invalid in file names on Windows,
    // therefore replace it with '%'.
    std::replace(BoundArch.begin(), BoundArch.end(), ':', '@');
  }

  llvm::PrettyStackTraceString CrashInfo("Computing output path");
  // Output to a user requested destination?
  if (AtTopLevel && !isa<DsymutilJobAction>(JA) && !isa<VerifyJobAction>(JA)) {
    if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o))
      return C.addResultFile(FinalOutput->getValue(), &JA);
    // Output to destination for -fsycl-device-only and Windows -o
    if (C.getArgs().hasArg(options::OPT_fsycl_device_only))
      if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT__SLASH_o))
        return C.addResultFile(FinalOutput->getValue(), &JA);
  }

  // For /P, preprocess to file named after BaseInput.
  if (C.getArgs().hasArg(options::OPT__SLASH_P) &&
      ((AtTopLevel && isa<PreprocessJobAction>(JA)) ||
       isa<OffloadBundlingJobAction>(JA))) {
    StringRef BaseName = llvm::sys::path::filename(BaseInput);
    StringRef NameArg;
    if (Arg *A = C.getArgs().getLastArg(options::OPT__SLASH_Fi))
      NameArg = A->getValue();
    return C.addResultFile(
        MakeCLOutputFilename(C.getArgs(), NameArg, BaseName, types::TY_PP_C),
        &JA);
  }

  // Redirect output for the generated source + integration footer.
  if (isa<AppendFooterJobAction>(JA)) {
    if (Arg *A = C.getArgs().getLastArg(options::OPT_fsycl_footer_path_EQ)) {
      SmallString<128> OutName(A->getValue());
      StringRef BaseName = llvm::sys::path::filename(BaseInput);
      if (isSaveTempsEnabled()) {
        // Retain the location specified by the user with -save-temps.
        const char *Suffix = types::getTypeTempSuffix(JA.getType());
        std::string::size_type End = std::string::npos;
        if (!types::appendSuffixForType(JA.getType()))
          End = BaseName.rfind('.');
        SmallString<128> Suffixed(BaseName.substr(0, End));
        Suffixed += OffloadingPrefix;
        Suffixed += '.';
        Suffixed += Suffix;
        llvm::sys::path::append(OutName, Suffixed.c_str());
      } else {
        std::string TmpName =
            GetTemporaryPath(llvm::sys::path::stem(BaseName),
                             types::getTypeTempSuffix(JA.getType()));
        llvm::sys::path::append(OutName, llvm::sys::path::filename(TmpName));
      }
      return C.addTempFile(C.getArgs().MakeArgString(OutName));
    }
  }

  // Default to writing to stdout?
  if (AtTopLevel && !CCGenDiagnostics && HasPreprocessOutput(JA)) {
    return "-";
  }

  if (JA.getType() == types::TY_ModuleFile &&
      C.getArgs().getLastArg(options::OPT_module_file_info)) {
    return "-";
  }

  if (JA.getType() == types::TY_PP_Asm &&
      C.getArgs().hasArg(options::OPT_dxc_Fc)) {
    StringRef FcValue = C.getArgs().getLastArgValue(options::OPT_dxc_Fc);
    // TODO: Should we use `MakeCLOutputFilename` here? If so, we can probably
    // handle this as part of the SLASH_Fa handling below.
    return C.addResultFile(C.getArgs().MakeArgString(FcValue.str()), &JA);
  }

  if (JA.getType() == types::TY_Object &&
      C.getArgs().hasArg(options::OPT_dxc_Fo)) {
    StringRef FoValue = C.getArgs().getLastArgValue(options::OPT_dxc_Fo);
    // TODO: Should we use `MakeCLOutputFilename` here? If so, we can probably
    // handle this as part of the SLASH_Fo handling below.
    return C.addResultFile(C.getArgs().MakeArgString(FoValue.str()), &JA);
  }

  // Is this the assembly listing for /FA?
  if (JA.getType() == types::TY_PP_Asm &&
      (C.getArgs().hasArg(options::OPT__SLASH_FA) ||
       C.getArgs().hasArg(options::OPT__SLASH_Fa))) {
    // Use /Fa and the input filename to determine the asm file name.
    StringRef BaseName = llvm::sys::path::filename(BaseInput);
    StringRef FaValue = C.getArgs().getLastArgValue(options::OPT__SLASH_Fa);
    return C.addResultFile(
        MakeCLOutputFilename(C.getArgs(), FaValue, BaseName, JA.getType()),
        &JA);
  }

  // DXC defaults to standard out when generating assembly. We check this after
  // any DXC flags that might specify a file.
  if (AtTopLevel && JA.getType() == types::TY_PP_Asm && IsDXCMode())
    return "-";

  bool SpecifiedModuleOutput =
      C.getArgs().hasArg(options::OPT_fmodule_output) ||
      C.getArgs().hasArg(options::OPT_fmodule_output_EQ);
  if (MultipleArchs && SpecifiedModuleOutput)
    Diag(clang::diag::err_drv_module_output_with_multiple_arch);

  // If we're emitting a module output with the specified option
  // `-fmodule-output`.
  if (!AtTopLevel && isa<PrecompileJobAction>(JA) &&
      JA.getType() == types::TY_ModuleFile && SpecifiedModuleOutput)
    return GetModuleOutputPath(C, JA, BaseInput);

  // Output to a temporary file?
  if ((!AtTopLevel && !isSaveTempsEnabled() &&
       (!C.getArgs().hasArg(options::OPT__SLASH_Fo) ||
        // FIXME - The use of /Fo is limited when offloading is enabled.  When
        // compiling to exe use of /Fo does not produce the named obj.  We also
        // should not use the named output when performing unbundling.
        (C.getArgs().hasArg(options::OPT__SLASH_Fo) &&
         (!JA.isOffloading(Action::OFK_None) ||
          isa<OffloadUnbundlingJobAction>(JA) ||
          JA.getOffloadingHostActiveKinds() > Action::OFK_Host)))) ||
      CCGenDiagnostics) {
    StringRef Name = llvm::sys::path::filename(BaseInput);
    std::pair<StringRef, StringRef> Split = Name.split('.');
    const char *Suffix =
        types::getTypeTempSuffix(JA.getType(), IsCLMode() || IsDXCMode());
    // The non-offloading toolchain on Darwin requires deterministic input
    // file name for binaries to be deterministic, therefore it needs unique
    // directory.
    llvm::Triple Triple(C.getDriver().getTargetTriple());
    bool NeedUniqueDirectory =
        (JA.getOffloadingDeviceKind() == Action::OFK_None ||
         JA.getOffloadingDeviceKind() == Action::OFK_Host) &&
        Triple.isOSDarwin();
    return CreateTempFile(C, Split.first, Suffix, MultipleArchs, BoundArch,
                          JA.getType(), NeedUniqueDirectory);
  }

  SmallString<128> BasePath(BaseInput);
  SmallString<128> ExternalPath("");
  StringRef BaseName;

  // Dsymutil actions should use the full path.
  if (isa<DsymutilJobAction>(JA) && C.getArgs().hasArg(options::OPT_dsym_dir)) {
    ExternalPath += C.getArgs().getLastArg(options::OPT_dsym_dir)->getValue();
    // We use posix style here because the tests (specifically
    // darwin-dsymutil.c) demonstrate that posix style paths are acceptable
    // even on Windows and if we don't then the similar test covering this
    // fails.
    llvm::sys::path::append(ExternalPath, llvm::sys::path::Style::posix,
                            llvm::sys::path::filename(BasePath));
    BaseName = ExternalPath;
  } else if (isa<DsymutilJobAction>(JA) || isa<VerifyJobAction>(JA))
    BaseName = BasePath;
  else
    BaseName = llvm::sys::path::filename(BasePath);

  // Determine what the derived output name should be.
  const char *NamedOutput;

  if ((JA.getType() == types::TY_Object || JA.getType() == types::TY_LTO_BC ||
       JA.getType() == types::TY_Archive) &&
      C.getArgs().hasArg(options::OPT__SLASH_Fo, options::OPT__SLASH_o)) {
    // The /Fo or /o flag decides the object filename.
    StringRef Val =
        C.getArgs()
            .getLastArg(options::OPT__SLASH_Fo, options::OPT__SLASH_o)
            ->getValue();
    NamedOutput =
        MakeCLOutputFilename(C.getArgs(), Val, BaseName, types::TY_Object);
  } else if (JA.getType() == types::TY_Image &&
             C.getArgs().hasArg(options::OPT__SLASH_Fe,
                                options::OPT__SLASH_o)) {
    // The /Fe or /o flag names the linked file.
    StringRef Val =
        C.getArgs()
            .getLastArg(options::OPT__SLASH_Fe, options::OPT__SLASH_o)
            ->getValue();
    NamedOutput =
        MakeCLOutputFilename(C.getArgs(), Val, BaseName, types::TY_Image);
  } else if (JA.getType() == types::TY_Image) {
    if (IsCLMode()) {
      // clang-cl uses BaseName for the executable name.
      NamedOutput =
          MakeCLOutputFilename(C.getArgs(), "", BaseName, types::TY_Image);
    } else {
      SmallString<128> Output(getDefaultImageName());
      // HIP image for device compilation with -fno-gpu-rdc is per compilation
      // unit.
      bool IsHIPNoRDC = JA.getOffloadingDeviceKind() == Action::OFK_HIP &&
                        !C.getArgs().hasFlag(options::OPT_fgpu_rdc,
                                             options::OPT_fno_gpu_rdc, false);
      bool UseOutExtension = IsHIPNoRDC || isa<OffloadPackagerJobAction>(JA);
      if (UseOutExtension) {
        Output = BaseName;
        llvm::sys::path::replace_extension(Output, "");
      }
      Output += OffloadingPrefix;
      if (MultipleArchs && !BoundArch.empty()) {
        Output += "-";
        Output.append(BoundArch);
      }
      if (UseOutExtension)
        Output += ".out";
      NamedOutput = C.getArgs().MakeArgString(Output.c_str());
    }
  } else if (JA.getType() == types::TY_PCH && IsCLMode()) {
    NamedOutput = C.getArgs().MakeArgString(GetClPchPath(C, BaseName));
  } else if ((JA.getType() == types::TY_Plist || JA.getType() == types::TY_AST) &&
             C.getArgs().hasArg(options::OPT__SLASH_o)) {
    StringRef Val =
        C.getArgs()
            .getLastArg(options::OPT__SLASH_o)
            ->getValue();
    NamedOutput =
        MakeCLOutputFilename(C.getArgs(), Val, BaseName, types::TY_Object);
  } else {
    const char *Suffix =
        types::getTypeTempSuffix(JA.getType(), IsCLMode() || IsDXCMode());
    assert(Suffix && "All types used for output should have a suffix.");

    std::string::size_type End = std::string::npos;
    if (!types::appendSuffixForType(JA.getType()))
      End = BaseName.rfind('.');
    SmallString<128> Suffixed(BaseName.substr(0, End));
    Suffixed += OffloadingPrefix;
    if (MultipleArchs && !BoundArch.empty()) {
      Suffixed += "-";
      Suffixed.append(BoundArch);
    }
    // When using both -save-temps and -emit-llvm, use a ".tmp.bc" suffix for
    // the unoptimized bitcode so that it does not get overwritten by the ".bc"
    // optimized bitcode output.
    auto IsAMDRDCInCompilePhase = [](const JobAction &JA,
                                     const llvm::opt::DerivedArgList &Args) {
      // The relocatable compilation in HIP and OpenMP implies -emit-llvm.
      // Similarly, use a ".tmp.bc" suffix for the unoptimized bitcode
      // (generated in the compile phase.)
      const ToolChain *TC = JA.getOffloadingToolChain();
      return isa<CompileJobAction>(JA) &&
             ((JA.getOffloadingDeviceKind() == Action::OFK_HIP &&
               Args.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                            false)) ||
              (JA.getOffloadingDeviceKind() == Action::OFK_OpenMP && TC &&
               TC->getTriple().isAMDGPU()));
    };
    if (!AtTopLevel && JA.getType() == types::TY_LLVM_BC &&
        (C.getArgs().hasArg(options::OPT_emit_llvm) ||
         IsAMDRDCInCompilePhase(JA, C.getArgs())))
      Suffixed += ".tmp";
    Suffixed += '.';
    Suffixed += Suffix;
    NamedOutput = C.getArgs().MakeArgString(Suffixed.c_str());
  }

  // Prepend object file path if -save-temps=obj
  if (!AtTopLevel && isSaveTempsObj() && C.getArgs().hasArg(options::OPT_o) &&
      JA.getType() != types::TY_PCH) {
    Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);
    SmallString<128> TempPath(FinalOutput->getValue());
    llvm::sys::path::remove_filename(TempPath);
    StringRef OutputFileName = llvm::sys::path::filename(NamedOutput);
    llvm::sys::path::append(TempPath, OutputFileName);
    NamedOutput = C.getArgs().MakeArgString(TempPath.c_str());
  }

  if (isSaveTempsEnabled()) {
    // If we're saving temps and the temp file conflicts with any
    // input/resulting file, then avoid overwriting.
    if (!AtTopLevel && NamedOutput == BaseName) {
      bool SameFile = false;
      SmallString<256> Result;
      llvm::sys::fs::current_path(Result);
      llvm::sys::path::append(Result, BaseName);
      llvm::sys::fs::equivalent(BaseInput, Result.c_str(), SameFile);
      // Must share the same path to conflict.
      if (SameFile) {
        StringRef Name = llvm::sys::path::filename(BaseInput);
        std::pair<StringRef, StringRef> Split = Name.split('.');
        std::string TmpName = GetTemporaryPath(
            Split.first,
            types::getTypeTempSuffix(JA.getType(), IsCLMode() || IsDXCMode()));
        return C.addTempFile(C.getArgs().MakeArgString(TmpName));
      }
    }

    const auto &ResultFiles = C.getResultFiles();
    const auto CollidingFilenameIt =
        llvm::find_if(ResultFiles, [NamedOutput](const auto &It) {
          return StringRef(NamedOutput).equals(It.second);
        });
    if (CollidingFilenameIt != ResultFiles.end()) {
      // Upon any collision, a unique hash will be appended to the filename,
      // similar to what is done for temporary files in the regular flow.
      StringRef CollidingName(CollidingFilenameIt->second);
      std::pair<StringRef, StringRef> Split = CollidingName.split('.');
      std::string UniqueName = GetUniquePath(
          Split.first,
          types::getTypeTempSuffix(JA.getType(), IsCLMode() || IsDXCMode()));
      return C.addTempFile(C.getArgs().MakeArgString(UniqueName));
    }
  }

  // Emit an error if PCH(Pre-Compiled Header) file generation is forced in
  // -fsycl mode.
  if (C.getArgs().hasFlag(options::OPT_fsycl, options::OPT_fno_sycl, false) &&
      JA.getType() == types::TY_PCH)
    Diag(clang::diag::err_drv_fsycl_with_pch);
  // As an annoying special case, PCH generation doesn't strip the pathname.
  if (JA.getType() == types::TY_PCH && !IsCLMode()) {
    llvm::sys::path::remove_filename(BasePath);
    if (BasePath.empty())
      BasePath = NamedOutput;
    else
      llvm::sys::path::append(BasePath, NamedOutput);
    return C.addResultFile(C.getArgs().MakeArgString(BasePath.c_str()), &JA);
  }

  return C.addResultFile(NamedOutput, &JA);
}

std::string Driver::GetFilePath(StringRef Name, const ToolChain &TC) const {
  // Search for Name in a list of paths.
  auto SearchPaths = [&](const llvm::SmallVectorImpl<std::string> &P)
      -> std::optional<std::string> {
    // Respect a limited subset of the '-Bprefix' functionality in GCC by
    // attempting to use this prefix when looking for file paths.
    for (const auto &Dir : P) {
      if (Dir.empty())
        continue;
      SmallString<128> P(Dir[0] == '=' ? SysRoot + Dir.substr(1) : Dir);
      llvm::sys::path::append(P, Name);
      if (llvm::sys::fs::exists(Twine(P)))
        return std::string(P);
    }
    return std::nullopt;
  };

  if (auto P = SearchPaths(PrefixDirs))
    return *P;

  SmallString<128> R(ResourceDir);
  llvm::sys::path::append(R, Name);
  if (llvm::sys::fs::exists(Twine(R)))
    return std::string(R);

  SmallString<128> P(TC.getCompilerRTPath());
  llvm::sys::path::append(P, Name);
  if (llvm::sys::fs::exists(Twine(P)))
    return std::string(P);

  SmallString<128> D(Dir);
  llvm::sys::path::append(D, "..", Name);
  if (llvm::sys::fs::exists(Twine(D)))
    return std::string(D);

  if (auto P = SearchPaths(TC.getLibraryPaths()))
    return *P;

  if (auto P = SearchPaths(TC.getFilePaths()))
    return *P;

  return std::string(Name);
}

void Driver::generatePrefixedToolNames(
    StringRef Tool, const ToolChain &TC,
    SmallVectorImpl<std::string> &Names) const {
  // FIXME: Needs a better variable than TargetTriple
  Names.emplace_back((TargetTriple + "-" + Tool).str());
  Names.emplace_back(Tool);
}

static bool ScanDirForExecutable(SmallString<128> &Dir, StringRef Name) {
  llvm::sys::path::append(Dir, Name);
  if (llvm::sys::fs::can_execute(Twine(Dir)))
    return true;
  llvm::sys::path::remove_filename(Dir);
  return false;
}

std::string Driver::GetProgramPath(StringRef Name, const ToolChain &TC) const {
  SmallVector<std::string, 2> TargetSpecificExecutables;
  generatePrefixedToolNames(Name, TC, TargetSpecificExecutables);

  // Respect a limited subset of the '-Bprefix' functionality in GCC by
  // attempting to use this prefix when looking for program paths.
  for (const auto &PrefixDir : PrefixDirs) {
    if (llvm::sys::fs::is_directory(PrefixDir)) {
      SmallString<128> P(PrefixDir);
      if (ScanDirForExecutable(P, Name))
        return std::string(P);
    } else {
      SmallString<128> P((PrefixDir + Name).str());
      if (llvm::sys::fs::can_execute(Twine(P)))
        return std::string(P);
    }
  }

  const ToolChain::path_list &List = TC.getProgramPaths();
  for (const auto &TargetSpecificExecutable : TargetSpecificExecutables) {
    // For each possible name of the tool look for it in
    // program paths first, then the path.
    // Higher priority names will be first, meaning that
    // a higher priority name in the path will be found
    // instead of a lower priority name in the program path.
    // E.g. <triple>-gcc on the path will be found instead
    // of gcc in the program path
    for (const auto &Path : List) {
      SmallString<128> P(Path);
      if (ScanDirForExecutable(P, TargetSpecificExecutable))
        return std::string(P);
    }

    // Fall back to the path
    if (llvm::ErrorOr<std::string> P =
            llvm::sys::findProgramByName(TargetSpecificExecutable))
      return *P;
  }

  return std::string(Name);
}

std::string Driver::GetTemporaryPath(StringRef Prefix, StringRef Suffix) const {
  SmallString<128> Path;
  std::error_code EC = llvm::sys::fs::createTemporaryFile(Prefix, Suffix, Path);
  if (EC) {
    Diag(clang::diag::err_unable_to_make_temp) << EC.message();
    return "";
  }

  return std::string(Path);
}

std::string Driver::GetUniquePath(StringRef BaseName, StringRef Ext) const {
  SmallString<128> Path;
  std::error_code EC = llvm::sys::fs::getPotentiallyUniqueFileName(
      Twine(BaseName) + Twine("-%%%%%%.") + Ext, Path);
  if (EC) {
    Diag(clang::diag::err_unable_to_make_temp) << EC.message();
    return "";
  }

  return std::string(Path.str());
}

std::string Driver::GetTemporaryDirectory(StringRef Prefix) const {
  SmallString<128> Path;
  std::error_code EC = llvm::sys::fs::createUniqueDirectory(Prefix, Path);
  if (EC) {
    Diag(clang::diag::err_unable_to_make_temp) << EC.message();
    return "";
  }

  return std::string(Path);
}

std::string Driver::GetClPchPath(Compilation &C, StringRef BaseName) const {
  SmallString<128> Output;
  if (Arg *FpArg = C.getArgs().getLastArg(options::OPT__SLASH_Fp)) {
    // FIXME: If anybody needs it, implement this obscure rule:
    // "If you specify a directory without a file name, the default file name
    // is VCx0.pch., where x is the major version of Visual C++ in use."
    Output = FpArg->getValue();

    // "If you do not specify an extension as part of the path name, an
    // extension of .pch is assumed. "
    if (!llvm::sys::path::has_extension(Output))
      Output += ".pch";
  } else {
    if (Arg *YcArg = C.getArgs().getLastArg(options::OPT__SLASH_Yc))
      Output = YcArg->getValue();
    if (Output.empty())
      Output = BaseName;
    llvm::sys::path::replace_extension(Output, ".pch");
  }
  return std::string(Output);
}

const ToolChain &Driver::getToolChain(const ArgList &Args,
                                      const llvm::Triple &Target) const {

  auto &TC = ToolChains[Target.str()];
  if (!TC) {
    switch (Target.getOS()) {
    case llvm::Triple::AIX:
      TC = std::make_unique<toolchains::AIX>(*this, Target, Args);
      break;
    case llvm::Triple::Haiku:
      TC = std::make_unique<toolchains::Haiku>(*this, Target, Args);
      break;
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS:
    case llvm::Triple::TvOS:
    case llvm::Triple::WatchOS:
    case llvm::Triple::XROS:
    case llvm::Triple::DriverKit:
      TC = std::make_unique<toolchains::DarwinClang>(*this, Target, Args);
      break;
    case llvm::Triple::DragonFly:
      TC = std::make_unique<toolchains::DragonFly>(*this, Target, Args);
      break;
    case llvm::Triple::OpenBSD:
      TC = std::make_unique<toolchains::OpenBSD>(*this, Target, Args);
      break;
    case llvm::Triple::NetBSD:
      TC = std::make_unique<toolchains::NetBSD>(*this, Target, Args);
      break;
    case llvm::Triple::FreeBSD:
      if (Target.isPPC())
        TC = std::make_unique<toolchains::PPCFreeBSDToolChain>(*this, Target,
                                                               Args);
      else
        TC = std::make_unique<toolchains::FreeBSD>(*this, Target, Args);
      break;
    case llvm::Triple::Linux:
    case llvm::Triple::ELFIAMCU:
      if (Target.getArch() == llvm::Triple::hexagon)
        TC = std::make_unique<toolchains::HexagonToolChain>(*this, Target,
                                                             Args);
      else if ((Target.getVendor() == llvm::Triple::MipsTechnologies) &&
               !Target.hasEnvironment())
        TC = std::make_unique<toolchains::MipsLLVMToolChain>(*this, Target,
                                                              Args);
      else if (Target.isPPC())
        TC = std::make_unique<toolchains::PPCLinuxToolChain>(*this, Target,
                                                              Args);
      else if (Target.getArch() == llvm::Triple::ve)
        TC = std::make_unique<toolchains::VEToolChain>(*this, Target, Args);
      else if (Target.isOHOSFamily())
        TC = std::make_unique<toolchains::OHOS>(*this, Target, Args);
      else
        TC = std::make_unique<toolchains::Linux>(*this, Target, Args);
      break;
    case llvm::Triple::NaCl:
      TC = std::make_unique<toolchains::NaClToolChain>(*this, Target, Args);
      break;
    case llvm::Triple::Fuchsia:
      TC = std::make_unique<toolchains::Fuchsia>(*this, Target, Args);
      break;
    case llvm::Triple::Solaris:
      TC = std::make_unique<toolchains::Solaris>(*this, Target, Args);
      break;
    case llvm::Triple::CUDA:
      TC = std::make_unique<toolchains::NVPTXToolChain>(*this, Target, Args);
      break;
    case llvm::Triple::AMDHSA:
      TC = std::make_unique<toolchains::ROCMToolChain>(*this, Target, Args);
      break;
    case llvm::Triple::AMDPAL:
    case llvm::Triple::Mesa3D:
      TC = std::make_unique<toolchains::AMDGPUToolChain>(*this, Target, Args);
      break;
    case llvm::Triple::Win32:
      switch (Target.getEnvironment()) {
      default:
        if (Target.isOSBinFormatELF())
          TC = std::make_unique<toolchains::Generic_ELF>(*this, Target, Args);
        else if (Target.isOSBinFormatMachO())
          TC = std::make_unique<toolchains::MachO>(*this, Target, Args);
        else
          TC = std::make_unique<toolchains::Generic_GCC>(*this, Target, Args);
        break;
      case llvm::Triple::GNU:
        TC = std::make_unique<toolchains::MinGW>(*this, Target, Args);
        break;
      case llvm::Triple::Itanium:
        TC = std::make_unique<toolchains::CrossWindowsToolChain>(*this, Target,
                                                                  Args);
        break;
      case llvm::Triple::MSVC:
      case llvm::Triple::UnknownEnvironment:
        if (Args.getLastArgValue(options::OPT_fuse_ld_EQ)
                .starts_with_insensitive("bfd"))
          TC = std::make_unique<toolchains::CrossWindowsToolChain>(
              *this, Target, Args);
        else
          TC =
              std::make_unique<toolchains::MSVCToolChain>(*this, Target, Args);
        break;
      }
      break;
    case llvm::Triple::PS4:
      TC = std::make_unique<toolchains::PS4CPU>(*this, Target, Args);
      break;
    case llvm::Triple::PS5:
      TC = std::make_unique<toolchains::PS5CPU>(*this, Target, Args);
      break;
    case llvm::Triple::Hurd:
      TC = std::make_unique<toolchains::Hurd>(*this, Target, Args);
      break;
    case llvm::Triple::LiteOS:
      TC = std::make_unique<toolchains::OHOS>(*this, Target, Args);
      break;
    case llvm::Triple::ZOS:
      TC = std::make_unique<toolchains::ZOS>(*this, Target, Args);
      break;
    case llvm::Triple::ShaderModel:
      TC = std::make_unique<toolchains::HLSLToolChain>(*this, Target, Args);
      break;
    default:
      // Of these targets, Hexagon is the only one that might have
      // an OS of Linux, in which case it got handled above already.
      switch (Target.getArch()) {
      case llvm::Triple::tce:
        TC = std::make_unique<toolchains::TCEToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::tcele:
        TC = std::make_unique<toolchains::TCELEToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::hexagon:
        TC = std::make_unique<toolchains::HexagonToolChain>(*this, Target,
                                                             Args);
        break;
      case llvm::Triple::lanai:
        TC = std::make_unique<toolchains::LanaiToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::xcore:
        TC = std::make_unique<toolchains::XCoreToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::wasm32:
      case llvm::Triple::wasm64:
        TC = std::make_unique<toolchains::WebAssembly>(*this, Target, Args);
        break;
      case llvm::Triple::avr:
        TC = std::make_unique<toolchains::AVRToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::msp430:
        TC =
            std::make_unique<toolchains::MSP430ToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::riscv32:
      case llvm::Triple::riscv64:
        if (toolchains::RISCVToolChain::hasGCCToolchain(*this, Args))
          TC =
              std::make_unique<toolchains::RISCVToolChain>(*this, Target, Args);
        else
          TC = std::make_unique<toolchains::BareMetal>(*this, Target, Args);
        break;
      case llvm::Triple::ve:
        TC = std::make_unique<toolchains::VEToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::spirv32:
      case llvm::Triple::spirv64:
        TC = std::make_unique<toolchains::SPIRVToolChain>(*this, Target, Args);
        break;
      case llvm::Triple::csky:
        TC = std::make_unique<toolchains::CSKYToolChain>(*this, Target, Args);
        break;
      default:
        if (toolchains::BareMetal::handlesTarget(Target))
          TC = std::make_unique<toolchains::BareMetal>(*this, Target, Args);
        else if (Target.isOSBinFormatELF())
          TC = std::make_unique<toolchains::Generic_ELF>(*this, Target, Args);
        else if (Target.isOSBinFormatMachO())
          TC = std::make_unique<toolchains::MachO>(*this, Target, Args);
        else
          TC = std::make_unique<toolchains::Generic_GCC>(*this, Target, Args);
      }
    }
  }

  return *TC;
}

const ToolChain &Driver::getOffloadingDeviceToolChain(
    const ArgList &Args, const llvm::Triple &Target, const ToolChain &HostTC,
    const Action::OffloadKind &TargetDeviceOffloadKind) const {
  // Use device / host triples offload kind as the key into the ToolChains map
  // because the device ToolChain we create depends on both.
  auto &TC = ToolChains[Target.str() + "/" + HostTC.getTriple().str() +
                        std::to_string(TargetDeviceOffloadKind)];
  if (!TC) {
    // Categorized by offload kind > arch rather than OS > arch like
    // the normal getToolChain call, as it seems a reasonable way to categorize
    // things.
    switch (TargetDeviceOffloadKind) {
    case Action::OFK_Cuda:
      TC = std::make_unique<toolchains::CudaToolChain>(
          *this, Target, HostTC, Args, TargetDeviceOffloadKind);
      break;
    case Action::OFK_HIP: {
      if (Target.getArch() == llvm::Triple::amdgcn &&
          Target.getVendor() == llvm::Triple::AMD &&
          Target.getOS() == llvm::Triple::AMDHSA)
        TC = std::make_unique<toolchains::HIPAMDToolChain>(
            *this, Target, HostTC, Args, TargetDeviceOffloadKind);
      else if (Target.getArch() == llvm::Triple::spirv64 &&
               Target.getVendor() == llvm::Triple::UnknownVendor &&
               Target.getOS() == llvm::Triple::UnknownOS)
        TC = std::make_unique<toolchains::HIPSPVToolChain>(*this, Target,
                                                           HostTC, Args);
      break;
    }
    case Action::OFK_OpenMP:
      // omp + nvptx
      TC = std::make_unique<toolchains::CudaToolChain>(
          *this, Target, HostTC, Args, TargetDeviceOffloadKind);
      break;
    case Action::OFK_SYCL:
      switch (Target.getArch()) {
      case llvm::Triple::spir:
      case llvm::Triple::spir64:
        TC = std::make_unique<toolchains::SYCLToolChain>(*this, Target, HostTC,
                                                         Args);
        break;
      case llvm::Triple::nvptx:
      case llvm::Triple::nvptx64:
        TC = std::make_unique<toolchains::CudaToolChain>(
            *this, Target, HostTC, Args, TargetDeviceOffloadKind);
        break;
      case llvm::Triple::amdgcn:
        TC = std::make_unique<toolchains::HIPAMDToolChain>(
            *this, Target, HostTC, Args, TargetDeviceOffloadKind);
        break;
      default:
        if (isSYCLNativeCPU(Args)) {
          TC = std::make_unique<toolchains::SYCLToolChain>(*this, Target,
                                                           HostTC, Args);
        }
        break;
      }
      break;
    default:
      break;
    }
  }

  return *TC;
}

bool Driver::ShouldUseClangCompiler(const JobAction &JA) const {
  // Say "no" if there is not exactly one input of a type clang understands.
  if (JA.size() != 1 ||
      !types::isAcceptedByClang((*JA.input_begin())->getType()))
    return false;

  // And say "no" if this is not a kind of action clang understands.
  if (!isa<PreprocessJobAction>(JA) && !isa<PrecompileJobAction>(JA) &&
      !isa<CompileJobAction>(JA) && !isa<BackendJobAction>(JA) &&
      !isa<ExtractAPIJobAction>(JA))
    return false;

  return true;
}

bool Driver::ShouldUseFlangCompiler(const JobAction &JA) const {
  // Say "no" if there is not exactly one input of a type flang understands.
  if (JA.size() != 1 ||
      !types::isAcceptedByFlang((*JA.input_begin())->getType()))
    return false;

  // And say "no" if this is not a kind of action flang understands.
  if (!isa<PreprocessJobAction>(JA) && !isa<CompileJobAction>(JA) &&
      !isa<BackendJobAction>(JA))
    return false;

  return true;
}

bool Driver::ShouldEmitStaticLibrary(const ArgList &Args) const {
  // Only emit static library if the flag is set explicitly.
  if (Args.hasArg(options::OPT_emit_static_lib))
    return true;
  return false;
}

/// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and return the
/// grouped values as integers. Numbers which are not provided are set to 0.
///
/// \return True if the entire string was parsed (9.2), or all groups were
/// parsed (10.3.5extrastuff).
bool Driver::GetReleaseVersion(StringRef Str, unsigned &Major, unsigned &Minor,
                               unsigned &Micro, bool &HadExtra) {
  HadExtra = false;

  Major = Minor = Micro = 0;
  if (Str.empty())
    return false;

  if (Str.consumeInteger(10, Major))
    return false;
  if (Str.empty())
    return true;
  if (!Str.consume_front("."))
    return false;

  if (Str.consumeInteger(10, Minor))
    return false;
  if (Str.empty())
    return true;
  if (!Str.consume_front("."))
    return false;

  if (Str.consumeInteger(10, Micro))
    return false;
  if (!Str.empty())
    HadExtra = true;
  return true;
}

/// Parse digits from a string \p Str and fulfill \p Digits with
/// the parsed numbers. This method assumes that the max number of
/// digits to look for is equal to Digits.size().
///
/// \return True if the entire string was parsed and there are
/// no extra characters remaining at the end.
bool Driver::GetReleaseVersion(StringRef Str,
                               MutableArrayRef<unsigned> Digits) {
  if (Str.empty())
    return false;

  unsigned CurDigit = 0;
  while (CurDigit < Digits.size()) {
    unsigned Digit;
    if (Str.consumeInteger(10, Digit))
      return false;
    Digits[CurDigit] = Digit;
    if (Str.empty())
      return true;
    if (!Str.consume_front("."))
      return false;
    CurDigit++;
  }

  // More digits than requested, bail out...
  return false;
}

llvm::opt::Visibility
Driver::getOptionVisibilityMask(bool UseDriverMode) const {
  if (!UseDriverMode)
    return llvm::opt::Visibility(options::ClangOption);
  if (IsCLMode())
    return llvm::opt::Visibility(options::CLOption);
  if (IsDXCMode())
    return llvm::opt::Visibility(options::DXCOption);
  if (IsFlangMode())  {
    return llvm::opt::Visibility(options::FlangOption);
  }
  return llvm::opt::Visibility(options::ClangOption);
}

const char *Driver::getExecutableForDriverMode(DriverMode Mode) {
  switch (Mode) {
  case GCCMode:
    return "clang";
  case GXXMode:
    return "clang++";
  case CPPMode:
    return "clang-cpp";
  case CLMode:
    return "clang-cl";
  case FlangMode:
    return "flang";
  case DXCMode:
    return "clang-dxc";
  }

  llvm_unreachable("Unhandled Mode");
}

bool clang::driver::isOptimizationLevelFast(const ArgList &Args) {
  return Args.hasFlag(options::OPT_Ofast, options::OPT_O_Group, false);
}

bool clang::driver::isObjectFile(std::string FileName) {
  if (llvm::sys::fs::is_directory(FileName))
    return false;
  if (!llvm::sys::path::has_extension(FileName))
    // Any file with no extension should be considered an Object. Take into
    // account -lsomelib library filenames.
    return FileName.rfind("-l", 0) != 0;
  std::string Ext(llvm::sys::path::extension(FileName).drop_front());
  // We cannot rely on lookupTypeForExtension solely as that has 'lib'
  // marked as an object.
  return (Ext != "lib" &&
          types::lookupTypeForExtension(Ext) == types::TY_Object);
}

bool clang::driver::isStaticArchiveFile(const StringRef &FileName) {
  if (!llvm::sys::path::has_extension(FileName))
    // Any file with no extension should not be considered an Archive.
    return false;
  llvm::file_magic Magic;
  llvm::identify_magic(FileName, Magic);
  // Only .lib and archive files are to be considered.
  return (Magic == llvm::file_magic::archive);
}

bool clang::driver::willEmitRemarks(const ArgList &Args) {
  // -fsave-optimization-record enables it.
  if (Args.hasFlag(options::OPT_fsave_optimization_record,
                   options::OPT_fno_save_optimization_record, false))
    return true;

  // -fsave-optimization-record=<format> enables it as well.
  if (Args.hasFlag(options::OPT_fsave_optimization_record_EQ,
                   options::OPT_fno_save_optimization_record, false))
    return true;

  // -foptimization-record-file alone enables it too.
  if (Args.hasFlag(options::OPT_foptimization_record_file_EQ,
                   options::OPT_fno_save_optimization_record, false))
    return true;

  // -foptimization-record-passes alone enables it too.
  if (Args.hasFlag(options::OPT_foptimization_record_passes_EQ,
                   options::OPT_fno_save_optimization_record, false))
    return true;
  return false;
}

llvm::StringRef clang::driver::getDriverMode(StringRef ProgName,
                                             ArrayRef<const char *> Args) {
  static StringRef OptName =
      getDriverOptTable().getOption(options::OPT_driver_mode).getPrefixedName();
  llvm::StringRef Opt;
  for (StringRef Arg : Args) {
    if (!Arg.starts_with(OptName))
      continue;
    Opt = Arg;
  }
  if (Opt.empty())
    Opt = ToolChain::getTargetAndModeFromProgramName(ProgName).DriverMode;
  return Opt.consume_front(OptName) ? Opt : "";
}

bool driver::IsClangCL(StringRef DriverMode) { return DriverMode.equals("cl"); }

llvm::Error driver::expandResponseFiles(SmallVectorImpl<const char *> &Args,
                                        bool ClangCLMode,
                                        llvm::BumpPtrAllocator &Alloc,
                                        llvm::vfs::FileSystem *FS) {
  // Parse response files using the GNU syntax, unless we're in CL mode. There
  // are two ways to put clang in CL compatibility mode: ProgName is either
  // clang-cl or cl, or --driver-mode=cl is on the command line. The normal
  // command line parsing can't happen until after response file parsing, so we
  // have to manually search for a --driver-mode=cl argument the hard way.
  // Finally, our -cc1 tools don't care which tokenization mode we use because
  // response files written by clang will tokenize the same way in either mode.
  enum { Default, POSIX, Windows } RSPQuoting = Default;
  for (const char *F : Args) {
    if (strcmp(F, "--rsp-quoting=posix") == 0)
      RSPQuoting = POSIX;
    else if (strcmp(F, "--rsp-quoting=windows") == 0)
      RSPQuoting = Windows;
  }

  // Determines whether we want nullptr markers in Args to indicate response
  // files end-of-lines. We only use this for the /LINK driver argument with
  // clang-cl.exe on Windows.
  bool MarkEOLs = ClangCLMode;

  llvm::cl::TokenizerCallback Tokenizer;
  if (RSPQuoting == Windows || (RSPQuoting == Default && ClangCLMode))
    Tokenizer = &llvm::cl::TokenizeWindowsCommandLine;
  else
    Tokenizer = &llvm::cl::TokenizeGNUCommandLine;

  if (MarkEOLs && Args.size() > 1 && StringRef(Args[1]).starts_with("-cc1"))
    MarkEOLs = false;

  llvm::cl::ExpansionContext ECtx(Alloc, Tokenizer);
  ECtx.setMarkEOLs(MarkEOLs);
  if (FS)
    ECtx.setVFS(FS);

  if (llvm::Error Err = ECtx.expandResponseFiles(Args))
    return Err;

  // If -cc1 came from a response file, remove the EOL sentinels.
  auto FirstArg = llvm::find_if(llvm::drop_begin(Args),
                                [](const char *A) { return A != nullptr; });
  if (FirstArg != Args.end() && StringRef(*FirstArg).starts_with("-cc1")) {
    // If -cc1 came from a response file, remove the EOL sentinels.
    if (MarkEOLs) {
      auto newEnd = std::remove(Args.begin(), Args.end(), nullptr);
      Args.resize(newEnd - Args.begin());
    }
  }

  return llvm::Error::success();
}

void Driver::populateSYCLDeviceTraitsMacrosArgs(
    const llvm::opt::ArgList &Args,
    const llvm::SmallVector<llvm::Triple, 4> &UniqueSYCLTriplesVec) {
  const auto &TargetTable = DeviceConfigFile::TargetTable;
  std::map<StringRef, unsigned int> AllDevicesHave;
  std::map<StringRef, bool> AnyDeviceHas;
  bool AnyDeviceHasAnyAspect = false;
  unsigned int ValidTargets = 0;
  for (const auto &TargetTriple : UniqueSYCLTriplesVec) {
    // Try and find the whole triple, if there's no match, remove parts of the
    // triple from the end to find partial matches.
    auto TargetTripleStr = TargetTriple.getTriple();
    bool Found = false;
    bool EmptyTriple = false;
    auto TripleIt = TargetTable.end();
    while (!Found && !EmptyTriple) {
      TripleIt = TargetTable.find(TargetTripleStr);
      Found = (TripleIt != TargetTable.end());
      if (!Found) {
        auto Pos = TargetTripleStr.find_last_of('-');
        EmptyTriple = (Pos == std::string::npos);
        TargetTripleStr =
            EmptyTriple ? TargetTripleStr : TargetTripleStr.substr(0, Pos);
      }
    }
    if (Found) {
      assert(TripleIt != TargetTable.end());
      const auto &TargetInfo = (*TripleIt).second;
      ++ValidTargets;
      const auto &AspectList = TargetInfo.aspects;
      const auto &MaySupportOtherAspects = TargetInfo.maySupportOtherAspects;
      if (!AnyDeviceHasAnyAspect)
        AnyDeviceHasAnyAspect = MaySupportOtherAspects;
      for (const auto &aspect : AspectList) {
        // If target has an entry in the config file, the set of aspects
        // supported by all devices supporting the target is 'AspectList'. If
        // there's no entry, such set is empty.
        const auto &AspectIt = AllDevicesHave.find(aspect);
        if (AspectIt != AllDevicesHave.end())
          ++AllDevicesHave[aspect];
        else
          AllDevicesHave[aspect] = 1;
        // If target has an entry in the config file AND
        // 'MaySupportOtherAspects' is false, the set of aspects supported by
        // any device supporting the target is 'AspectList'. If there's no
        // entry OR 'MaySupportOtherAspects' is true, such set contains all
        // the aspects.
        AnyDeviceHas[aspect] = true;
      }
    }
  }

  if (ValidTargets == 0) {
    // If there's no entry for the target in the device config file, the set
    // of aspects supported by any device supporting the target contains all
    // the aspects.
    AnyDeviceHasAnyAspect = true;
  }

  if (AnyDeviceHasAnyAspect) {
    // There exists some target that supports any given aspect.
    SmallString<64> MacroAnyDeviceAnyAspect(
        "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1");
    SYCLDeviceTraitsMacrosArgs.push_back(
        Args.MakeArgString(MacroAnyDeviceAnyAspect));
  } else {
    // Some of the aspects are not supported at all by any of the targets.
    // Thus, we need to define individual macros for each supported aspect.
    for (const auto &[TargetKey, SupportedTarget] : AnyDeviceHas) {
      assert(SupportedTarget);
      SmallString<64> MacroAnyDevice("-D__SYCL_ANY_DEVICE_HAS_");
      MacroAnyDevice += TargetKey;
      MacroAnyDevice += "__=1";
      SYCLDeviceTraitsMacrosArgs.push_back(Args.MakeArgString(MacroAnyDevice));
    }
  }
  for (const auto &[TargetKey, SupportedTargets] : AllDevicesHave) {
    if (SupportedTargets != ValidTargets)
      continue;
    SmallString<64> MacroAllDevices("-D__SYCL_ALL_DEVICES_HAVE_");
    MacroAllDevices += TargetKey;
    MacroAllDevices += "__=1";
    SYCLDeviceTraitsMacrosArgs.push_back(Args.MakeArgString(MacroAllDevices));
  }
}
