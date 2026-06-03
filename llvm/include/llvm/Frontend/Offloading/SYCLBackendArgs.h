//===- SYCLBackendArgs.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Driver-independent translation of SYCL backend / linker target arguments
// (e.g. -Xsycl-target-backend, -Xs, -ftarget-register-alloc-mode=, the
// implied flags driven by -g/-O0, and the intel_gpu_* arch resolution) into
// flag strings consumable by ocloc / opencl-aot. This logic was previously
// inlined in clang/lib/Driver/ToolChains/SYCL.cpp; relocating it here lets
// both the Clang Driver and clang-sycl-linker call it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OFFLOADING_SYCLBACKENDARGS_H
#define LLVM_FRONTEND_OFFLOADING_SYCLBACKENDARGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

#include <string>

namespace llvm::offloading::sycl {

/// Numeric option-IDs from the caller's OptTable. Values are the integer IDs
/// from clang/Driver/Options.inc (Driver-side) or the equivalent IDs in
/// clang-sycl-linker's option table (tool-side). The library doesn't include
/// either header to stay layer-independent.
struct BackendOptIds {
  unsigned g_Group;
  unsigned g0;
  unsigned _SLASH_Z7;
  unsigned O_Group;
  unsigned O0;
  unsigned Xs;
  unsigned Xs_separate;
  unsigned Xsycl_backend;
  unsigned Xsycl_backend_EQ;
  unsigned Xsycl_linker;
  unsigned Xsycl_linker_EQ;
  unsigned ftarget_register_alloc_mode_EQ;
  unsigned fsycl_fp64_conv_emu;
  unsigned ftarget_compile_fast;
  unsigned ftarget_export_symbols;
  unsigned fno_target_export_symbols;
  unsigned foffload_fp32_prec_div;
  unsigned foffload_fp32_prec_sqrt;
  unsigned offload_targets_EQ;
};

/// Inputs needed by the translators. Carried as a struct so the Driver-side
/// and tool-side adapters can populate them without sharing types.
struct BackendArgsInput {
  const opt::ArgList &Args;
  const llvm::Triple &Triple;
  /// Resolved Intel GPU device name (may be empty).
  StringRef Device;
  /// Whether the run is operating on the implied SPIR/SPIRV default triple.
  bool IsSYCLDefaultTripleImplied;
  /// Whether the host toolchain targets MSVC environment.
  bool HostIsWindowsMSVCEnv;
  /// JobAction's offloading arch (e.g. "pvc"); empty if not applicable.
  /// Used by addSPIRVImpliedTargetArgs to derive `-device <arch>` for
  /// intel_gpu_* targets.
  StringRef JobOffloadingArch;
  BackendOptIds Ids;

  /// Resolves a user-level `-Xsycl-target-backend=<value>` triple-or-arch
  /// alias into a normalized triple. Provided by the caller because the
  /// Driver has its own `Driver::getSYCLDeviceTriple()` with diagnostics.
  /// Tool-side adapters can wrap a free function with the same shape.
  function_ref<llvm::Triple(StringRef Value, const opt::Arg *A)>
      ResolveDeviceTriple;

  /// Emit `warn_drv_ftarget_register_alloc_mode_pvc`-style warning.
  /// Args: <flag>, <mode>.
  function_ref<void(StringRef Flag, StringRef Mode)> WarnPVCDeprecatedGRFFn;
  /// Emit `err_drv_unsupported_option_argument`-style error.
  /// Args: <option-spelling>, <bad-value>.
  function_ref<void(StringRef OptionSpelling, StringRef BadValue)>
      ErrUnsupportedOptionArgumentFn;
  /// Emit `err_drv_Xsycl_target_missing_triple`-style error.
  /// Args: <option-spelling>.
  function_ref<void(StringRef OptionSpelling)> ErrXsyclTargetMissingTripleFn;
  /// Emit `err_drv_unsupported_opt_for_target`-style error.
  /// Args: <option>, <target>.
  function_ref<void(StringRef Option, StringRef Target)>
      ErrUnsupportedOptForTargetFn;
};

/// Pure helpers (no Driver dependencies). Same algorithm as the in-Driver
/// SYCL::gen::resolveGenDevice / getGenGRFFlag.
LLVM_ABI StringRef resolveGenDevice(StringRef DeviceName);
LLVM_ABI StringRef getGenGRFFlag(StringRef GRFMode);

/// Tokenize a GNU-quoted string and append its tokens to CmdArgs, copied
/// through Args's string saver.
LLVM_ABI void parseTargetOpts(StringRef ArgString, const opt::ArgList &Args,
                              opt::ArgStringList &CmdArgs);

/// Implements the body of `SYCLToolChain::AddSPIRVImpliedTargetArgs`.
/// Synthesizes ocloc / opencl-aot flags from -g, -O0,
/// -ftarget-register-alloc-mode=, -fsycl-fp64-conv-emu,
/// -ftarget-compile-fast, -ftarget-export-symbols,
/// -foffload-fp32-prec-{div,sqrt}, plus intel_gpu_* arch -> -device <name>
/// resolution.
LLVM_ABI void addSPIRVImpliedTargetArgs(const BackendArgsInput &In,
                                        opt::ArgStringList &CmdArgs);

/// Implements the body of `SYCLToolChain::TranslateBackendTargetArgs`.
LLVM_ABI void translateBackendTargetArgs(const BackendArgsInput &In,
                                         opt::ArgStringList &CmdArgs);

/// Implements the body of `SYCLToolChain::TranslateLinkerTargetArgs`.
LLVM_ABI void translateLinkerTargetArgs(const BackendArgsInput &In,
                                        opt::ArgStringList &CmdArgs);

/// Implements `SYCLToolChain::TranslateTargetOpt` for an arbitrary
/// (Opt, Opt_EQ) pair. Used by both translateBackendTargetArgs and
/// translateLinkerTargetArgs and exposed for direct callers in the Driver
/// (e.g. -Xdevice-post-link).
LLVM_ABI void translateTargetOpt(const BackendArgsInput &In,
                                 opt::OptSpecifier Opt,
                                 opt::OptSpecifier Opt_EQ,
                                 opt::ArgStringList &CmdArgs);

/// Implements `SYCLToolChain::TranslateGPUTargetOpt`.
LLVM_ABI void translateGPUTargetOpt(const BackendArgsInput &In,
                                    opt::OptSpecifier Opt_EQ,
                                    opt::ArgStringList &CmdArgs);

/// Tests whether any -device value in CmdArgs is a PVC device.
/// On match, writes the matched device argument to DevArg.
LLVM_ABI bool hasPVCDevice(ArrayRef<const char *> CmdArgs, std::string &DevArg);

} // namespace llvm::offloading::sycl

#endif // LLVM_FRONTEND_OFFLOADING_SYCLBACKENDARGS_H
