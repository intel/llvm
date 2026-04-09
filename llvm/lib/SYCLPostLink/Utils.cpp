//===------------ Utils.cpp - Utility functions for SYCL Offloading -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Low-level utility functions for SYCL post-link processing.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLPostLink/Utils.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/DeviceConfigFile.hpp"
#include "llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using PropSetRegTy = llvm::util::PropertySetRegistry;

namespace {

PropSetRegTy
computeModulePropertiesHelper(const module_split::ModuleDesc &MD,
                              const sycl::GlobalBinImageProps &GlobProps,
                              bool AllowDeviceImageDependencies,
                              module_split::IRSplitMode SplitMode) {
  PropSetRegTy PropSet;
  // For bf16 devicelib module, no kernel included and no specialization
  // constant used, skip regular Prop emit. However, we have fallback and
  // native version of bf16 devicelib and we need new property values to
  // indicate all exported function.
  if (!MD.isSYCLDeviceLib())
    PropSet = sycl::computeModuleProperties(
        MD.getModule(), MD.entries(), GlobProps, AllowDeviceImageDependencies);
  else
    PropSet = sycl::computeDeviceLibProperties(MD.getModule(), MD.Name);

  // When the split mode is none, the required work group size will be added
  // to the whole module, which will make the runtime unable to
  // launch the other kernels in the module that have different
  // required work group sizes or no required work group sizes. So we need to
  // remove the required work group size metadata in this case.
  if (SplitMode == module_split::SPLIT_NONE)
    PropSet.remove(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS,
                   PropSetRegTy::PROPERTY_REQD_WORK_GROUP_SIZE);
  return PropSet;
}

} // end anonymous namespace

Error llvm::sycl_post_link::saveModuleIR(Module &M, const StringRef Filename,
                                         bool OutputAssembly) {
  std::error_code EC;
  raw_fd_ostream Out{Filename, EC, sys::fs::OF_None};
  if (EC)
    return createStringError(EC, "error opening the file '" + Filename + "'");

  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  if (OutputAssembly)
    MPM.addPass(PrintModulePass(Out));
  else
    MPM.addPass(BitcodeWriterPass(Out));
  MPM.run(M, MAM);
  return Error::success();
}

bool llvm::sycl_post_link::isTargetCompatibleWithModule(
    const std::string &Target, module_split::ModuleDesc &IrMD) {
  // When the user does not specify a target,
  // (e.g. -o out.table compared to -o intel_gpu_pvc,out-pvc.table)
  // Target will be empty and we will not want to perform any filtering, so
  // we return true here.
  if (Target.empty())
    return true;

  // TODO: If a target not found in the device config file is passed,
  // to SYCLPostLink, then we should probably throw an error. However,
  // since not all the information for all the targets is filled out
  // right now, we return true, having the affect that unrecognized
  // targets have no filtering applied to them.
  if (!is_contained(DeviceConfigFile::TargetTable, Target))
    return true;

  const DeviceConfigFile::TargetInfo &TargetInfo =
      DeviceConfigFile::TargetTable[Target];
  const SYCLDeviceRequirements &ModuleReqs =
      IrMD.getOrComputeDeviceRequirements();

  // Check to see if all the requirements of the input module
  // are compatbile with the target.
  for (const auto &Aspect : ModuleReqs.Aspects) {
    if (!is_contained(TargetInfo.aspects, Aspect.Name))
      return false;
  }

  // Check if module sub group size is compatible with the target.
  // For ESIMD, the reqd_sub_group_size will be 1; this is not
  // a supported by any backend (e.g. no backend can support a kernel
  // with sycl::reqd_sub_group_size(1)), but for ESIMD, this is
  // a special case.
  if (!IrMD.isESIMD() && ModuleReqs.SubGroupSize.has_value() &&
      !is_contained(TargetInfo.subGroupSizes, *ModuleReqs.SubGroupSize))
    return false;

  return true;
}

Error llvm::sycl_post_link::saveModuleProperties(
    const module_split::ModuleDesc &MD,
    const sycl::GlobalBinImageProps &GlobProps, StringRef Filename,
    StringRef Target, bool AllowDeviceImageDependencies,
    module_split::IRSplitMode SplitMode) {
  PropSetRegTy PropSet = computeModulePropertiesHelper(
      MD, GlobProps, AllowDeviceImageDependencies, SplitMode);

  if (!Target.empty())
    PropSet.add(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS, "compile_target",
                Target);

  return writeToOutput(Filename, [&](raw_ostream &OS) -> Error {
    PropSet.write(OS);
    return Error::success();
  });
}

Error llvm::sycl_post_link::saveModuleSymbolTable(
    const module_split::ModuleDesc &MD, StringRef Filename) {
  std::string SymT =
      sycl::computeModuleSymbolTable(MD.getModule(), MD.entries());
  return writeToOutput(Filename, [&](raw_ostream &OS) -> Error {
    OS << SymT;
    return Error::success();
  });
}
