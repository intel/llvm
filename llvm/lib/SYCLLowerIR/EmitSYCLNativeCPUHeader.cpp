//===---- EmitSYCLHCHeader.cpp - Emit SYCL Native CPU Helper Header Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits the SYCL Native CPU helper headers, containing the kernel definition
// and handlers.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/EmitSYCLNativeCPUHeader.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>
#include <numeric>

using namespace llvm;

namespace {
SmallVector<bool> getArgMask(const Function *F) {
  SmallVector<bool> Res;
  auto *UsedNode = F->getMetadata("sycl_kernel_omit_args");
  if (!UsedNode) {
    // the metadata node is not available if -fenable-sycl-dae
    // was not set; set everything to true in the mask.
    for (unsigned I = 0; I < F->getFunctionType()->getNumParams(); I++) {
      Res.push_back(true);
    }
    return Res;
  }
  auto NumOperands = UsedNode->getNumOperands();
  for (unsigned I = 0; I < NumOperands; I++) {
    auto &Op = UsedNode->getOperand(I);
    if (auto *CAM = dyn_cast<ConstantAsMetadata>(Op.get())) {
      if (auto *Const = dyn_cast<ConstantInt>(CAM->getValue())) {
        auto Val = Const->getValue();
        Res.push_back(!Val.getBoolValue());
      } else {
        report_fatal_error("Unable to retrieve constant int from "
                           "sycl_kernel_omit_args metadata node");
      }
    } else {
      report_fatal_error(
          "Error while processing sycl_kernel_omit_args metadata node");
    }
  }
  return Res;
}

SmallVector<StringRef> getArgTypeNames(const Function *F) {
  SmallVector<StringRef> Res;
  auto *TNNode = F->getMetadata("kernel_arg_type");
  assert(TNNode &&
         "kernel_arg_type metadata node is required for sycl native CPU");
  auto NumOperands = TNNode->getNumOperands();
  for (unsigned I = 0; I < NumOperands; I++) {
    auto &Op = TNNode->getOperand(I);
    auto *MDS = dyn_cast<MDString>(Op.get());
    if (!MDS)
      report_fatal_error("error while processing kernel_arg_types metadata");
    Res.push_back(MDS->getString());
  }
  return Res;
}

void emitKernelDecl(const Function *F, const SmallVector<bool> &ArgMask,
                    const SmallVector<StringRef> &ArgTypeNames,
                    raw_ostream &O) {
  auto EmitArgDecl = [&](const Argument *Arg, unsigned Index) {
    Type *ArgTy = Arg->getType();
    if (isa<PointerType>(ArgTy))
      return "void *";
    return ArgTypeNames[Index].data();
  };

  auto NumParams = F->getFunctionType()->getNumParams();
  O << "extern \"C\" void " << F->getName() << "(";

  unsigned I = 0, UsedI = 0;
  for (; I + 1 < ArgMask.size() && UsedI + 1 < NumParams; I++) {
    if (!ArgMask[I])
      continue;
    O << EmitArgDecl(F->getArg(UsedI), I) << ", ";
    UsedI++;
  }

  // parameters may have been removed.
  bool NoUsedArgs = true;
  for (auto &Entry : ArgMask) {
    NoUsedArgs &= !Entry;
  }
  if (NoUsedArgs) {
    O << ");\n";
    return;
  }
  // find the index of the last used arg
  while (!ArgMask[I] && I + 1 < ArgMask.size())
    I++;
  O << EmitArgDecl(F->getArg(UsedI), I) << ", __nativecpu_state *);\n";
}

void emitSubKernelHandler(const Function *F, const SmallVector<bool> &ArgMask,
                          const SmallVector<StringRef> &ArgTypeNames,
                          raw_ostream &O) {
  SmallVector<unsigned> UsedArgIdx;
  auto EmitParamCast = [&](Argument *Arg, unsigned Index) {
    std::string Res;
    llvm::raw_string_ostream OS(Res);
    UsedArgIdx.push_back(Index);
    if (isa<PointerType>(Arg->getType())) {
      OS << "  void* arg" << Index << " = ";
      OS << "MArgs[" << Index << "].getPtr();\n";
      return OS.str();
    }
    auto TN = ArgTypeNames[Index].str();
    OS << "  " << TN << " arg" << Index << " = ";
    OS << "*(" << TN << "*)"
       << "MArgs[" << Index << "].getPtr();\n";
    return OS.str();
  };

  O << "\ninline static void " << F->getName() << "subhandler(";
  O << "const sycl::detail::NativeCPUArgDesc *MArgs, "
       "__nativecpu_state *state) {\n";
  // Retrieve only the args that are used
  for (unsigned I = 0, UsedI = 0;
       I < ArgMask.size() && UsedI < F->getFunctionType()->getNumParams();
       I++) {
    if (ArgMask[I]) {
      O << EmitParamCast(F->getArg(UsedI), I);
      UsedI++;
    }
  }
  // Emit the actual kernel call
  O << "  " << F->getName() << "(";
  if (UsedArgIdx.size() == 0) {
    O << ");\n";
  } else {
    for (unsigned I = 0; I < UsedArgIdx.size() - 1; I++) {
      O << "arg" << UsedArgIdx[I] << ", ";
    }
    if (UsedArgIdx.size() >= 1)
      O << "arg" << UsedArgIdx.back();
    O << ", state);\n";
  }
  O << "};\n\n";
}

// Todo: maybe we could use clang-offload-wrapper for this,
// the main thing that prevents use from using clang-offload-wrapper
// right now is the fact that we need the subhandler
// to figure out which args are used or not, and so the BinaryStart entry
// need to point to the subhandler, and I'm not sure how to do that in
// clang-offload-wrapper. If we figure out a better way to deal with unused
// kernel args, we can probably get rid of the subhandler and make BinaryStart
// point the the actual kernel function pointer, which should be doable in
// clang-offload-wrapper.
void emitSYCLRegisterLib(const Function *F, raw_ostream &O) {
  auto KernelName = F->getName();
  std::string SubHandlerName = (KernelName + "subhandler").str();
  static const char *BinariesT = "pi_device_binaries_struct";
  static const char *BinaryT = "pi_device_binary_struct";
  static const char *OffloadEntryT = "_pi_offload_entry_struct";
  std::string Binaries = (BinariesT + KernelName).str();
  std::string Binary = (BinaryT + KernelName).str();
  std::string OffloadEntry = (OffloadEntryT + KernelName).str();
  // Fill in the offload entry struct for this kernel
  O << "static " << OffloadEntryT << " " << OffloadEntry << "{"
    << "(void*)&" << SubHandlerName << ", "            // addr
    << "const_cast<char*>(\"" << KernelName << "\"), " // name
    << "1, "                                           // size
    << "0, "                                           // flags
    << "0 "                                            // reserved
    << "};\n";
  // Fill in the binary struct
  O << "static " << BinaryT << " " << Binary << "{"
    << "0, "                                             // Version
    << "4, "                                             // Kind
    << "0, "                                             // Format
    << "__SYCL_PI_DEVICE_BINARY_TARGET_UNKNOWN, "        // Device target spec
    << "nullptr, "                                       // Compile options
    << "nullptr, "                                       // Link options
    << "nullptr, "                                       // Manifest start
    << "nullptr, "                                       // Manifest end
    << "(unsigned char*)&" << SubHandlerName << ", "     // BinaryStart
    << "(unsigned char*)&" << SubHandlerName << " + 1, " // BinaryEnd
    << "&" << OffloadEntry << ", "                       // EntriesBegin
    << "&" << OffloadEntry << "+1, "                     // EntriesEnd
    << "nullptr, "                                       // PropertySetsBegin
    << "nullptr "                                        // PropertySetsEnd
    << "};\n";
  // Fill in the binaries struct
  O << "static " << BinariesT << " " << Binaries << "{"
    << "0, "                 // Version
    << "1, "                 // NumDeviceBinaries
    << "&" << Binary << ", " // DeviceBinaries
    << "nullptr, "           // HostEntriesBegin
    << "nullptr "            // HostEntriesEnd
    << "};\n";

  // Define a struct and use its constructor to call __sycl_register_lib
  std::string InitNativeCPU = ("init_native_cpu" + KernelName).str();
  std::string InitNativeCPUT = InitNativeCPU + "_t";
  O << "struct " << InitNativeCPUT << "{\n"
    << "\t" << InitNativeCPUT << "(){\n"
    << "\t\t"
    << "__sycl_register_lib(&" << Binaries << ");\n"
    << "\t}\n"
    << "};\n"
    << "static " << InitNativeCPUT << " " << InitNativeCPU << ";\n";
}

} // namespace

PreservedAnalyses EmitSYCLNativeCPUHeaderPass::run(Module &M,
                                                   ModuleAnalysisManager &MAM) {
  SmallVector<Function *> Kernels;
  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
      Kernels.push_back(&F);
  }

  // Emit native CPU helper header
  if (NativeCPUHeaderName == "") {
    report_fatal_error("No file name for Native CPU helper header specified",
                       false);
  }
  int HCHeaderFD = 0;
  std::error_code EC =
      llvm::sys::fs::openFileForWrite(NativeCPUHeaderName, HCHeaderFD);
  if (EC) {
    report_fatal_error(StringRef(EC.message()), false);
  }
  llvm::raw_fd_ostream O(HCHeaderFD, true);
  O << "#pragma once\n";
  O << "#include <sycl/detail/native_cpu.hpp>\n";
  O << "#include <sycl/detail/pi.h>\n";
  O << "extern \"C\" void __sycl_register_lib(pi_device_binaries desc);\n";

  for (auto *F : Kernels) {
    auto ArgMask = getArgMask(F);
    auto ArgTypeNames = getArgTypeNames(F);
    emitKernelDecl(F, ArgMask, ArgTypeNames, O);
    emitSubKernelHandler(F, ArgMask, ArgTypeNames, O);
    emitSYCLRegisterLib(F, O);
  }

  return PreservedAnalyses::all();
}
