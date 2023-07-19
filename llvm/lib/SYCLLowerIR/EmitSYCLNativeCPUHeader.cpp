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

void emitSubKernelHandler(const Function *F, raw_ostream &O) {
  O << "\nextern \"C\" void " << F->getName() << "subhandler(";
  O << "const sycl::detail::NativeCPUArgDesc *MArgs, "
       "__nativecpu_state *state);\n";
  return;
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
    emitSubKernelHandler(F, O);
    emitSYCLRegisterLib(F, O);
  }

  return PreservedAnalyses::all();
}
