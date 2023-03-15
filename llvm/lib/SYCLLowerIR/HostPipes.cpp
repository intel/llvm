//===----- HostPipes.cpp - SYCL Device Globals Pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/HostPipes.h"
#include "llvm/SYCLLowerIR/CompileTimePropertiesPass.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <cassert>

using namespace llvm;

namespace {

constexpr StringRef SYCL_HOST_PIPE_ATTR = "sycl-host-pipe";
constexpr StringRef SYCL_HOST_PIPE_SIZE_ATTR = "sycl-host-pipe-size";
constexpr StringRef SYCL_UNIQUE_ID_ATTR = "sycl-unique-id";

/// Returns the size (in bytes) of the type \c T of the host
/// pipe variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-host-pipe-size.
///
/// @param GV [in] Host Pipe variable.
///
/// @returns the size (int bytes) of the underlying type \c T of the
/// host pipe variable represented in the LLVM IR by  @GV.
uint32_t getHostPipeTypeSize(const GlobalVariable &GV) {
  assert(GV.hasAttribute(SYCL_HOST_PIPE_SIZE_ATTR) &&
         "The host pipe variable must have the 'sycl-host-pipe-size' "
         "attribute that must contain a number representing the size of the "
         "underlying type T of the host pipe variable");
  return getAttributeAsInteger<uint32_t>(GV, SYCL_HOST_PIPE_SIZE_ATTR);
}

} // anonymous namespace

namespace llvm {

/// Return \c true if the variable @GV is a host pipe variable.
///
/// The function checks whether the variable has the LLVM IR attribute \c
/// sycl-host-pipe.
/// @param GV [in] A variable to test.
///
/// @return \c true if the variable is a host pipe variable, \c false
/// otherwise.
bool isHostPipeVariable(const GlobalVariable &GV) {
  return GV.hasAttribute(SYCL_HOST_PIPE_ATTR);
}

/// Returns the unique id for the host pipe variable.
///
/// The function gets this value from the LLVM IR attribute \c
/// sycl-unique-id.
///
/// @param GV [in] Device Global variable.
///
/// @returns the unique id of the host pipe variable represented
/// in the LLVM IR by \c GV.
StringRef getHostPipeVariableUniqueId(const GlobalVariable &GV) {
  assert(GV.hasAttribute(SYCL_UNIQUE_ID_ATTR) &&
         "a 'sycl-unique-id' string must be associated with every host "
         "pipe variable");
  return GV.getAttribute(SYCL_UNIQUE_ID_ATTR).getValueAsString();
}

HostPipePropertyMapTy collectHostPipeProperties(const Module &M) {
  HostPipePropertyMapTy HPM;
  auto HostPipeNum = count_if(M.globals(), isHostPipeVariable);
  if (HostPipeNum == 0)
    return HPM;

  HPM.reserve(HostPipeNum);

  for (auto &GV : M.globals()) {
    if (!isHostPipeVariable(GV))
      continue;

    HPM[getHostPipeVariableUniqueId(GV)] = {getHostPipeTypeSize(GV)};
  }

  return HPM;
}

} // namespace llvm
