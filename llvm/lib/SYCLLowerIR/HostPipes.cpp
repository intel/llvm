//===------------- HostPipes.cpp - SYCL Host Pipes Pass -------------------===//
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <cassert>

using namespace llvm;

namespace {

constexpr StringRef SYCL_HOST_PIPE_ATTR = "sycl-host-pipe";

} // anonymous namespace

namespace llvm {

/// Return \c true if the variable @GV is a device global variable.
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

} // namespace llvm
