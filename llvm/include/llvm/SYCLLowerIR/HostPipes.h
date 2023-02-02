//===------- HostPipes.h - get required into about SYCL Host Pipes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file contains a number of functions to extract corresponding attributes
// of host pipe variables and save them as a property set for the runtime.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/MapVector.h"

#include <cstdint>

namespace llvm {

class GlobalVariable;

/// Return \c true if the variable @GV is a host pipe variable.
///
/// @param GV [in] A variable to test.
///
/// @return \c true if the variable is a host pipe variable, \c false
/// otherwise.
bool isHostPipeVariable(const GlobalVariable &GV);

} // end namespace llvm
