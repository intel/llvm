//=- WebAssemblySelectionDAGInfo.h - WebAssembly SelectionDAG Info -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the WebAssembly subclass for
/// SelectionDAGTargetInfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

namespace llvm {

class WebAssemblySelectionDAGInfo final : public SelectionDAGTargetInfo {
public:
  ~WebAssemblySelectionDAGInfo() override;
};

} // end namespace llvm

#endif
