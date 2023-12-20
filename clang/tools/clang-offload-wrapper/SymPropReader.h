//===- SymPropReader.h --r-------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_OFFLOAD_WRAPPER_SYM_PROP_H
#define LLVM_CLANG_TOOLS_CLANG_OFFLOAD_WRAPPER_SYM_PROP_H

#include "llvm/IR/Constant.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

class SymPropReader {
private:
  // Table containing list of names of wrapped BC files
  std::unique_ptr<util::SimpleTable> SymPropTable{nullptr};

  int BCFileIndex{0}; // Index into SymPropTable to select wrapped BC file
  int ImageCnt{0};    // Number of images in selected wrapped BC file
  int ImageIndex{0};  // Index to select image in a wrapped BC file

  // Initializer for all device images.  Size is ImageCnt
  Constant *DeviceImagesInitializer{nullptr};
  // Initializer for the ImageIndex-th image in DeviceImagesInitializer
  Constant *CurrentDeviceImageInitializer{nullptr};

  LLVMContext SymPropsC;
  ExitOnError SymPropsExitOnErr;
  // Module holding loaded wrapped BC file
  std::unique_ptr<Module> CurrentSymPropsM{nullptr};

  // Initializer containing names of all entry points
  const Constant *EntriesInitializer{nullptr};

public:
  SymPropReader(StringRef SymPropBCFiles, StringRef ToolName) {
    Expected<std::unique_ptr<util::SimpleTable>> TPtr =
        util::SimpleTable::read(SymPropBCFiles);

    if (!TPtr) {
      logAllUnhandledErrors(std::move(TPtr.takeError()),
                            WithColor::error(errs(), ToolName));
      exit(1);
    }

    SymPropTable = std::move(*TPtr);
  }

  void getNextDeviceImageInitializer();
  uint64_t getNumEntries();
  StringRef getEntryName(uint64_t i);
  std::unique_ptr<llvm::util::PropertySetRegistry> getPropRegistry();
};
#endif
