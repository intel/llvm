//===- DeviceCompilation.h - Compile SYCL device code with libtooling -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "RTC.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Module.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>

namespace jit_compiler {

using ModuleUPtr = std::unique_ptr<llvm::Module>;

llvm::Expected<std::string>
calculateHash(InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
              const llvm::opt::InputArgList &UserArgList);

llvm::Expected<ModuleUPtr>
compileDeviceCode(InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
                  const llvm::opt::InputArgList &UserArgList,
                  std::string &BuildLog, llvm::LLVMContext &Context);

llvm::Error linkDeviceLibraries(llvm::Module &Module,
                                const llvm::opt::InputArgList &UserArgList,
                                std::string &BuildLog);

using PostLinkResult = std::pair<RTCBundleInfo, llvm::SmallVector<ModuleUPtr>>;
llvm::Expected<PostLinkResult>
performPostLink(ModuleUPtr Module, const llvm::opt::InputArgList &UserArgList);

llvm::Expected<llvm::opt::InputArgList>
parseUserArgs(View<const char *> UserArgs);

void encodeBuildOptions(RTCBundleInfo &BundleInfo,
                        const llvm::opt::InputArgList &UserArgList);

void configureDiagnostics(llvm::LLVMContext &Context, std::string &BuildLog);

} // namespace jit_compiler
