//==---------------- windows_pi.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include <windows.h>
#include <winreg.h>

namespace pi {

void *loadOsLibrary(const std::string &PluginPath) {
  return (void *)LoadLibraryA(PluginPath.c_str());
}

void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName) {
  return GetProcAddress((HMODULE)Library, FunctionName.c_str());
}

} // namespace pi
