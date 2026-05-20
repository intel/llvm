/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <dlfcn.h>
#include <optional>

#include "ur_filesystem_resolved.hpp"
#include "ur_loader.hpp"

namespace fs = filesystem;

namespace ur_loader {

std::optional<fs::path> getLoaderLibPath() {
  Dl_info info;
  if (dladdr((void *)getLoaderLibPath, &info)) {
    auto libPath = fs::path(info.dli_fname);
    if (fs::exists(libPath)) {
      return fs::absolute(libPath).parent_path();
    }
  }

  return std::nullopt;
}

std::optional<fs::path> getAdapterNameAsPath(std::string adapterName) {
  return fs::path(adapterName);
}

} // namespace ur_loader
