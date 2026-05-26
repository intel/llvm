/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UR_ADAPTER_SEARCH_HPP
#define UR_ADAPTER_SEARCH_HPP 1

#include <optional>

#include "ur_filesystem_resolved.hpp"

namespace fs = filesystem;

namespace ur_loader {

std::optional<fs::path> getLoaderLibPath();
std::optional<fs::path> getAdapterNameAsPath(std::string adapterName);

} // namespace ur_loader

#endif /* UR_ADAPTER_SEARCH_HPP */
