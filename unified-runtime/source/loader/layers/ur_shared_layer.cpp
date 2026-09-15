/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_shared_layer.cpp
 *
 */

#include "ur_shared_layer.hpp"
#include "logger/ur_logger.hpp"
#include "ur_adapter_search.hpp"

#include <algorithm>

namespace ur_loader {

namespace {

/// @brief Candidate paths for a layer library, belonging to this loader first.
std::vector<fs::path> getLayerLoadPaths(const std::string &libraryName) {
  std::vector<fs::path> paths;

  if (auto dir = getLoaderLibPath(); dir.has_value()) {
    paths.emplace_back(dir.value() / libraryName);
  }
  if (auto path = getAdapterNameAsPath(libraryName); path.has_value()) {
    paths.emplace_back(path.value());
  }

  return paths;
}

} // namespace

ur_result_t SharedLayer::load() {
  if (library) {
    return UR_RESULT_SUCCESS;
  }

  for (const auto &path : getLayerLoadPaths(libraryName)) {
    auto handle = LibLoader::loadAdapterLibrary(path.string().c_str());
    if (!handle) {
      continue;
    }

    auto pfnGetInterface = reinterpret_cast<ur_pfnLoaderLayerGetInterface_t>(
        LibLoader::getFunctionPtr(handle.get(),
                                  UR_LAYER_GET_INTERFACE_FUNC_NAME));
    ur_layer_interface_t candidate = {};
    if (!pfnGetInterface ||
        pfnGetInterface(UR_LAYER_INTERFACE_VERSION, &candidate) !=
            UR_RESULT_SUCCESS ||
        !candidate.pfnInit || !candidate.pfnTearDown) {
      UR_LOG(ERR, "{} doesn't implement layer interface version {}",
             path.string(), UR_LAYER_INTERFACE_VERSION);
      continue;
    }

    library = std::move(handle);
    layerInterface = candidate;
    return UR_RESULT_SUCCESS;
  }

  // At QUIET, which always prints: this aborts the loader initialization.
  UR_LOG(QUIET,
         "the {} layer is enabled but its library couldn't be loaded, make "
         "sure it is installed next to the loader",
         libraryName);

  return UR_RESULT_ERROR_LAYER_NOT_PRESENT;
}

ur_result_t SharedLayer::init(ur_dditable_t *dditable,
                              const std::set<std::string> &enabledLayerNames,
                              codeloc_data) {
  bool enabled = std::any_of(
      layerNames.begin(), layerNames.end(), [&enabledLayerNames](auto &name) {
        return enabledLayerNames.find(name) != enabledLayerNames.end();
      });
  if (!enabled) {
    return UR_RESULT_SUCCESS;
  }

  if (ur_result_t result = load(); result != UR_RESULT_SUCCESS) {
    return result;
  }

  // The layer picks the names it implements out of all the enabled ones itself.
  std::vector<const char *> names;
  names.reserve(enabledLayerNames.size());
  for (const auto &name : enabledLayerNames) {
    names.push_back(name.c_str());
  }

  return layerInterface.pfnInit(dditable, names.data(),
                                static_cast<uint32_t>(names.size()));
}

ur_result_t SharedLayer::tearDown() {
  if (!library) {
    return UR_RESULT_SUCCESS;
  }

  ur_result_t result = layerInterface.pfnTearDown();
  layerInterface = {};
  // Stays mapped as long as another loader instance still references it.
  library.reset();

  return result;
}

} // namespace ur_loader
