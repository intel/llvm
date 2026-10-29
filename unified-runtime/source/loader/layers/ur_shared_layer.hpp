/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_shared_layer.hpp
 *
 */

#ifndef UR_SHARED_LAYER_HPP
#define UR_SHARED_LAYER_HPP 1

#include "ur_layer_interface.h"
#include "ur_lib_loader.hpp"
#include "ur_proxy_layer.hpp"

#include <string>
#include <vector>

namespace ur_loader {

/// @brief A layer implemented in a separate shared library, so that all loader
///        instances of a process share one instance of it.
///
/// A static loader can be linked into several libraries of one process, each
/// copy with its own hidden state. A layer compiled into the loader would be
/// duplicated along with it, which a layer holding state that has to exist
/// exactly once cannot tolerate.
class __urdlllocal SharedLayer : public proxy_layer_context_t {
public:
  SharedLayer(std::string libraryName, std::vector<std::string> layerNames)
      : libraryName(std::move(libraryName)), layerNames(std::move(layerNames)) {
  }

  ur_result_t init(ur_dditable_t *dditable,
                   const std::set<std::string> &enabledLayerNames,
                   codeloc_data codelocData) override;
  ur_result_t tearDown() override;

private:
  ur_result_t load();

  const std::string libraryName;
  const std::vector<std::string> layerNames;

  LibLoader::Lib library;
  ur_layer_interface_t layerInterface = {};
};

} // namespace ur_loader

#endif /* UR_SHARED_LAYER_HPP */
