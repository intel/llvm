/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer_proxy.hpp
 *
 */

#pragma once

#include "ur_shared_layer.hpp"
#include "ur_util.hpp"

namespace ur_sanitizer_layer_proxy {

/// @brief Loader side of the device sanitizer layer.
class __urdlllocal context_t final : public ur_loader::SharedLayer,
                                     public AtomicSingleton<context_t> {
public:
  context_t()
      : SharedLayer(MAKE_LIBRARY_NAME("ur_sanitizer_layer", "0"), getNames()) {}

  /// Known without loading the library. Keep in sync with
  /// ur_sanitizer_layer::context_t.
  static std::vector<std::string> getNames() {
    return {"UR_LAYER_ASAN", "UR_LAYER_MSAN", "UR_LAYER_TSAN"};
  }
};

inline context_t *getContext() { return context_t::get_direct(); }

} // namespace ur_sanitizer_layer_proxy
