//==---------- ur.hpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#pragma once

#include <sycl/backend_types.hpp>
#include <ur_api.h>

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {
class Adapter;
using AdapterPtr = std::shared_ptr<Adapter>;

namespace ur {
void *getURLoaderLibrary();

// Performs UR one-time initialization.
std::vector<AdapterPtr> &
initializeUr(ur_loader_config_handle_t LoaderConfig = nullptr);

// Get the adapter serving given backend.
template <backend BE> const AdapterPtr &getAdapter();
} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
