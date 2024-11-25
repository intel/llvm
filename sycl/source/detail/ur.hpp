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

#include <ur_api.h>

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
enum class backend : char;
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

// Convert from UR backend to SYCL backend enum
backend convertUrBackend(ur_backend_t UrBackend);

} // namespace detail
} // namespace _V1
} // namespace sycl
