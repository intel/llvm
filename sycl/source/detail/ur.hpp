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
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
enum class backend : char;
namespace detail {
class Adapter;

namespace ur {
void *getURLoaderLibrary();

// Performs UR one-time initialization.
std::vector<Adapter*>&
initializeUr(ur_loader_config_handle_t LoaderConfig = nullptr);

// Get the adapter serving given backend.
template <backend BE> Adapter& getAdapter();
} // namespace ur

// Convert from UR backend to SYCL backend enum
backend convertUrBackend(ur_backend_t UrBackend);

template <auto ApiKind, typename SyclImplTy, typename DescTy>
std::string urGetInfoString(SyclImplTy &SyclImpl, DescTy Desc) {
  // Avoid explicit type to keep template-type-dependent.
  auto &adapter = SyclImpl.getAdapter();
  size_t ResultSize = 0;
  auto Handle = SyclImpl.getHandleRef();
  adapter.template call<ApiKind>(Handle, Desc,
                                  /*propSize=*/0,
                                  /*pPropValue=*/nullptr, &ResultSize);
  if (ResultSize == 0)
    return std::string{};

  std::string Result;
  // C++23's `resize_and_overwrite` would be better...
  //
  // UR counts null terminator in the size, std::string doesn't. Adjust by "-1"
  // for that.
  Result.resize(ResultSize - 1);
  adapter.template call<ApiKind>(Handle, Desc, ResultSize, Result.data(),
                                  nullptr);

  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
