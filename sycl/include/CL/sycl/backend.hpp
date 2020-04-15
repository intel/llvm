//==---------------- backend.hpp - SYCL PI backends ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj) ->
    typename interop<BackendName, SyclObjectT>::type {
  return Obj.template get_native<BackendName>();
}

// Native handle of an accessor should be accessed through interop_handler
template <backend BackendName, typename DataT, int Dimensions,
          access::mode AccessMode, access::target AccessTarget,
          access::placeholder IsPlaceholder>
auto get_native(const accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder> &Obj) ->
    typename interop<BackendName, accessor<DataT, Dimensions, AccessMode,
                                           AccessTarget, IsPlaceholder>>::type =
    delete;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
