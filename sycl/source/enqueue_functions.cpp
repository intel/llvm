//==------ enqueue_functions.hpp ------- SYCL enqueue free functions -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

__SYCL_EXPORT void memcpy(queue Q, void *Dest, const void *Src, size_t NumBytes,
                          const sycl::detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  auto QueueImplPtr = detail::getSyclObjImpl(Q);
  return QueueImplPtr->memcpy(QueueImplPtr, Dest, Src, NumBytes, {},
                              /*CallerNeedsEvent=*/false, CodeLoc);
}

__SYCL_EXPORT void memset(queue Q, void *Ptr, int Value, size_t NumBytes) {
  auto QueueImplPtr = detail::getSyclObjImpl(Q);
  return QueueImplPtr->memset(QueueImplPtr, Ptr, Value, NumBytes, {},
                              /*CallerNeedsEvent=*/false);
}

__SYCL_EXPORT void mem_advise(queue Q, void *Ptr, size_t NumBytes, int Advice) {
  auto QueueImplPtr = detail::getSyclObjImpl(Q);
  return QueueImplPtr->mem_advise(QueueImplPtr, Ptr, NumBytes,
                                  pi_mem_advice(Advice), {},
                                  /*CallerNeedsEvent=*/false);
}
