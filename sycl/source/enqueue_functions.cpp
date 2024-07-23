//==------ enqueue_functions.hpp ------- SYCL enqueue free functions -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT void memcpy(queue Q, void *Dest, const void *Src, size_t NumBytes,
                          const sycl::detail::code_location &CodeLoc) {
  sycl::detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  auto QueueImplPtr = sycl::detail::getSyclObjImpl(Q);
  QueueImplPtr->memcpy(QueueImplPtr, Dest, Src, NumBytes, {},
                       /*CallerNeedsEvent=*/false, CodeLoc);
}

__SYCL_EXPORT void memset(queue Q, void *Ptr, int Value, size_t NumBytes,
                          const sycl::detail::code_location &CodeLoc) {
  sycl::detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  auto QueueImplPtr = sycl::detail::getSyclObjImpl(Q);
  QueueImplPtr->memset(QueueImplPtr, Ptr, Value, NumBytes, {},
                       /*CallerNeedsEvent=*/false);
}

__SYCL_EXPORT void mem_advise(queue Q, void *Ptr, size_t NumBytes, int Advice,
                              const sycl::detail::code_location &CodeLoc) {
  sycl::detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  auto QueueImplPtr = sycl::detail::getSyclObjImpl(Q);
  QueueImplPtr->mem_advise(QueueImplPtr, Ptr, NumBytes, pi_mem_advice(Advice),
                           {}, /*CallerNeedsEvent=*/false);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
