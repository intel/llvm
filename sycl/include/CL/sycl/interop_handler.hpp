//==------- interop_handler.hpp - Argument for codeplay_introp_task --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Interoperability handler
//
class interop_handler {
  // Make accessor class friend to access the detail mem objects
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder,
            typename PropertyListT>
  friend class accessor;

public:
  using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
  using ReqToMem = std::pair<detail::Requirement *, pi_mem>;

  interop_handler(std::vector<ReqToMem> MemObjs, QueueImplPtr Queue)
      : MQueue(std::move(Queue)), MMemObjs(std::move(MemObjs)) {}

  template <backend BackendName = backend::opencl>
  auto get_queue() const -> typename interop<BackendName, queue>::type {
    return reinterpret_cast<typename interop<BackendName, queue>::type>(
        GetNativeQueue());
  }

  template <backend BackendName = backend::opencl, typename DataT, int Dims,
            access::mode AccessMode, access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  auto get_mem(accessor<DataT, Dims, AccessMode, AccessTarget,
                        access::placeholder::false_t>
                   Acc) const ->
      typename interop<BackendName,
                       accessor<DataT, Dims, AccessMode, AccessTarget,
                                access::placeholder::false_t>>::type {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    return getMemImpl<BackendName, DataT, Dims, AccessMode, AccessTarget,
                      access::placeholder::false_t>(
        detail::getSyclObjImpl(*AccBase).get());
  }

private:
  QueueImplPtr MQueue;
  std::vector<ReqToMem> MMemObjs;

  template <backend BackendName, typename DataT, int Dims,
            access::mode AccessMode, access::target AccessTarget,
            access::placeholder IsPlaceholder>
  auto getMemImpl(detail::Requirement *Req) const -> typename interop<
      BackendName,
      accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder>>::type {
    return (typename interop<BackendName,
                             accessor<DataT, Dims, AccessMode, AccessTarget,
                                      IsPlaceholder>>::type)GetNativeMem(Req);
  }

  __SYCL_EXPORT pi_native_handle GetNativeMem(detail::Requirement *Req) const;
  __SYCL_EXPORT pi_native_handle GetNativeQueue() const;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
