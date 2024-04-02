//==-------------- weak_object.hpp --- SYCL weak objects -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for access_mode
#include <sycl/accessor.hpp>      // for host_acce...
#include <sycl/context.hpp>       // for context
#include <sycl/device.hpp>        // for device
#include <sycl/event.hpp>         // for event
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp> // for accessor
#include <sycl/ext/oneapi/weak_object.hpp>      // for weak_object
#include <sycl/kernel.hpp>                      // for kernel
#include <sycl/kernel_bundle.hpp>               // for kernel_id
#include <sycl/kernel_bundle_enums.hpp>         // for bundle_state
#include <sycl/platform.hpp>                    // for platform
#include <sycl/properties/image_properties.hpp> // for sampled_i...
#include <sycl/queue.hpp>                       // for queue
#include <sycl/stream.hpp>                      // for stream

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {

namespace detail {
template <typename SyclObject> struct owner_less_base {
  bool operator()(const SyclObject &lhs, const SyclObject &rhs) const noexcept {
    return lhs.ext_oneapi_owner_before(rhs);
  }

  bool operator()(const weak_object<SyclObject> &lhs,
                  const weak_object<SyclObject> &rhs) const noexcept {
    return lhs.owner_before(rhs);
  }

  bool operator()(const SyclObject &lhs,
                  const weak_object<SyclObject> &rhs) const noexcept {
    return lhs.ext_oneapi_owner_before(rhs);
  }

  bool operator()(const weak_object<SyclObject> &lhs,
                  const SyclObject &rhs) const noexcept {
    return lhs.owner_before(rhs);
  }
};
} // namespace detail

template <typename SyclObject> struct owner_less;

template <>
struct owner_less<context> : public detail::owner_less_base<context> {};
template <>
struct owner_less<device> : public detail::owner_less_base<device> {};
template <> struct owner_less<event> : public detail::owner_less_base<event> {};
template <>
struct owner_less<kernel> : public detail::owner_less_base<kernel> {};
template <>
struct owner_less<kernel_id> : public detail::owner_less_base<kernel_id> {};
template <>
struct owner_less<platform> : public detail::owner_less_base<platform> {};
template <> struct owner_less<queue> : public detail::owner_less_base<queue> {};
template <>
struct owner_less<stream> : public detail::owner_less_base<stream> {};

template <bundle_state State>
struct owner_less<device_image<State>>
    : public detail::owner_less_base<device_image<State>> {};

template <bundle_state State>
struct owner_less<kernel_bundle<State>>
    : public detail::owner_less_base<kernel_bundle<State>> {};

template <typename DataT, int Dimensions, typename AllocatorT>
struct owner_less<buffer<DataT, Dimensions, AllocatorT>>
    : public detail::owner_less_base<buffer<DataT, Dimensions, AllocatorT>> {};

template <int Dimensions, typename AllocatorT>
struct owner_less<unsampled_image<Dimensions, AllocatorT>>
    : public detail::owner_less_base<unsampled_image<Dimensions, AllocatorT>> {
};

template <int Dimensions, typename AllocatorT>
struct owner_less<sampled_image<Dimensions, AllocatorT>>
    : public detail::owner_less_base<sampled_image<Dimensions, AllocatorT>> {};

template <typename DataT, int Dimensions, access_mode AccessMode,
          target AccessTarget, access::placeholder isPlaceholder>
struct owner_less<
    accessor<DataT, Dimensions, AccessMode, AccessTarget, isPlaceholder>>
    : public detail::owner_less_base<accessor<DataT, Dimensions, AccessMode,
                                              AccessTarget, isPlaceholder>> {};

template <typename DataT, int Dimensions, access_mode AccessMode>
struct owner_less<host_accessor<DataT, Dimensions, AccessMode>>
    : public detail::owner_less_base<
          host_accessor<DataT, Dimensions, AccessMode>> {};

template <typename DataT, int Dimensions>
struct owner_less<local_accessor<DataT, Dimensions>>
    : public detail::owner_less_base<local_accessor<DataT, Dimensions>> {};

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget>
struct owner_less<
    unsampled_image_accessor<DataT, Dimensions, AccessMode, AccessTarget>>
    : public detail::owner_less_base<unsampled_image_accessor<
          DataT, Dimensions, AccessMode, AccessTarget>> {};

template <typename DataT, int Dimensions, access_mode AccessMode>
struct owner_less<host_unsampled_image_accessor<DataT, Dimensions, AccessMode>>
    : public detail::owner_less_base<
          host_unsampled_image_accessor<DataT, Dimensions, AccessMode>> {};

template <typename DataT, int Dimensions, image_target AccessTarget>
struct owner_less<sampled_image_accessor<DataT, Dimensions, AccessTarget>>
    : public detail::owner_less_base<
          sampled_image_accessor<DataT, Dimensions, AccessTarget>> {};

template <typename DataT, int Dimensions>
struct owner_less<host_sampled_image_accessor<DataT, Dimensions>>
    : public detail::owner_less_base<
          host_sampled_image_accessor<DataT, Dimensions>> {};

} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
