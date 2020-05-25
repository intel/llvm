//==- sub_group_host.hpp --- SYCL sub-group for host device  ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/intel/functional.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>
#ifndef __SYCL_DEVICE_ONLY__

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <typename T, access::address_space Space> class multi_ptr;
namespace intel {
struct sub_group {

  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = size_t;
  static constexpr int dimensions = 1;

  /* --- common interface members --- */

  id<1> get_local_id() const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }
  range<1> get_local_range() const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  range<1> get_max_local_range() const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  id<1> get_group_id() const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  size_t get_group_range() const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  size_t get_uniform_group_range() const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  /* --- vote / ballot functions --- */

  bool any(bool) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  bool all(bool) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  /* --- collectives --- */

  template <typename T> T broadcast(T, id<1>) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, class BinaryOperation>
  T reduce(T, BinaryOperation) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, class BinaryOperation>
  T reduce(T, T, BinaryOperation) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, class BinaryOperation>
  T exclusive_scan(T, BinaryOperation) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, class BinaryOperation>
  T exclusive_scan(T, T, BinaryOperation) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, class BinaryOperation>
  T inclusive_scan(T, BinaryOperation) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, class BinaryOperation>
  T inclusive_scan(T, BinaryOperation, T) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  /* --- one - input shuffles --- */
  /* indices in [0 , sub - group size ) */

  template <typename T> T shuffle(T, id<1>) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T> T shuffle_down(T, uint32_t) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }
  template <typename T> T shuffle_up(T, uint32_t) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T> T shuffle_xor(T, id<1>) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  /* --- two - input shuffles --- */
  /* indices in [0 , 2* sub - group size ) */
  template <typename T> T shuffle(T, T, id<1>) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }
  template <typename T> T shuffle_down(T, T, uint32_t) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }
  template <typename T> T shuffle_up(T, T, uint32_t) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  /* --- sub - group load / stores --- */
  /* these can map to SIMD or block read / write hardware where available */
  template <typename T, access::address_space Space>
  T load(const multi_ptr<T, Space>) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <int N, typename T, access::address_space Space>
  vec<T, N> load(const multi_ptr<T, Space>) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <typename T, access::address_space Space>
  void store(multi_ptr<T, Space>, const T &) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  template <int N, typename T, access::address_space Space>
  void store(multi_ptr<T, Space>, const vec<T, N> &) const {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

  /* --- synchronization functions --- */
  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    (void)accessSpace;
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }

protected:
  template <int dimensions> friend class cl::sycl::nd_item;
  sub_group() {
    throw runtime_error("Subgroups are not supported on host device. ",
                        PI_INVALID_DEVICE);
  }
};
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif
