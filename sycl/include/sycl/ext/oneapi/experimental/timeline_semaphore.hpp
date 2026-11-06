//==------ timeline_semaphore.hpp --- SYCL timeline semaphores -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/queue.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace sycl {
inline namespace _V1 {

namespace detail {
class timeline_semaphore_impl;
} // namespace detail

namespace ext::oneapi::experimental {

class __SYCL_EXPORT timeline_semaphore
    : public sycl::detail::OwnerLessBase<timeline_semaphore> {
  friend sycl::detail::ImplUtils;

public:
  template <typename PropertyListT = empty_properties_t,
            typename = std::enable_if_t<detail::all_are_properties_of_v<
                timeline_semaphore, PropertyListT>>>
  explicit timeline_semaphore(const context &ctxt, uint64_t initial_value = 0,
                              PropertyListT = {}) {
    create(ctxt, initial_value);
  }

  timeline_semaphore(const timeline_semaphore &rhs) = default;
  timeline_semaphore(timeline_semaphore &&rhs) = default;

  timeline_semaphore &operator=(const timeline_semaphore &rhs) = default;
  timeline_semaphore &operator=(timeline_semaphore &&rhs) = default;

  ~timeline_semaphore() = default;

  bool operator==(const timeline_semaphore &rhs) const {
    return impl == rhs.impl;
  }
  bool operator!=(const timeline_semaphore &rhs) const {
    return !(*this == rhs);
  }

  context get_context() const;

  void wait(uint64_t value) const;

  void signal(uint64_t value);

private:
  std::shared_ptr<sycl::detail::timeline_semaphore_impl> impl;

  void create(const context &ctxt, uint64_t initial_value);
  timeline_semaphore(
      std::shared_ptr<sycl::detail::timeline_semaphore_impl> Impl)
      : impl(std::move(Impl)) {}
};

__SYCL_EXPORT void enqueue_signal(queue q, timeline_semaphore &sem,
                                  uint64_t value,
                                  const sycl::detail::code_location &CodeLoc =
                                      sycl::detail::code_location::current());

__SYCL_EXPORT void enqueue_wait(queue q, const timeline_semaphore &sem,
                                uint64_t value,
                                const sycl::detail::code_location &CodeLoc =
                                    sycl::detail::code_location::current());

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

template <>
struct std::hash<sycl::ext::oneapi::experimental::timeline_semaphore>
    : public sycl::detail::sycl_obj_hash<
          sycl::ext::oneapi::experimental::timeline_semaphore> {};
