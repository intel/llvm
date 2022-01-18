//==------- group_helpers_sorters.hpp - SYCL sorters and group helpers -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
#include <CL/sycl/detail/group_sort_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// ---- group helpers
template <typename Group, std::size_t Extent> class group_with_scratchpad {
  Group g;
  sycl::span<std::byte, Extent> scratch;

public:
  group_with_scratchpad(Group g_, sycl::span<std::byte, Extent> scratch_)
      : g(g_), scratch(scratch_) {}
  Group get_group() const { return g; }
  sycl::span<std::byte, Extent> get_memory() const { return scratch; }
};

// ---- sorters
template <typename Compare = std::less<>> class default_sorter {
  Compare comp;
  std::byte *scratch;
  std::size_t scratch_size;

public:
  template <std::size_t Extent>
  default_sorter(sycl::span<std::byte, Extent> scratch_,
                 Compare comp_ = Compare())
      : comp(comp_), scratch(scratch_.data()), scratch_size(scratch_.size()) {}

  template <typename Group, typename Ptr>
  void operator()(Group g, Ptr first, Ptr last) {
#ifdef __SYCL_DEVICE_ONLY__
    using T = typename sycl::detail::GetValueType<Ptr>::type;
    if (scratch_size >= memory_required<T>(Group::fence_scope, last - first))
      sycl::detail::merge_sort(g, first, last - first, comp, scratch);
      // TODO: it's better to add else branch
#else
    (void)g;
    (void)first;
    (void)last;
    throw sycl::exception(
        std::error_code(PI_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter constructor is not supported on host device.");
#endif
  }

  template <typename Group, typename T> T operator()(Group g, T val) {
#ifdef __SYCL_DEVICE_ONLY__
    auto range_size = g.get_local_range().size();
    if (scratch_size >= memory_required<T>(Group::fence_scope, range_size)) {
      auto id = sycl::detail::Builder::getNDItem<Group::dimensions>();
      std::size_t local_id = id.get_local_linear_id();
      T *temp = reinterpret_cast<T *>(scratch);
      ::new (temp + local_id) T(val);
      sycl::detail::merge_sort(g, temp, range_size, comp,
                               scratch + range_size * sizeof(T));
      val = temp[local_id];
    }
    // TODO: it's better to add else branch
#else
    (void)g;
    throw sycl::exception(
        std::error_code(PI_INVALID_DEVICE, sycl::sycl_category()),
        "default_sorter operator() is not supported on host device.");
#endif
    return val;
  }

  template <typename T>
  static constexpr std::size_t memory_required(sycl::memory_scope,
                                               std::size_t range_size) {
    return range_size * sizeof(T) + alignof(T);
  }

  template <typename T, int dim = 1>
  static constexpr std::size_t memory_required(sycl::memory_scope scope,
                                               sycl::range<dim> r) {
    return 2 * memory_required<T>(scope, r.size());
  }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif // __cplusplus >=201703L
