//==------- group_helpers_sorters.hpp - SYCL sorters and group helpers -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/group_sort_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// ---- group helpers
template<typename Group, std::size_t Extent>
class group_with_scratchpad
{
    Group g;
    sycl::span<std::uint8_t, Extent> scratch;
public:
    group_with_scratchpad(Group g_, sycl::span<std::uint8_t, Extent> scratch_)
        : g(g_), scratch(scratch_) {}
    Group get_group() const {return g;}
    sycl::span<std::uint8_t, Extent>
    get_memory() const {return scratch;}
};

// ---- sorters
template<typename Compare = std::less<>>
class default_sorter
{
    Compare comp;
    std::uint8_t* scratch;
    std::size_t scratch_size;
public:
    template<std::size_t Extent>
    default_sorter(sycl::span<std::uint8_t, Extent> scratch_, Compare comp_ = Compare())
        : comp(comp_), scratch(scratch_.data()), scratch_size(scratch_.size()) {}

    template<typename Group, typename Ptr>
    void operator()(Group g, Ptr begin, Ptr end)
    {
#ifdef __SYCL_DEVICE_ONLY__
        using T = typename std::iterator_traits<Ptr>::value_type;
        if(scratch_size >= memory_required<T>(sycl::memory_scope::work_group/*Group::fence_scope*/, end - begin))
            cl::sycl::detail::merge_sort(g, begin, end - begin, comp, scratch);
        // TODO: it's better to add else branch
#endif
    }

    template<typename Group, typename T>
    T operator()(Group g, T val)
    {
#ifdef __SYCL_DEVICE_ONLY__
        auto range_size = g.get_local_range(0);
        if(scratch_size >= memory_required<T>(sycl::memory_scope::work_group/*Group::fence_scope*/, range_size))
        {
            uint32_t local_id = id.get_local_id();
            T* temp = reinterpret_cast<T*>(scratch);
            temp[local_id] = val;
            cl::sycl::detail::merge_sort(g, temp, range_size, comp, scratch + range_size * sizeof(T));
            val = temp[local_id];
        }
        // TODO: it's better to add else branch
#endif
        return val;
    }

    template<typename T>
    static constexpr std::size_t
    memory_required(sycl::memory_scope scope, std::size_t range_size)
    {
        return range_size * sizeof(T);
    }

    template<typename T, int dim = 1>
    static constexpr std::size_t
    memory_required(sycl::memory_scope scope, sycl::range<dim> r)
    {
        return 2 * r.size() * sizeof(T);
    }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
