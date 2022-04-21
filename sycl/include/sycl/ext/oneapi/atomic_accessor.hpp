//==-- atomic_accessor.hpp - SYCL_ONEAPI_extended_atomics atomic_accessor --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/accessor.hpp>
#include <sycl/ext/oneapi/atomic_enums.hpp>
#include <sycl/ext/oneapi/atomic_ref.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

#if __cplusplus >= 201703L

template <memory_order> struct order_tag_t {
  explicit order_tag_t() = default;
};
inline constexpr order_tag_t<memory_order::relaxed> relaxed_order{};
inline constexpr order_tag_t<memory_order::acquire> acquire_order{};
inline constexpr order_tag_t<memory_order::release> release_order{};
inline constexpr order_tag_t<memory_order::acq_rel> acq_rel_order{};
inline constexpr order_tag_t<memory_order::seq_cst> seq_cst_order{};

template <memory_scope> struct scope_tag_t {
  explicit scope_tag_t() = default;
};
inline constexpr scope_tag_t<memory_scope::work_item> work_item_scope{};
inline constexpr scope_tag_t<memory_scope::sub_group> sub_group_scope{};
inline constexpr scope_tag_t<memory_scope::work_group> work_group_scope{};
inline constexpr scope_tag_t<memory_scope::device> device_scope{};
inline constexpr scope_tag_t<memory_scope::system> system_scope{};

#endif

template <typename DataT, int Dimensions, memory_order DefaultOrder,
          memory_scope DefaultScope,
          access::target AccessTarget = access::target::device,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class atomic_accessor
    : public accessor<DataT, Dimensions, access::mode::read_write, AccessTarget,
                      IsPlaceholder, ext::oneapi::accessor_property_list<>> {

  using AccessorT =
      accessor<DataT, Dimensions, access::mode::read_write, AccessTarget,
               IsPlaceholder, ext::oneapi::accessor_property_list<>>;

private:
  using AccessorT::getLinearIndex;
  using AccessorT::getQualifiedPtr;

  // Prevent non-atomic access to atomic accessor
  multi_ptr<DataT, AccessorT::AS> get_pointer() const = delete;

protected:
  using AccessorT::AdjustedDim;

public:
  using value_type = DataT;
  using reference =
      atomic_ref<DataT, DefaultOrder, DefaultScope, AccessorT::AS>;

  using AccessorT::AccessorT;

#if __cplusplus >= 201703L

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            memory_order Order, memory_scope Scope>
  atomic_accessor(buffer<T, Dims, AllocatorT> &BufferRef, order_tag_t<Order>,
                  scope_tag_t<Scope>, const property_list &PropertyList = {})
      : atomic_accessor(BufferRef, PropertyList) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            memory_order Order, memory_scope Scope>
  atomic_accessor(buffer<T, Dims, AllocatorT> &BufferRef,
                  handler &CommandGroupHandler, order_tag_t<Order>,
                  scope_tag_t<Scope>, const property_list &PropertyList = {})
      : atomic_accessor(BufferRef, CommandGroupHandler, PropertyList) {}

#endif

  // Override subscript operators and conversions to wrap in an atomic_ref
  template <int Dims = Dimensions>
  operator typename detail::enable_if_t<Dims == 0, reference>() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return reference(getQualifiedPtr()[LinearIndex]);
  }

  template <int Dims = Dimensions>
  typename detail::enable_if_t<(Dims > 0), reference>
  operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return reference(getQualifiedPtr()[LinearIndex]);
  }

  template <int Dims = Dimensions>
  typename detail::enable_if_t<Dims == 1, reference>
  operator[](size_t Index) const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>(Index));
    return reference(getQualifiedPtr()[LinearIndex]);
  }
};

#if __cplusplus >= 201703L

template <typename DataT, int Dimensions, typename AllocatorT,
          memory_order Order, memory_scope Scope>
atomic_accessor(buffer<DataT, Dimensions, AllocatorT>, order_tag_t<Order>,
                scope_tag_t<Scope>, property_list = {})
    -> atomic_accessor<DataT, Dimensions, Order, Scope, target::device,
                       access::placeholder::true_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          memory_order Order, memory_scope Scope>
atomic_accessor(buffer<DataT, Dimensions, AllocatorT>, handler,
                order_tag_t<Order>, scope_tag_t<Scope>, property_list = {})
    -> atomic_accessor<DataT, Dimensions, Order, Scope, target::device,
                       access::placeholder::false_t>;

#endif

} // namespace oneapi
} // namespace ext

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
