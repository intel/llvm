//==--------- dynamic.hpp --- SYCL graph extension -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "command_graph.hpp"
#include "common.hpp"                  // for graph_state
#include <sycl/accessor.hpp>           // for local_accessor
#include <sycl/detail/export.hpp>      // for __SYCL_EXPORT
#include <sycl/detail/kernel_desc.hpp> // for kernel_param_kind_t
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp> // for work_group_memory
#include <sycl/ext/oneapi/properties/properties.hpp> // for empty_properties_t

#include <functional> // for function
#include <memory>     // for shared_ptr
#include <vector>     // for vector

namespace sycl {
inline namespace _V1 {
// Forward declarations
class handler;
class property_list;

namespace detail {
// Forward declarations
class AccessorBaseHost;
} // namespace detail

namespace ext {
namespace oneapi {
namespace experimental {
// Forward declarations
class raw_kernel_arg;
template <typename, typename> class work_group_memory;

namespace detail {
// Forward declarations
class dynamic_parameter_impl;
class dynamic_command_group_impl;
} // namespace detail

class __SYCL_EXPORT dynamic_command_group {
public:
  dynamic_command_group(
      const command_graph<graph_state::modifiable> &Graph,
      const std::vector<std::function<void(handler &)>> &CGFList);

  size_t get_active_index() const;
  void set_active_index(size_t Index);

  /// Common Reference Semantics
  friend bool operator==(const dynamic_command_group &LHS,
                         const dynamic_command_group &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const dynamic_command_group &LHS,
                         const dynamic_command_group &RHS) {
    return !operator==(LHS, RHS);
  }

private:
  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  std::shared_ptr<detail::dynamic_command_group_impl> impl;
};

namespace detail {
class __SYCL_EXPORT dynamic_parameter_base {
public:
  dynamic_parameter_base(size_t ParamSize, const void *Data);
  dynamic_parameter_base();

  dynamic_parameter_base(
      const std::shared_ptr<detail::dynamic_parameter_impl> &impl);

  /// Common Reference Semantics
  friend bool operator==(const dynamic_parameter_base &LHS,
                         const dynamic_parameter_base &RHS) {
    return LHS.impl == RHS.impl;
  }
  friend bool operator!=(const dynamic_parameter_base &LHS,
                         const dynamic_parameter_base &RHS) {
    return !operator==(LHS, RHS);
  }

protected:
  void updateValue(const void *NewValue, size_t Size);

  // Update a sycl_ext_oneapi_raw_kernel_arg parameter. Size parameter is
  // ignored as it represents sizeof(raw_kernel_arg), which doesn't represent
  // the number of underlying bytes.
  void updateValue(const raw_kernel_arg *NewRawValue, size_t Size);

  void updateAccessor(const sycl::detail::AccessorBaseHost *Acc);

  std::shared_ptr<dynamic_parameter_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
};

class __SYCL_EXPORT dynamic_work_group_memory_base
    : public dynamic_parameter_base {

public:
  dynamic_work_group_memory_base() = default;

  dynamic_work_group_memory_base(size_t BufferSizeInBytes);

protected:
  void updateWorkGroupMem(size_t NewBufferSizeInBytes);
};

class __SYCL_EXPORT dynamic_local_accessor_base
    : public dynamic_parameter_base {
public:
  dynamic_local_accessor_base() = default;

  dynamic_local_accessor_base(sycl::range<3> AllocationSize, int Dims,
                              int ElemSize, const property_list &PropList);

protected:
  void updateLocalAccessor(sycl::range<3> NewAllocationSize);
};
} // namespace detail

template <typename DataT, typename PropertyListT = empty_properties_t>
class __SYCL_SPECIAL_CLASS __SYCL_TYPE(dynamic_work_group_memory)
    dynamic_work_group_memory
#ifndef __SYCL_DEVICE_ONLY__
    : public detail::dynamic_work_group_memory_base
#endif
{
public:
  // Check that DataT is an unbounded array type.
  static_assert(std::is_array_v<DataT> && std::extent_v<DataT, 0> == 0);
  static_assert(std::is_same_v<PropertyListT, empty_properties_t>);

  // Frontend requires special types to have a default constructor in order to
  // have a uniform way of initializing an object of special type to then call
  // the __init method on it. This is purely an implementation detail and not
  // part of the spec.
  // TODO: Revisit this once https://github.com/intel/llvm/issues/16061 is
  // closed.
  dynamic_work_group_memory() = default;

#ifndef __SYCL_DEVICE_ONLY__
  /// Constructs a new dynamic_work_group_memory object.
  /// @param Num Number of elements in the unbounded array DataT.
  dynamic_work_group_memory(size_t Num)
      : detail::dynamic_work_group_memory_base(
            Num * sizeof(std::remove_extent_t<DataT>)) {}
#else
  dynamic_work_group_memory(size_t /*Num*/) {}
#endif

  work_group_memory<DataT, PropertyListT> get() const {
#ifndef __SYCL_DEVICE_ONLY__
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Error: dynamic_work_group_memory::get() can be only "
                          "called on the device!");
#endif
    return WorkGroupMem;
  }

  /// Updates on the host this dynamic_work_group_memory and all registered
  /// nodes with a new buffer size.
  /// @param Num The new number of elements in the unbounded array.
  void update([[maybe_unused]] size_t Num) {
#ifndef __SYCL_DEVICE_ONLY__
    updateWorkGroupMem(Num * sizeof(std::remove_extent_t<DataT>));
#endif
  }

private:
  work_group_memory<DataT, PropertyListT> WorkGroupMem;

#ifdef __SYCL_DEVICE_ONLY__
  using value_type = std::remove_all_extents_t<DataT>;
  using decoratedPtr = typename sycl::detail::DecoratedType<
      value_type, access::address_space::local_space>::type *;

  void __init(decoratedPtr Ptr) { this->WorkGroupMem.__init(Ptr); }

  [[maybe_unused]] unsigned char
      Padding[sizeof(detail::dynamic_work_group_memory_base)];
#endif
};

template <typename DataT, int Dimensions = 1>
class __SYCL_SPECIAL_CLASS __SYCL_TYPE(dynamic_local_accessor)
    dynamic_local_accessor
#ifndef __SYCL_DEVICE_ONLY__
    : public detail::dynamic_local_accessor_base
#endif
{
public:
  static_assert(Dimensions > 0 && Dimensions <= 3);

  // Frontend requires special types to have a default constructor in order to
  // have a uniform way of initializing an object of special type to then call
  // the __init method on it. This is purely an implementation detail and not
  // part of the spec.
  // TODO: Revisit this once https://github.com/intel/llvm/issues/16061 is
  // closed.
  dynamic_local_accessor() = default;

#ifndef __SYCL_DEVICE_ONLY__
  /// Constructs a new dynamic_local_accessor object.
  /// @param AllocationSize The size of the local accessor.
  /// @param PropList List of properties for the underlying accessor.
  dynamic_local_accessor(range<Dimensions> AllocationSize,
                         const property_list &PropList = {})
      : detail::dynamic_local_accessor_base(
            detail::convertToArrayOfN<3, 1>(AllocationSize), Dimensions,
            sizeof(DataT), PropList) {}
#else
  dynamic_local_accessor(range<Dimensions> /* AllocationSize */,
                         const property_list & /*PropList */ = {}) {}
#endif

  local_accessor<DataT, Dimensions> get() const {
#ifndef __SYCL_DEVICE_ONLY__
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Error: dynamic_local_accessor::get() can be only "
                          "called on the device!");
#endif
    return LocalAccessor;
  }

  /// Updates on the host this dynamic_local_accessor and all registered
  /// nodes with a new size.
  /// @param Num The new number of elements in the unbounded array.
  void update([[maybe_unused]] range<Dimensions> NewAllocationSize) {
#ifndef __SYCL_DEVICE_ONLY__
    updateLocalAccessor(detail::convertToArrayOfN<3, 1>(NewAllocationSize));
#endif
  }

private:
  local_accessor<DataT, Dimensions> LocalAccessor;

#ifdef __SYCL_DEVICE_ONLY__
  void __init(typename local_accessor<DataT, Dimensions>::ConcreteASPtrType Ptr,
              range<Dimensions> AccessRange, range<Dimensions> range,
              id<Dimensions> id) {
    this->LocalAccessor.__init(Ptr, AccessRange, range, id);
  }

  [[maybe_unused]] unsigned char
      Padding[sizeof(detail::dynamic_local_accessor_base)];
#endif
};

template <typename ValueT>
class dynamic_parameter : public detail::dynamic_parameter_base {
  static constexpr bool IsAccessor =
      std::is_base_of_v<sycl::detail::AccessorBaseHost, ValueT>;
  static constexpr sycl::detail::kernel_param_kind_t ParamType =
      IsAccessor ? sycl::detail::kernel_param_kind_t::kind_accessor
      : std::is_pointer_v<ValueT>
          ? sycl::detail::kernel_param_kind_t::kind_pointer
          : sycl::detail::kernel_param_kind_t::kind_std_layout;

public:
  /// Constructs a new dynamic parameter.
  /// @param Graph The graph associated with this parameter.
  /// @param Param A reference value for this parameter used for CTAD.
  dynamic_parameter(const ValueT &Param)
      : detail::dynamic_parameter_base(sizeof(ValueT), &Param) {}

  /// Updates this dynamic parameter and all registered nodes with a new value.
  /// @param NewValue The new value for the parameter.
  void update(const ValueT &NewValue) {
    if constexpr (IsAccessor) {
      detail::dynamic_parameter_base::updateAccessor(&NewValue);
    } else {
      detail::dynamic_parameter_base::updateValue(&NewValue, sizeof(ValueT));
    }
  }
};

/// Additional CTAD deduction guides.
template <typename ValueT>
dynamic_parameter(
    const experimental::command_graph<graph_state::modifiable> &Graph,
    const ValueT &Param) -> dynamic_parameter<ValueT>;

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
