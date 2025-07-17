//==--------- dynamic_impl.hpp - SYCL graph extension ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/accessor_impl.hpp> // for LocalAccessorImplHost
#include <sycl/detail/cg_types.hpp> // for CGType
#include <sycl/ext/oneapi/experimental/graph/common.hpp>   // for graph_state
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp> // for raw_kernel_arg

#include <cstring> // for memcpy
#include <vector>  // for vector

namespace sycl {
inline namespace _V1 {
// Forward declarations
class handler;

// Forward declarations
namespace detail {
class CG;
} // namespace detail

namespace ext {
namespace oneapi {
namespace experimental {
// Forward declarations
template <graph_state State> class command_graph;
template <typename, typename> class dynamic_work_group_memory;
template <typename ValueT> class dynamic_parameter;
class dynamic_command_group;

namespace detail {
// Forward declarations
class node_impl;
class graph_impl;

class dynamic_command_group_impl
    : public std::enable_shared_from_this<dynamic_command_group_impl> {
public:
  dynamic_command_group_impl(
      const command_graph<graph_state::modifiable> &Graph);

  /// Returns the index of the active command-group
  size_t getActiveIndex() const { return MActiveCGF; }

  /// Returns the number of CGs in the dynamic command-group.
  size_t getNumCGs() const { return MCommandGroups.size(); }

  /// Set the index of the active command-group.
  /// @param Index The new index.
  void setActiveIndex(size_t Index);

  /// Instantiates a command-group object for each CGF in the list.
  /// @param CGFList List of CGFs to finalize with a handler into CG objects.
  void
  finalizeCGFList(const std::vector<std::function<void(handler &)>> &CGFList);

  /// Retrieve CG at the currently active index
  /// @param Shared pointer to the active CG object.
  std::shared_ptr<sycl::detail::CG> getActiveCG() const {
    return MCommandGroups[MActiveCGF];
  }

  /// Graph this dynamic command-group is associated with.
  std::shared_ptr<graph_impl> MGraph;

  /// Index of active command-group
  std::atomic<size_t> MActiveCGF;

  /// List of command-groups for dynamic command-group nodes
  std::vector<std::shared_ptr<sycl::detail::CG>> MCommandGroups;

  /// List of nodes using this dynamic command-group.
  std::vector<std::weak_ptr<node_impl>> MNodes;

  unsigned long long getID() const { return MID; }

  /// Type of the CGs in this dynamic command-group
  sycl::detail::CGType MCGType = sycl::detail::CGType::None;

private:
  unsigned long long MID;
  // Used for std::hash in order to create a unique hash for the instance.
  inline static std::atomic<unsigned long long> NextAvailableID = 0;
};

class dynamic_parameter_impl {
public:
  dynamic_parameter_impl()
      : MID(NextAvailableID.fetch_add(1, std::memory_order_relaxed)) {}

  dynamic_parameter_impl(size_t ParamSize, const void *Data)
      : MValueStorage(ParamSize),
        MID(NextAvailableID.fetch_add(1, std::memory_order_relaxed)) {
    std::memcpy(MValueStorage.data(), Data, ParamSize);
  }

  /// sycl_ext_oneapi_raw_kernel_arg constructor
  /// Parameter size is taken from member of raw_kernel_arg object.
  dynamic_parameter_impl(size_t, raw_kernel_arg *Data)
      : MID(NextAvailableID.fetch_add(1, std::memory_order_relaxed)) {
    size_t RawArgSize = Data->MArgSize;
    const void *RawArgData = Data->MArgData;
    MValueStorage.reserve(RawArgSize);
    std::memcpy(MValueStorage.data(), RawArgData, RawArgSize);
  }

  /// Register a node with this dynamic parameter
  /// @param NodeImpl The node to be registered
  /// @param ArgIndex The arg index for the kernel arg associated with this
  /// dynamic_parameter in NodeImpl
  void registerNode(std::shared_ptr<node_impl> NodeImpl, int ArgIndex) {
    MNodes.emplace_back(NodeImpl, ArgIndex);
  }

  /// Struct detailing an instance of the usage of the dynamic parameter in a
  /// dynamic CG.
  struct DynamicCGInfo {
    /// Dynamic command-group that uses this dynamic parameter.
    std::weak_ptr<dynamic_command_group_impl> DynCG;
    /// Index of the CG in the Dynamic CG that uses this dynamic parameter.
    size_t CGIndex;
    /// The arg index in the kernel the dynamic parameter is used.
    int ArgIndex;
  };

  /// Registers a dynamic command-group with this dynamic parameter.
  /// @param DynCG The dynamic command-group to register.
  /// @param CGIndex Index of the CG in DynCG using this dynamic parameter.
  /// @param ArgIndex The arg index in the kernel the dynamic parameter is used.
  void registerDynCG(std::shared_ptr<dynamic_command_group_impl> DynCG,
                     size_t CGIndex, int ArgIndex) {
    MDynCGs.emplace_back(DynamicCGInfo{DynCG, CGIndex, ArgIndex});
  }

  /// Get a pointer to the internal value of this dynamic parameter
  void *getValue() { return MValueStorage.data(); }

  /// Update sycl_ext_oneapi_raw_kernel_arg parameter
  /// @param NewRawValue Pointer to a raw_kernel_arg object.
  /// @param Size Parameter is ignored.
  void updateValue(const raw_kernel_arg *NewRawValue, size_t Size);

  /// Update the internal value of this dynamic parameter as well as the value
  /// of this parameter in all registered nodes and dynamic CGs.
  /// @param NewValue Pointer to the new value
  /// @param Size Size of the data pointer to by NewValue
  void updateValue(const void *NewValue, size_t Size);

  /// Update the internal value of this dynamic parameter as well as the value
  /// of this parameter in all registered nodes and dynamic CGs. Should only be
  /// called for accessor dynamic_parameters.
  /// @param Acc The new accessor value
  void updateAccessor(const sycl::detail::AccessorBaseHost *Acc);

  /// Static helper function for updating command-group value arguments.
  /// @param CG The command-group to update the argument information for.
  /// @param ArgIndex The argument index to update.
  /// @param NewValue Pointer to the new value.
  /// @param Size Size of the data pointer to by NewValue
  static void updateCGArgValue(std::shared_ptr<sycl::detail::CG> CG,
                               int ArgIndex, const void *NewValue, size_t Size);

  /// Static helper function for updating command-group accessor arguments.
  /// @param CG The command-group to update the argument information for.
  /// @param ArgIndex The argument index to update.
  /// @param Acc The new accessor value
  static void updateCGAccessor(std::shared_ptr<sycl::detail::CG> CG,
                               int ArgIndex,
                               const sycl::detail::AccessorBaseHost *Acc);

  unsigned long long getID() const { return MID; }

  // Weak ptrs to node_impls which will be updated
  std::vector<std::pair<std::weak_ptr<node_impl>, int>> MNodes;
  // Dynamic command-groups which will be updated
  std::vector<DynamicCGInfo> MDynCGs;
  std::vector<std::byte> MValueStorage;

private:
  unsigned long long MID;
  // Used for std::hash in order to create a unique hash for the instance.
  inline static std::atomic<unsigned long long> NextAvailableID = 0;
};

class dynamic_work_group_memory_impl : public dynamic_parameter_impl {

public:
  dynamic_work_group_memory_impl(size_t BufferSizeInBytes)
      : BufferSizeInBytes(BufferSizeInBytes) {}

  virtual ~dynamic_work_group_memory_impl() = default;

  /// Update the internal value of this dynamic parameter as well as the value
  /// of this parameter in all registered nodes and dynamic CGs.
  /// @param NewBufferSizeInBytes The total size in bytes of the new
  /// work_group_memory array.
  void updateWorkGroupMem(size_t NewBufferSizeInBytes);

  /// Static helper function for updating command-group
  /// dynamic_work_group_memory arguments.
  /// @param CG The command-group to update the argument information for.
  /// @param ArgIndex The argument index to update.
  /// @param NewBufferSizeInBytes The total size in bytes of the new
  /// work_group_memory array.
  void updateCGWorkGroupMem(std::shared_ptr<sycl::detail::CG> &CG, int ArgIndex,
                            size_t NewBufferSizeInBytes);

  size_t BufferSizeInBytes;
};

class dynamic_local_accessor_impl : public dynamic_parameter_impl {

public:
  dynamic_local_accessor_impl(sycl::range<3> AllocationSize, int Dims,
                              int ElemSize, const property_list &PropList);

  virtual ~dynamic_local_accessor_impl() = default;

  /// Update the internal value of this dynamic parameter as well as the value
  /// of this parameter in all registered nodes and dynamic CGs.
  /// @param NewAllocationSize The new allocation size for the
  /// dynamic_local_accessor.
  void updateLocalAccessor(range<3> NewAllocationSize);

  /// Static helper function for updating command-group dynamic_local_accessor
  /// arguments.
  /// @param CG The command-group to update the argument information for.
  /// @param ArgIndex The argument index to update.
  /// @param NewAllocationSize The new allocation size for the
  /// dynamic_local_accessor.
  void updateCGLocalAccessor(std::shared_ptr<sycl::detail::CG> &CG,
                             int ArgIndex, range<3> NewAllocationSize);

  sycl::detail::LocalAccessorImplHost LAccImplHost;
};
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

namespace std {
template <>
struct __SYCL_EXPORT
    hash<sycl::ext::oneapi::experimental::dynamic_command_group> {
  size_t operator()(const sycl::ext::oneapi::experimental::dynamic_command_group
                        &DynamicCGH) const;
};

template <typename ValueT>
struct hash<sycl::ext::oneapi::experimental::dynamic_parameter<ValueT>> {
  size_t
  operator()(const sycl::ext::oneapi::experimental::dynamic_parameter<ValueT>
                 &DynamicParam) const {
    auto ID = sycl::detail::getSyclObjImpl(DynamicParam)->getID();
    return std::hash<decltype(ID)>()(ID);
  }
};

template <typename DataT, typename PropertyListT>
struct hash<sycl::ext::oneapi::experimental::dynamic_work_group_memory<
    DataT, PropertyListT>> {
  size_t
  operator()(const sycl::ext::oneapi::experimental::dynamic_work_group_memory<
             DataT, PropertyListT> &DynWorkGroupMem) const {
    auto ID = sycl::detail::getSyclObjImpl(DynWorkGroupMem)->getID();
    return std::hash<decltype(ID)>()(ID);
  }
};
} // namespace std
