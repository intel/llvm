//==------------ accessor.hpp - SYCL standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>                     // for target, mode
#include <sycl/aliases.hpp>                           // for float4, int4
#include <sycl/aspects.hpp>                           // for aspect
#include <sycl/atomic.hpp>                            // for atomic
#include <sycl/buffer.hpp>                            // for range
#include <sycl/detail/accessor_iterator.hpp>          // for accessor_iterator
#include <sycl/detail/common.hpp>                     // for code_location
#include <sycl/detail/defines.hpp>                    // for __SYCL_SPECIAL...
#include <sycl/detail/defines_elementary.hpp>         // for __SYCL2020_DEP...
#include <sycl/detail/export.hpp>                     // for __SYCL_EXPORT
#include <sycl/detail/generic_type_traits.hpp>        // for is_genint, Try...
#include <sycl/detail/handler_proxy.hpp>              // for associateWithH...
#include <sycl/detail/helpers.hpp>                    // for loop
#include <sycl/detail/image_accessor_util.hpp>        // for imageReadSampl...
#include <sycl/detail/owner_less_base.hpp>            // for OwnerLessBase
#include <sycl/detail/pi.h>                           // for PI_ERROR_INVAL...
#include <sycl/detail/property_helper.hpp>            // for PropWithDataKind
#include <sycl/detail/property_list_base.hpp>         // for PropertyListBase
#include <sycl/detail/type_list.hpp>                  // for is_contained
#include <sycl/detail/type_traits.hpp>                // for const_if_const_AS
#include <sycl/device.hpp>                            // for device
#include <sycl/exception.hpp>                         // for make_error_code
#include <sycl/ext/oneapi/accessor_property_list.hpp> // for accessor_prope...
#include <sycl/ext/oneapi/weak_object_base.hpp>       // for getSyclWeakObj...
#include <sycl/id.hpp>                                // for id
#include <sycl/image.hpp>                             // for image, image_c...
#include <sycl/multi_ptr.hpp>                         // for multi_ptr
#include <sycl/pointers.hpp>                          // for local_ptr, glo...
#include <sycl/properties/accessor_properties.hpp>    // for buffer_location
#include <sycl/properties/buffer_properties.hpp>      // for buffer, buffer...
#include <sycl/property_list.hpp>                     // for property_list
#include <sycl/range.hpp>                             // for range
#include <sycl/sampler.hpp>                           // for addressing_mode
#include <sycl/types.hpp>                             // for vec

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/detail/image_ocl_types.hpp>
#endif

#include <cstddef>     // for size_t
#include <functional>  // for hash
#include <iterator>    // for reverse_iterator
#include <limits>      // for numeric_limits
#include <memory>      // for shared_ptr
#include <optional>    // for nullopt, optional
#include <stdint.h>    // for uint32_t
#include <tuple>       // for _Swallow_assign
#include <type_traits> // for enable_if_t
#include <typeinfo>    // for type_info
#include <variant>     // for hash

/// \file accessor.hpp
/// The file contains implementations of accessor class.
///
/// Objects of accessor class define a requirement to access some SYCL memory
/// object or local memory of the device.
///
/// Basically there are 3 distinct types of accessors.
///
/// One of them is an accessor to a SYCL buffer object(Buffer accessor) which
/// has the richest interface. It supports things like accessing only a part of
/// buffer, multidimensional access using sycl::id, conversions to various
/// multi_ptr and atomic classes.
///
/// Second type is an accessor to a SYCL image object(Image accessor) which has
/// "image" specific methods for reading and writing.
///
/// Finally, accessor to local memory(Local accessor) doesn't require access to
/// any SYCL memory object, but asks for some local memory on device to be
/// available. Some methods overlap with ones that "Buffer accessor" provides.
///
/// Buffer and Image accessors create the requirement to access some SYCL memory
/// object(or part of it). SYCL RT must detect when two kernels want to access
/// the same memory objects and make sure they are executed in correct order.
///
/// "accessor_common" class that contains several common methods between Buffer
/// and Local accessors.
///
/// Accessors have different representation on host and on device. On host they
/// have non-templated base class, that is needed to safely work with any
/// accessor type. Furhermore on host we need some additional fields in order
/// to implement functionality required by Specification, for example during
/// lifetime of a host accessor other operations with memory object the accessor
/// refers to should be blocked and when all references to the host accessor are
/// desctructed, the memory this host accessor refers to should be "written
/// back".
///
/// The scheme of inheritance for host side:
///
/// \dot
/// digraph G {
///    node [shape="box"];
///    graph [splines=ortho];
///    a1 [label =
///   "accessor(1)\nFor targets:\nhost_buffer\nglobal_buffer\nconstant_buffer"];
///    a2 [label = "accessor(2)\nFor targets:\n host_image"];
///    a3 [label = "accessor(3)\nFor targets:\nlocal"];
///    a4 [label = "accessor(4)\nFor targets:\nimage"];
///    a5 [label = "accessor(5)\nFor targets:\nimage_array"];
///    "AccessorBaseHost" -> "image_accessor";
///    "AccessorBaseHost" -> a1;
///    "accessor_common" -> a1;
///    "accessor_common" -> a3;
///    "LocalAccessorBaseHost" -> a3;
///    "image_accessor" -> a2;
///    "image_accessor" -> a4;
///    "image_accessor" -> a5;
/// }
/// \enddot
///
//  +------------------+     +-----------------+     +-----------------------+
//  |                  |     |                 |     |                       |
//  | AccessorBaseHost |     | accessor_common |     | LocalAccessorBaseHost |
//  |                  |     |                 |     |                       |
//  +------------------+     +-----+-----------+     +--------+--------------+
//         |     |                     |   |                   |
//         |     +-----------+    +----+   +---------+  +------+
//         |                 |    |                  |  |
//         v                 v    v                  v  v
//  +----------------+  +-----------------+   +-------------+
//  |                |  |   accessor(1)   |   | accessor(3) |
//  | image_accessor |  +-----------------|   +-------------+
//  |                |  | for targets:    |   | for target: |
//  +---+---+---+----+  |                 |   |             |
//      |   |   |       | host_buffer     |   | local       |
//      |   |   |       | global_buffer   |   +-------------+
//      |   |   |       | constant_buffer |
//      |   |   |       +-----------------+
//      |   |   |
//      |   |   +------------------------------------+
//      |   |                                        |
//      |   +----------------------+                 |
//      v                          v                 v
//  +-----------------+    +--------------+    +-------------+
//  |     acessor(2)  |    |  accessor(4) |    | accessor(5) |
//  +-----------------+    +--------------+    +-------------+
//  | for targets:    |    | for targets: |    | for target: |
//  |                 |    |              |    |             |
//  | host_image      |    |  image       |    | image_array |
//  +-----------------+    +--------------+    +-------------+
//
/// \file accessor.hpp
///
/// For host side AccessorBaseHost/LocalAccessorBaseHost contains shared_ptr
/// which points to AccessorImplHost/LocalAccessorImplHost object.
///
/// The scheme of inheritance for device side:
/// \dot
/// digraph Diagram {
///    node [shape="box"];
///    a1 [label =
///   "accessor(1)\nFor targets:\nhost_buffer\nglobal_buffer\nconstant_buffer"];
///    a2 [label = "accessor(2)\nFor targets:\nhost_image"];
///    a3 [label = "accessor(3)\nFor targets:\nlocal"];
///    a4 [label = "accessor(4)\nFor targets:\nimage"];
///    a5 [label = "accessor(5)\nFor targets:\nimage_array"];
///    "accessor_common" -> a1;
///    "accessor_common" -> a3;
///    "image_accessor" -> a2;
///    "image_accessor" -> a4;
///    "image_accessor" -> a5;
///    a1 -> "host_accessor";
/// }
/// \enddot
///
//
//                            +-----------------+
//                            |                 |
//                            | accessor_common |
//                            |                 |
//                            +-----+-------+---+
//                                     |       |
//                                +----+       +-----+
//                                |                  |
//                                v                  v
//  +----------------+  +-----------------+   +-------------+
//  |                |  |   accessor(1)   |   | accessor(3) |
//  | image_accessor |  +-----------------|   +-------------+
//  |                |  | for targets:    |   | for target: |
//  +---+---+---+----+  |                 |   |             |
//      |   |   |       | host_buffer     |   | local       |
//      |   |   |       | global_buffer   |   +-------------+
//      |   |   |       | constant_buffer |
//      |   |   |       +-----------------+
//      |   |   |                 |
//      |   |   |                 v
//      |   |   |       +-----------------+
//      |   |   |       |                 |
//      |   |   |       |  host_accessor  |
//      |   |   |       |                 |
//      |   |   |       +-----------------+
//      |   |   |
//      |   |   +------------------------------------+
//      |   |                                        |
//      |   +----------------------+                 |
//      v                          v                 v
//  +-----------------+    +--------------+    +-------------+
//  |     acessor(2)  |    |  accessor(4) |    | accessor(5) |
//  +-----------------+    +--------------+    +-------------+
//  | for targets:    |    | for targets: |    | for target: |
//  |                 |    |              |    |             |
//  | host_image      |    |  image       |    | image_array |
//  +-----------------+    +--------------+    +-------------+
//
/// \file accessor.hpp
///
/// For device side AccessorImplHost/LocalAccessorImplHost are fileds of
/// accessor(1) and accessor(3).
///
/// accessor(1) declares accessor as a template class and implements accessor
/// class for access targets: host_buffer, global_buffer and constant_buffer.
///
/// accessor(3) specializes accessor(1) for the local access target.
///
/// image_accessor contains implements interfaces for access targets:
/// host_image, image and image_array. But there are three distinct
/// specializations of the accessor(1) (accessor(2), accessor(4), accessor(5))
/// that are just inherited from image_accessor.
///
/// accessor_common contains several helpers common for both accessor(1) and
/// accessor(3)

namespace sycl {
inline namespace _V1 {
class stream;
namespace ext::intel::esimd::detail {
// Forward declare a "back-door" access class to support ESIMD.
class AccessorPrivateProxy;
} // namespace ext::intel::esimd::detail

template <typename DataT, int Dimensions = 1,
          access::mode AccessMode = access::mode::read_write,
          access::target AccessTarget = access::target::device,
          access::placeholder IsPlaceholder = access::placeholder::false_t,
          typename PropertyListT = ext::oneapi::accessor_property_list<>>
class accessor;

namespace detail {

// A helper structure which is shared between buffer accessor and accessor_impl
// TODO: Unify with AccessorImplDevice?
struct AccHostDataT {
  AccHostDataT(const sycl::id<3> &Offset, const sycl::range<3> &Range,
               const sycl::range<3> &MemoryRange, void *Data = nullptr)
      : MOffset(Offset), MAccessRange(Range), MMemoryRange(MemoryRange),
        MData(Data) {}

  sycl::id<3> MOffset;
  sycl::range<3> MAccessRange;
  sycl::range<3> MMemoryRange;
  void *MData = nullptr;
  void *Reserved = nullptr;
};

void __SYCL_EXPORT constructorNotification(void *BufferObj, void *AccessorObj,
                                           access::target Target,
                                           access::mode Mode,
                                           const code_location &CodeLoc);

void __SYCL_EXPORT unsampledImageConstructorNotification(
    void *ImageObj, void *AccessorObj,
    const std::optional<image_target> &Target, access::mode Mode,
    const void *Type, uint32_t ElemSize, const code_location &CodeLoc);

void __SYCL_EXPORT sampledImageConstructorNotification(
    void *ImageObj, void *AccessorObj,
    const std::optional<image_target> &Target, const void *Type,
    uint32_t ElemSize, const code_location &CodeLoc);

template <typename T>
using IsPropertyListT = typename std::is_base_of<PropertyListBase, T>;

template <typename T>
using IsRunTimePropertyListT =
    typename std::is_same<ext::oneapi::accessor_property_list<>, T>;

template <typename T> struct IsCxPropertyList {
  constexpr static bool value = false;
};

template <typename... Props>
struct IsCxPropertyList<ext::oneapi::accessor_property_list<Props...>> {
  constexpr static bool value = true;
};

template <> struct IsCxPropertyList<ext::oneapi::accessor_property_list<>> {
  constexpr static bool value = false;
};

// Zero-dimensional accessors references at-most a single element, so the range
// is either 0 if the associated buffer is empty or 1 otherwise.
template <typename BufferT>
sycl::range<1> GetZeroDimAccessRange(BufferT Buffer) {
  return std::min(Buffer.size(), size_t{1});
}

__SYCL_EXPORT device getDeviceFromHandler(handler &CommandGroupHandlerRef);

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT = ext::oneapi::accessor_property_list<>>
class accessor_common {
protected:
  constexpr static access::address_space AS = TargetToAS<AccessTarget>::AS;

  constexpr static bool IsHostBuf = AccessTarget == access::target::host_buffer;
  constexpr static bool IsHostTask = AccessTarget == access::target::host_task;
  // SYCL2020 4.7.6.9.4.3
  // IsPlaceHolder template parameter has no bearing on whether the accessor
  // instance is a placeholder. This is determined solely by the constructor.
  // The rule seems to be: if the constructor receives a CommandGroupHandler
  // it is NOT a placeholder. Otherwise, it is a placeholder.
  // However, according to 4.7.6.9.4.6. accessor specialization with
  // target::host_buffer is never a placeholder. So, if the constructor
  // used receives a CommandGroupHandler, the accessor will never be a
  // placeholder. If it doesn't, but IsHostBuf is true, it won't be a
  // placeholder either. Otherwise, the accessor is a placeholder.
  constexpr static bool IsPlaceH = !IsHostBuf;

  // TODO: SYCL 2020 deprecates four of the target enum values
  // and replaces them with 2 (device and host_task). May want
  // to change these constexpr.
  constexpr static bool IsGlobalBuf =
      AccessTarget == access::target::global_buffer;

  constexpr static bool IsConstantBuf =
      AccessTarget == access::target::constant_buffer;

  constexpr static bool IsAccessAnyWrite =
      AccessMode == access::mode::write ||
      AccessMode == access::mode::read_write ||
      AccessMode == access::mode::discard_write ||
      AccessMode == access::mode::discard_read_write;

  constexpr static bool IsAccessReadOnly = AccessMode == access::mode::read;
  static constexpr bool IsConst = std::is_const_v<DataT>;

  constexpr static bool IsAccessReadWrite =
      AccessMode == access::mode::read_write;

  constexpr static bool IsAccessAtomic = AccessMode == access::mode::atomic;

  using RefType = detail::const_if_const_AS<AS, DataT> &;
  using ConstRefType = const DataT &;
  using PtrType = detail::const_if_const_AS<AS, DataT> *;

  // The class which allows to access value of N dimensional accessor using N
  // subscript operators, e.g. accessor[2][2][3]
  template <int SubDims,
            typename AccType =
                accessor<DataT, Dimensions, AccessMode, AccessTarget,
                         IsPlaceholder, PropertyListT>>
  class AccessorSubscript {
    static constexpr int Dims = Dimensions;

    mutable id<Dims> MIDs;
    AccType MAccessor;

  public:
    AccessorSubscript(AccType Accessor, id<Dims> IDs)
        : MIDs(IDs), MAccessor(Accessor) {}

    // Only accessor class is supposed to use this c'tor for the first
    // operator[].
    AccessorSubscript(AccType Accessor, size_t Index) : MAccessor(Accessor) {
      MIDs[0] = Index;
    }

    template <int CurDims = SubDims, typename = std::enable_if_t<(CurDims > 1)>>
    auto operator[](size_t Index) {
      MIDs[Dims - CurDims] = Index;
      return AccessorSubscript<CurDims - 1, AccType>(MAccessor, MIDs);
    }

    template <int CurDims = SubDims,
              typename = std::enable_if_t<CurDims == 1 && (IsAccessReadOnly ||
                                                           IsAccessAnyWrite)>>
    typename AccType::reference operator[](size_t Index) const {
      MIDs[Dims - CurDims] = Index;
      return MAccessor[MIDs];
    }

    template <int CurDims = SubDims>
    typename std::enable_if_t<CurDims == 1 && IsAccessAtomic, atomic<DataT, AS>>
    operator[](size_t Index) const {
      MIDs[Dims - CurDims] = Index;
      return MAccessor[MIDs];
    }
  };
};

template <typename DataT> constexpr access::mode accessModeFromConstness() {
  if constexpr (std::is_const_v<DataT>)
    return access::mode::read;
  else
    return access::mode::read_write;
}

template <typename MayBeTag1, typename MayBeTag2>
constexpr access::mode deduceAccessMode() {
  // property_list = {} is not properly detected by deduction guide,
  // when parameter is passed without curly braces: access(buffer, no_init)
  // thus simplest approach is to check 2 last arguments for being a tag
  if constexpr (std::is_same_v<MayBeTag1, mode_tag_t<access::mode::read>> ||
                std::is_same_v<MayBeTag2, mode_tag_t<access::mode::read>>) {
    return access::mode::read;
  }

  if constexpr (std::is_same_v<MayBeTag1, mode_tag_t<access::mode::write>> ||
                std::is_same_v<MayBeTag2, mode_tag_t<access::mode::write>>) {
    return access::mode::write;
  }

  if constexpr (std::is_same_v<
                    MayBeTag1,
                    mode_target_tag_t<access::mode::read,
                                      access::target::constant_buffer>> ||
                std::is_same_v<
                    MayBeTag2,
                    mode_target_tag_t<access::mode::read,
                                      access::target::constant_buffer>>) {
    return access::mode::read;
  }

  if constexpr (std::is_same_v<MayBeTag1,
                               mode_target_tag_t<access::mode::read,
                                                 access::target::host_task>> ||
                std::is_same_v<MayBeTag2,
                               mode_target_tag_t<access::mode::read,
                                                 access::target::host_task>>) {
    return access::mode::read;
  }

  if constexpr (std::is_same_v<MayBeTag1,
                               mode_target_tag_t<access::mode::write,
                                                 access::target::host_task>> ||
                std::is_same_v<MayBeTag2,
                               mode_target_tag_t<access::mode::write,
                                                 access::target::host_task>>) {
    return access::mode::write;
  }

  return access::mode::read_write;
}

template <typename MayBeTag1, typename MayBeTag2>
constexpr access::target deduceAccessTarget(access::target defaultTarget) {
  if constexpr (std::is_same_v<
                    MayBeTag1,
                    mode_target_tag_t<access::mode::read,
                                      access::target::constant_buffer>> ||
                std::is_same_v<
                    MayBeTag2,
                    mode_target_tag_t<access::mode::read,
                                      access::target::constant_buffer>>) {
    return access::target::constant_buffer;
  }

  if constexpr (
      std::is_same_v<MayBeTag1, mode_target_tag_t<access::mode::read,
                                                  access::target::host_task>> ||
      std::is_same_v<MayBeTag2, mode_target_tag_t<access::mode::read,
                                                  access::target::host_task>> ||
      std::is_same_v<MayBeTag1, mode_target_tag_t<access::mode::write,
                                                  access::target::host_task>> ||
      std::is_same_v<MayBeTag2, mode_target_tag_t<access::mode::write,
                                                  access::target::host_task>> ||
      std::is_same_v<MayBeTag1, mode_target_tag_t<access::mode::read_write,
                                                  access::target::host_task>> ||
      std::is_same_v<MayBeTag2, mode_target_tag_t<access::mode::read_write,
                                                  access::target::host_task>>) {
    return access::target::host_task;
  }

  return defaultTarget;
}

template <int Dims> class LocalAccessorBaseDevice {
public:
  LocalAccessorBaseDevice(sycl::range<Dims> Size)
      : AccessRange(Size),
        MemRange(InitializedVal<Dims, range>::template get<0>()) {}
  // TODO: Actually we need only one field here, but currently compiler requires
  // all of them.
  range<Dims> AccessRange;
  range<Dims> MemRange;
  id<Dims> Offset;

  bool operator==(const LocalAccessorBaseDevice &Rhs) const {
    return (AccessRange == Rhs.AccessRange);
  }
};

// The class describes a requirement to access a SYCL memory object such as
// sycl::buffer and sycl::image. For example, each accessor used in a kernel,
// except one with access target "local", adds such requirement for the command
// group.

template <int Dims> class AccessorImplDevice {
public:
  AccessorImplDevice() = default;
  AccessorImplDevice(id<Dims> Offset, range<Dims> AccessRange,
                     range<Dims> MemoryRange)
      : Offset(Offset), AccessRange(AccessRange), MemRange(MemoryRange) {}

  id<Dims> Offset;
  range<Dims> AccessRange;
  range<Dims> MemRange;

  bool operator==(const AccessorImplDevice &Rhs) const {
    return (Offset == Rhs.Offset && AccessRange == Rhs.AccessRange &&
            MemRange == Rhs.MemRange);
  }
};

class AccessorImplHost;

void __SYCL_EXPORT addHostAccessorAndWait(AccessorImplHost *Req);

class SYCLMemObjI;

using AccessorImplPtr = std::shared_ptr<AccessorImplHost>;

class __SYCL_EXPORT AccessorBaseHost {
protected:
  AccessorBaseHost(const AccessorImplPtr &Impl) : impl{Impl} {}

public:
  // TODO: the following function to be removed during next ABI break window
  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {});
  // TODO: the following function to be removed during next ABI break window
  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, bool IsPlaceH, int OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {});

  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, size_t OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {});

  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, bool IsPlaceH, size_t OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {});

public:
  id<3> &getOffset();
  range<3> &getAccessRange();
  range<3> &getMemoryRange();
  void *getPtr() noexcept;
  unsigned int getElemSize() const;

  const id<3> &getOffset() const;
  const range<3> &getAccessRange() const;
  const range<3> &getMemoryRange() const;
  void *getPtr() const noexcept;
  bool isPlaceholder() const;
  bool isMemoryObjectUsedByGraph() const;

  detail::AccHostDataT &getAccData();

  const property_list &getPropList() const;

  void *getMemoryObject() const;

  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <typename, int, access::mode, access::target, access::placeholder,
            typename>
  friend class accessor;

  AccessorImplPtr impl;

private:
  friend class sycl::ext::intel::esimd::detail::AccessorPrivateProxy;
};

class LocalAccessorImplHost;
using LocalAccessorImplPtr = std::shared_ptr<LocalAccessorImplHost>;

class __SYCL_EXPORT LocalAccessorBaseHost {
protected:
  LocalAccessorBaseHost(const LocalAccessorImplPtr &Impl) : impl{Impl} {}

public:
  LocalAccessorBaseHost(sycl::range<3> Size, int Dims, int ElemSize,
                        const property_list &PropertyList = {});
  sycl::range<3> &getSize();
  const sycl::range<3> &getSize() const;
  void *getPtr();
  void *getPtr() const;
  int getNumOfDims();
  int getElementSize();
  const property_list &getPropList() const;

protected:
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  LocalAccessorImplPtr impl;
};

class UnsampledImageAccessorImplHost;
class SampledImageAccessorImplHost;
using UnsampledImageAccessorImplPtr =
    std::shared_ptr<UnsampledImageAccessorImplHost>;
using SampledImageAccessorImplPtr =
    std::shared_ptr<SampledImageAccessorImplHost>;

void __SYCL_EXPORT
addHostUnsampledImageAccessorAndWait(UnsampledImageAccessorImplHost *Req);
void __SYCL_EXPORT
addHostSampledImageAccessorAndWait(SampledImageAccessorImplHost *Req);

class __SYCL_EXPORT UnsampledImageAccessorBaseHost {
protected:
  UnsampledImageAccessorBaseHost(const UnsampledImageAccessorImplPtr &Impl)
      : impl{Impl} {}

public:
  UnsampledImageAccessorBaseHost(sycl::range<3> Size, access_mode AccessMode,
                                 void *SYCLMemObject, int Dims, int ElemSize,
                                 id<3> Pitch, image_channel_type ChannelType,
                                 image_channel_order ChannelOrder,
                                 const property_list &PropertyList = {});
  const sycl::range<3> &getSize() const;
  void *getMemoryObject() const;
  detail::AccHostDataT &getAccData();
  void *getPtr();
  void *getPtr() const;
  int getNumOfDims() const;
  int getElementSize() const;
  id<3> getPitch() const;
  image_channel_type getChannelType() const;
  image_channel_order getChannelOrder() const;
  const property_list &getPropList() const;

protected:
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  UnsampledImageAccessorImplPtr impl;

  // The function references helper methods required by GDB pretty-printers
  void GDBMethodsAnchor() {
#ifndef NDEBUG
    const auto *this_const = this;
    (void)getSize();
    (void)this_const->getSize();
    (void)getPtr();
    (void)this_const->getPtr();
#endif
  }

#ifndef __SYCL_DEVICE_ONLY__
  // Reads a pixel of the underlying image at the specified coordinate. It is
  // the responsibility of the caller to ensure that the coordinate type is
  // valid.
  template <typename DataT, typename CoordT>
  DataT read(const CoordT &Coords) const noexcept {
    image_sampler Smpl{addressing_mode::none,
                       coordinate_normalization_mode::unnormalized,
                       filtering_mode::nearest};
    return imageReadSamplerHostImpl<CoordT, DataT>(
        Coords, Smpl, getSize(), getPitch(), getChannelType(),
        getChannelOrder(), getPtr(), getElementSize());
  }

  // Writes to a pixel of the underlying image at the specified coordinate. It
  // is the responsibility of the caller to ensure that the coordinate type is
  // valid.
  template <typename DataT, typename CoordT>
  void write(const CoordT &Coords, const DataT &Color) const {
    imageWriteHostImpl(Coords, Color, getPitch(), getElementSize(),
                       getChannelType(), getChannelOrder(), getPtr());
  }
#endif
};

class __SYCL_EXPORT SampledImageAccessorBaseHost {
protected:
  SampledImageAccessorBaseHost(const SampledImageAccessorImplPtr &Impl)
      : impl{Impl} {}

public:
  SampledImageAccessorBaseHost(sycl::range<3> Size, void *SYCLMemObject,
                               int Dims, int ElemSize, id<3> Pitch,
                               image_channel_type ChannelType,
                               image_channel_order ChannelOrder,
                               image_sampler Sampler,
                               const property_list &PropertyList = {});
  const sycl::range<3> &getSize() const;
  void *getMemoryObject() const;
  detail::AccHostDataT &getAccData();
  void *getPtr();
  void *getPtr() const;
  int getNumOfDims() const;
  int getElementSize() const;
  id<3> getPitch() const;
  image_channel_type getChannelType() const;
  image_channel_order getChannelOrder() const;
  image_sampler getSampler() const;
  const property_list &getPropList() const;

protected:
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  SampledImageAccessorImplPtr impl;

  // The function references helper methods required by GDB pretty-printers
  void GDBMethodsAnchor() {
#ifndef NDEBUG
    const auto *this_const = this;
    (void)getSize();
    (void)this_const->getSize();
    (void)getPtr();
    (void)this_const->getPtr();
#endif
  }

#ifndef __SYCL_DEVICE_ONLY__
  // Reads a pixel of the underlying image at the specified coordinate. It is
  // the responsibility of the caller to ensure that the coordinate type is
  // valid.
  template <typename DataT, typename CoordT>
  DataT read(const CoordT &Coords) const {
    return imageReadSamplerHostImpl<CoordT, DataT>(
        Coords, getSampler(), getSize(), getPitch(), getChannelType(),
        getChannelOrder(), getPtr(), getElementSize());
  }
#endif
};

template <int Dim, typename T> struct IsValidCoordDataT;
template <typename T> struct IsValidCoordDataT<1, T> {
  constexpr static bool value = detail::is_contained<
      T, detail::type_list<opencl::cl_int, opencl::cl_float>>::type::value;
};
template <typename T> struct IsValidCoordDataT<2, T> {
  constexpr static bool value = detail::is_contained<
      T, detail::type_list<vec<opencl::cl_int, 2>,
                           vec<opencl::cl_float, 2>>>::type::value;
};
template <typename T> struct IsValidCoordDataT<3, T> {
  constexpr static bool value = detail::is_contained<
      T, detail::type_list<vec<opencl::cl_int, 4>,
                           vec<opencl::cl_float, 4>>>::type::value;
};

template <int Dim, typename T> struct IsValidUnsampledCoord2020DataT;
template <typename T> struct IsValidUnsampledCoord2020DataT<1, T> {
  constexpr static bool value = std::is_same_v<T, int>;
};
template <typename T> struct IsValidUnsampledCoord2020DataT<2, T> {
  constexpr static bool value = std::is_same_v<T, int2>;
};
template <typename T> struct IsValidUnsampledCoord2020DataT<3, T> {
  constexpr static bool value = std::is_same_v<T, int4>;
};

template <int Dim, typename T> struct IsValidSampledCoord2020DataT;
template <typename T> struct IsValidSampledCoord2020DataT<1, T> {
  constexpr static bool value = std::is_same_v<T, float>;
};
template <typename T> struct IsValidSampledCoord2020DataT<2, T> {
  constexpr static bool value = std::is_same_v<T, float2>;
};
template <typename T> struct IsValidSampledCoord2020DataT<3, T> {
  constexpr static bool value = std::is_same_v<T, float4>;
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __image_array_slice__;

// Image accessor
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class image_accessor
#ifndef __SYCL_DEVICE_ONLY__
    : public detail::AccessorBaseHost {
  size_t MImageCount;
  image_channel_order MImgChannelOrder;
  image_channel_type MImgChannelType;
#else
{

  using OCLImageTy = typename detail::opencl_image_type<Dimensions, AccessMode,
                                                        AccessTarget>::type;
  OCLImageTy MImageObj;
  char MPadding[sizeof(detail::AccessorBaseHost) +
                sizeof(size_t /*MImageCount*/) + sizeof(image_channel_order) +
                sizeof(image_channel_type) - sizeof(OCLImageTy)];

protected:
  void imageAccessorInit(OCLImageTy Image) { MImageObj = Image; }

private:
#endif
  template <typename T1, int T2, access::mode T3, access::placeholder T4>
  friend class __image_array_slice__;

  constexpr static bool IsHostImageAcc =
      (AccessTarget == access::target::host_image);

  constexpr static bool IsImageAcc = (AccessTarget == access::target::image);

  constexpr static bool IsImageArrayAcc =
      (AccessTarget == access::target::image_array);

  constexpr static bool IsImageAccessWriteOnly =
      (AccessMode == access::mode::write ||
       AccessMode == access::mode::discard_write);

  constexpr static bool IsImageAccessAnyWrite =
      (IsImageAccessWriteOnly || AccessMode == access::mode::read_write);

  constexpr static bool IsImageAccessReadOnly =
      (AccessMode == access::mode::read);

  constexpr static bool IsImageAccessAnyRead =
      (IsImageAccessReadOnly || AccessMode == access::mode::read_write);

  static_assert(std::is_same_v<DataT, vec<opencl::cl_int, 4>> ||
                    std::is_same_v<DataT, vec<opencl::cl_uint, 4>> ||
                    std::is_same_v<DataT, vec<opencl::cl_float, 4>> ||
                    std::is_same_v<DataT, vec<opencl::cl_half, 4>>,
                "The data type of an image accessor must be only cl_int4, "
                "cl_uint4, cl_float4 or cl_half4 from SYCL namespace");

  static_assert(IsImageAcc || IsHostImageAcc || IsImageArrayAcc,
                "Expected image type");

  static_assert(IsPlaceholder == access::placeholder::false_t,
                "Expected false as Placeholder value for image accessor.");

  static_assert(
      ((IsImageAcc || IsImageArrayAcc) &&
       (IsImageAccessWriteOnly || IsImageAccessReadOnly)) ||
          (IsHostImageAcc && (IsImageAccessAnyWrite || IsImageAccessAnyRead)),
      "Access modes can be only read/write/discard_write for image/image_array "
      "target accessor, or they can be only "
      "read/write/discard_write/read_write for host_image target accessor.");

  static_assert(Dimensions > 0 && Dimensions <= 3,
                "Dimensions can be 1/2/3 for image accessor.");

#ifdef __SYCL_DEVICE_ONLY__

  sycl::vec<int, Dimensions> getRangeInternal() const {
    return __invoke_ImageQuerySize<sycl::vec<int, Dimensions>, OCLImageTy>(
        MImageObj);
  }

  size_t getElementSize() const {
    int ChannelType = __invoke_ImageQueryFormat<int, OCLImageTy>(MImageObj);
    int ChannelOrder = __invoke_ImageQueryOrder<int, OCLImageTy>(MImageObj);
    int ElementSize = getSPIRVElementSize(ChannelType, ChannelOrder);
    return ElementSize;
  }

#else

  sycl::vec<int, Dimensions> getRangeInternal() const {
    // TODO: Implement for host.
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "image::getRangeInternal() is not implemented for host");
    return sycl::vec<int, Dimensions>{1};
  }

#endif

#ifndef __SYCL_DEVICE_ONLY__
protected:
  image_accessor(const AccessorImplPtr &Impl) : AccessorBaseHost{Impl} {}
#endif // __SYCL_DEVICE_ONLY__

private:
  friend class sycl::ext::intel::esimd::detail::AccessorPrivateProxy;

#ifdef __SYCL_DEVICE_ONLY__
  const OCLImageTy getNativeImageObj() const { return MImageObj; }
#endif // __SYCL_DEVICE_ONLY__

public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  // image_accessor Constructors.

#ifdef __SYCL_DEVICE_ONLY__
  // Default constructor for objects later initialized with __init member.
  image_accessor() {}
#endif

  // Available only when: accessTarget == access::target::host_image
  // template <typename AllocatorT>
  // accessor(image<dimensions, AllocatorT> &imageRef);
  template <
      typename AllocatorT, int Dims = Dimensions,
      typename = std::enable_if_t<(Dims > 0 && Dims <= 3) && IsHostImageAcc>>
  image_accessor(image<Dims, AllocatorT> &ImageRef, int ImageElementSize)
#ifdef __SYCL_DEVICE_ONLY__
  {
    (void)ImageRef;
    (void)ImageElementSize;
    // No implementation needed for device. The constructor is only called by
    // host.
  }
#else
      : AccessorBaseHost({ImageRef.getRowPitch(), ImageRef.getSlicePitch(), 0},
                         detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                         detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                         AccessMode, detail::getSyclObjImpl(ImageRef).get(),
                         Dimensions, ImageElementSize, size_t(0)),
        MImageCount(ImageRef.size()),
        MImgChannelOrder(ImageRef.getChannelOrder()),
        MImgChannelType(ImageRef.getChannelType()) {
    addHostAccessorAndWait(AccessorBaseHost::impl.get());
  }
#endif

  // Available only when: accessTarget == access::target::image
  // template <typename AllocatorT>
  // accessor(image<dimensions, AllocatorT> &imageRef,
  //          handler &commandGroupHandlerRef);
  template <typename AllocatorT, int Dims = Dimensions,
            typename = std::enable_if_t<(Dims > 0 && Dims <= 3) && IsImageAcc>>
  image_accessor(image<Dims, AllocatorT> &ImageRef,
                 handler &CommandGroupHandlerRef, int ImageElementSize)
#ifdef __SYCL_DEVICE_ONLY__
  {
    (void)ImageRef;
    (void)CommandGroupHandlerRef;
    (void)ImageElementSize;
    // No implementation needed for device. The constructor is only called by
    // host.
  }
#else
      : AccessorBaseHost({ImageRef.getRowPitch(), ImageRef.getSlicePitch(), 0},
                         detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                         detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                         AccessMode, detail::getSyclObjImpl(ImageRef).get(),
                         Dimensions, ImageElementSize, size_t(0)),
        MImageCount(ImageRef.size()),
        MImgChannelOrder(ImageRef.getChannelOrder()),
        MImgChannelType(ImageRef.getChannelType()) {

    device Device = getDeviceFromHandler(CommandGroupHandlerRef);
    if (!Device.has(aspect::ext_intel_legacy_image))
      throw sycl::exception(
          sycl::errc::feature_not_supported,
          "SYCL 1.2.1 images are not supported by this device.");
  }
#endif

  /* -- common interface members -- */

  // operator == and != need to be defined only for host application as per the
  // SYCL spec 1.2.1
#ifndef __SYCL_DEVICE_ONLY__
  bool operator==(const image_accessor &Rhs) const { return Rhs.impl == impl; }
#else
  // The operator with __SYCL_DEVICE_ONLY__ need to be declared for compilation
  // of host application with device compiler.
  // Usage of this operator inside the kernel code will give a runtime failure.
  bool operator==(const image_accessor &Rhs) const;
#endif

  bool operator!=(const image_accessor &Rhs) const { return !(Rhs == *this); }

  // get_count() method : Returns the number of elements of the SYCL image this
  // SYCL accessor is accessing.
  //
  // get_range() method :  Returns a range object which represents the number of
  // elements of dataT per dimension that this accessor may access.
  // The range object returned must equal to the range of the image this
  // accessor is associated with.

#ifdef __SYCL_DEVICE_ONLY__

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return get_range<Dimensions>().size(); }

  template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 1>>
  range<1> get_range() const {
    int Range = getRangeInternal();
    return range<1>(Range);
  }
  template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 2>>
  range<2> get_range() const {
    int2 Range = getRangeInternal();
    return range<2>(Range[0], Range[1]);
  }
  template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 3>>
  range<3> get_range() const {
    int3 Range = getRangeInternal();
    return range<3>(Range[0], Range[1], Range[2]);
  }

#else
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); };
  size_t size() const noexcept { return MImageCount; };

  template <int Dims = Dimensions, typename = std::enable_if_t<(Dims > 0)>>
  range<Dims> get_range() const {
    return detail::convertToArrayOfN<Dims, 1>(getAccessRange());
  }

#endif

  // Available only when:
  // (accessTarget == access::target::image && accessMode == access::mode::read)
  // || (accessTarget == access::target::host_image && ( accessMode ==
  // access::mode::read || accessMode == access::mode::read_write))
  template <typename CoordT, int Dims = Dimensions,
            typename = std::enable_if_t<
                (Dims > 0) && (IsValidCoordDataT<Dims, CoordT>::value) &&
                (detail::is_genint_v<CoordT>)&&(
                    (IsImageAcc && IsImageAccessReadOnly) ||
                    (IsHostImageAcc && IsImageAccessAnyRead))>>
  DataT read(const CoordT &Coords) const {
#ifdef __SYCL_DEVICE_ONLY__
    return __invoke__ImageRead<DataT, OCLImageTy, CoordT>(MImageObj, Coords);
#else
    sampler Smpl(coordinate_normalization_mode::unnormalized,
                 addressing_mode::none, filtering_mode::nearest);
    return read<CoordT, Dims>(Coords, Smpl);
#endif
  }

  // Available only when:
  // (accessTarget == access::target::image && accessMode == access::mode::read)
  // || (accessTarget == access::target::host_image && ( accessMode ==
  // access::mode::read || accessMode == access::mode::read_write))
  template <typename CoordT, int Dims = Dimensions,
            typename = std::enable_if_t<
                (Dims > 0) && (IsValidCoordDataT<Dims, CoordT>::value) &&
                ((IsImageAcc && IsImageAccessReadOnly) ||
                 (IsHostImageAcc && IsImageAccessAnyRead))>>
  DataT read(const CoordT &Coords, const sampler &Smpl) const {
#ifdef __SYCL_DEVICE_ONLY__
    return __invoke__ImageReadSampler<DataT, OCLImageTy, CoordT>(
        MImageObj, Coords, Smpl.impl.m_Sampler);
#else
    return imageReadSamplerHostImpl<CoordT, DataT>(
        Coords, Smpl, getAccessRange() /*Image Range*/,
        getOffset() /*Image Pitch*/, MImgChannelType, MImgChannelOrder,
        AccessorBaseHost::getPtr() /*ptr to image*/,
        AccessorBaseHost::getElemSize());
#endif
  }

  // Available only when:
  // (accessTarget == access::target::image && (accessMode ==
  // access::mode::write || accessMode == access::mode::discard_write)) ||
  // (accessTarget == access::target::host_image && (accessMode ==
  // access::mode::write || accessMode == access::mode::discard_write ||
  // accessMode == access::mode::read_write))
  template <
      typename CoordT, int Dims = Dimensions,
      typename = std::enable_if_t<(Dims > 0) &&
                                  (detail::is_genint_v<CoordT>)&&(
                                      IsValidCoordDataT<Dims, CoordT>::value) &&
                                  ((IsImageAcc && IsImageAccessWriteOnly) ||
                                   (IsHostImageAcc && IsImageAccessAnyWrite))>>
  void write(const CoordT &Coords, const DataT &Color) const {
#ifdef __SYCL_DEVICE_ONLY__
    __invoke__ImageWrite<OCLImageTy, CoordT, DataT>(MImageObj, Coords, Color);
#else
    imageWriteHostImpl(Coords, Color, getOffset() /*ImagePitch*/,
                       AccessorBaseHost::getElemSize(), MImgChannelType,
                       MImgChannelOrder,
                       AccessorBaseHost::getPtr() /*Ptr to Image*/);
#endif
  }
};

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __image_array_slice__ {

  static_assert(Dimensions < 3,
                "Image slice cannot have more then 2 dimensions");

  constexpr static int AdjustedDims = (Dimensions == 2) ? 4 : Dimensions + 1;

  template <typename CoordT,
            typename CoordElemType =
                typename detail::TryToGetElementType<CoordT>::type>
  sycl::vec<CoordElemType, AdjustedDims>
  getAdjustedCoords(const CoordT &Coords) const {
    CoordElemType LastCoord = 0;

    if (std::is_same<float, CoordElemType>::value) {
      sycl::vec<int, Dimensions + 1> Size = MBaseAcc.getRangeInternal();
      LastCoord =
          MIdx / static_cast<float>(Size.template swizzle<Dimensions>());
    } else {
      LastCoord = MIdx;
    }

    sycl::vec<CoordElemType, Dimensions> LeftoverCoords{LastCoord};
    sycl::vec<CoordElemType, AdjustedDims> AdjustedCoords{Coords,
                                                          LeftoverCoords};
    return AdjustedCoords;
  }

public:
  __image_array_slice__(
      accessor<DataT, Dimensions, AccessMode, access::target::image_array,
               IsPlaceholder, ext::oneapi::accessor_property_list<>>
          BaseAcc,
      size_t Idx)
      : MBaseAcc(BaseAcc), MIdx(Idx) {}

  template <typename CoordT, int Dims = Dimensions,
            typename = std::enable_if_t<
                (Dims > 0) && (IsValidCoordDataT<Dims, CoordT>::value)>>
  DataT read(const CoordT &Coords) const {
    return MBaseAcc.read(getAdjustedCoords(Coords));
  }

  template <typename CoordT, int Dims = Dimensions,
            typename = std::enable_if_t<(Dims > 0) &&
                                        IsValidCoordDataT<Dims, CoordT>::value>>
  DataT read(const CoordT &Coords, const sampler &Smpl) const {
    return MBaseAcc.read(getAdjustedCoords(Coords), Smpl);
  }

  template <typename CoordT, int Dims = Dimensions,
            typename = std::enable_if_t<(Dims > 0) &&
                                        IsValidCoordDataT<Dims, CoordT>::value>>
  void write(const CoordT &Coords, const DataT &Color) const {
    return MBaseAcc.write(getAdjustedCoords(Coords), Color);
  }

#ifdef __SYCL_DEVICE_ONLY__
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return get_range<Dimensions>().size(); }

  template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 1>>
  range<1> get_range() const {
    int2 Count = MBaseAcc.getRangeInternal();
    return range<1>(Count.x());
  }
  template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 2>>
  range<2> get_range() const {
    int3 Count = MBaseAcc.getRangeInternal();
    return range<2>(Count.x(), Count.y());
  }

#else

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept {
    return MBaseAcc.MImageCount / MBaseAcc.getAccessRange()[Dimensions];
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<(Dims == 1 || Dims == 2)>>
  range<Dims> get_range() const {
    return detail::convertToArrayOfN<Dims, 1>(MBaseAcc.getAccessRange());
  }

#endif

private:
  size_t MIdx;
  accessor<DataT, Dimensions, AccessMode, access::target::image_array,
           IsPlaceholder, ext::oneapi::accessor_property_list<>>
      MBaseAcc;
};

} // namespace detail

/// Buffer accessor.
///
/// \sa buffer
///
/// \ingroup sycl_api_acc
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class __SYCL_EBO __SYCL_SPECIAL_CLASS __SYCL_TYPE(accessor) accessor :
#ifndef __SYCL_DEVICE_ONLY__
    public detail::AccessorBaseHost,
#endif
    public detail::accessor_common<DataT, Dimensions, AccessMode, AccessTarget,
                                   IsPlaceholder, PropertyListT>,
    public detail::OwnerLessBase<
        accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder,
                 PropertyListT>> {
protected:
  static_assert((AccessTarget == access::target::global_buffer ||
                 AccessTarget == access::target::constant_buffer ||
                 AccessTarget == access::target::host_buffer ||
                 AccessTarget == access::target::host_task),
                "Expected buffer type");

  static_assert((AccessTarget == access::target::global_buffer ||
                 AccessTarget == access::target::host_buffer ||
                 AccessTarget == access::target::host_task) ||
                    (AccessTarget == access::target::constant_buffer &&
                     AccessMode == access::mode::read),
                "Access mode can be only read for constant buffers");

  static_assert(detail::IsPropertyListT<PropertyListT>::value,
                "PropertyListT must be accessor_property_list");

  using AccessorCommonT =
      detail::accessor_common<DataT, Dimensions, AccessMode, AccessTarget,
                              IsPlaceholder, PropertyListT>;

  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;

  using AccessorCommonT::AS;
  // Cannot do "using AccessorCommonT::Flag" as it doesn't work with g++ as host
  // compiler, for some reason.
  static constexpr bool IsAccessAnyWrite = AccessorCommonT::IsAccessAnyWrite;
  static constexpr bool IsAccessReadOnly = AccessorCommonT::IsAccessReadOnly;
  static constexpr bool IsConstantBuf = AccessorCommonT::IsConstantBuf;
  static constexpr bool IsGlobalBuf = AccessorCommonT::IsGlobalBuf;
  static constexpr bool IsHostBuf = AccessorCommonT::IsHostBuf;
  static constexpr bool IsPlaceH = AccessorCommonT::IsPlaceH;
  static constexpr bool IsConst = AccessorCommonT::IsConst;
  static constexpr bool IsHostTask = AccessorCommonT::IsHostTask;
  template <int Dims>
  using AccessorSubscript =
      typename AccessorCommonT::template AccessorSubscript<Dims>;

  static_assert(
      !IsConst || IsAccessReadOnly,
      "A const qualified DataT is only allowed for a read-only accessor");

  using ConcreteASPtrType = typename detail::DecoratedType<
      typename std::conditional_t<IsAccessReadOnly && !IsConstantBuf,
                                  const DataT, DataT>,
      AS>::type *;

  using RefType = detail::const_if_const_AS<AS, DataT> &;
  using ConstRefType = const DataT &;
  using PtrType = detail::const_if_const_AS<AS, DataT> *;

  template <int Dims = Dimensions> size_t getLinearIndex(id<Dims> Id) const {

    size_t Result = 0;
    detail::loop<Dims>([&, this](size_t I) {
      Result = Result * getMemoryRange()[I] + Id[I];
      // We've already adjusted for the accessor's offset in the __init, so
      // don't include it here in case of device.
#ifndef __SYCL_DEVICE_ONLY__
      if constexpr (!(PropertyListT::template has_property<
                        sycl::ext::oneapi::property::no_offset>())) {
        Result += getOffset()[I];
      }
#endif // __SYCL_DEVICE_ONLY__
    });

    return Result;
  }

  template <typename T, int Dims>
  struct IsSameAsBuffer
      : std::bool_constant<std::is_same_v<T, DataT> && (Dims > 0) &&
                           (Dims == Dimensions)> {};

  static access::mode getAdjustedMode(const PropertyListT &PropertyList) {
    access::mode AdjustedMode = AccessMode;

    if (PropertyList.template has_property<property::no_init>() ||
        PropertyList.template has_property<property::noinit>()) {
      if (AdjustedMode == access::mode::write) {
        AdjustedMode = access::mode::discard_write;
      } else if (AdjustedMode == access::mode::read_write) {
        AdjustedMode = access::mode::discard_read_write;
      }
    }

    return AdjustedMode;
  }

  template <typename TagT>
  struct IsValidTag
      : std::disjunction<
            std::is_same<TagT, mode_tag_t<AccessMode>>,
            std::is_same<TagT, mode_target_tag_t<AccessMode, AccessTarget>>> {};

  template <typename DataT_, int Dimensions_, access::mode AccessMode_,
            access::target AccessTarget_, access::placeholder IsPlaceholder_,
            typename PropertyListT_>
  friend class accessor;

#ifdef __SYCL_DEVICE_ONLY__

  id<AdjustedDim> &getOffset() { return impl.Offset; }
  range<AdjustedDim> &getAccessRange() { return impl.AccessRange; }
  range<AdjustedDim> &getMemoryRange() { return impl.MemRange; }

  const id<AdjustedDim> &getOffset() const { return impl.Offset; }
  const range<AdjustedDim> &getAccessRange() const { return impl.AccessRange; }
  const range<AdjustedDim> &getMemoryRange() const { return impl.MemRange; }

  detail::AccessorImplDevice<AdjustedDim> impl;

  union {
    ConcreteASPtrType MData;
  };

  void __init(ConcreteASPtrType Ptr, range<AdjustedDim> AccessRange,
              range<AdjustedDim> MemRange, id<AdjustedDim> Offset) {
    MData = Ptr;
    detail::loop<AdjustedDim>([&, this](size_t I) {
      if constexpr (!(PropertyListT::template has_property<
                        sycl::ext::oneapi::property::no_offset>())) {
        getOffset()[I] = Offset[I];
      }
      getAccessRange()[I] = AccessRange[I];
      getMemoryRange()[I] = MemRange[I];
    });

    // Adjust for offsets as that part is invariant for all invocations of
    // operator[]. Will have to re-adjust in get_pointer.
    MData += getTotalOffset();
  }

  // __init variant used by the device compiler for ESIMD kernels.
  // TODO: In ESIMD accessors usage is limited for now - access range, mem
  // range and offset are not supported.
  void __init_esimd(ConcreteASPtrType Ptr) {
    MData = Ptr;
#ifdef __ESIMD_FORCE_STATELESS_MEM
    detail::loop<AdjustedDim>([&, this](size_t I) {
      getOffset()[I] = 0;
      getAccessRange()[I] = 0;
      getMemoryRange()[I] = 0;
    });
#endif
  }

  ConcreteASPtrType getQualifiedPtr() const noexcept { return MData; }

#ifndef __SYCL_DEVICE_ONLY__
  using AccessorBaseHost::impl;
#endif

public:
  // Default constructor for objects later initialized with __init member.
  accessor()
      : impl({}, detail::InitializedVal<AdjustedDim, range>::template get<0>(),
             detail::InitializedVal<AdjustedDim, range>::template get<0>()) {}

#else
  accessor(const detail::AccessorImplPtr &Impl)
      : detail::AccessorBaseHost{Impl} {}

  void *getPtr() { return AccessorBaseHost::getPtr(); }

  const id<3> getOffset() const {
    if constexpr (IsHostBuf)
      return MAccData ? MAccData->MOffset : id<3>();
    else
      return AccessorBaseHost::getOffset();
  }
  const range<3> &getAccessRange() const {
    return AccessorBaseHost::getAccessRange();
  }
  const range<3> getMemoryRange() const {
    if constexpr (IsHostBuf)
      return MAccData ? MAccData->MMemoryRange : range(0, 0, 0);
    else
      return AccessorBaseHost::getMemoryRange();
  }

  void *getPtr() const { return AccessorBaseHost::getPtr(); }

  void initHostAcc() { MAccData = &getAccData(); }

  // The function references helper methods required by GDB pretty-printers
  void GDBMethodsAnchor() {
#ifndef NDEBUG
    const auto *this_const = this;
    (void)getMemoryRange();
    (void)this_const->getMemoryRange();
    (void)getOffset();
    (void)this_const->getOffset();
    (void)getPtr();
    (void)this_const->getPtr();
    (void)getAccessRange();
    (void)this_const->getAccessRange();
#endif
  }

  detail::AccHostDataT *MAccData = nullptr;

  char padding[sizeof(detail::AccessorImplDevice<AdjustedDim>) +
               sizeof(PtrType) - sizeof(detail::AccessorBaseHost) -
               sizeof(MAccData)];

  PtrType getQualifiedPtr() const noexcept {
    if constexpr (IsHostBuf)
      return MAccData ? reinterpret_cast<PtrType>(MAccData->MData) : nullptr;
    else
      return reinterpret_cast<PtrType>(AccessorBaseHost::getPtr());
  }

public:
  accessor()
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0}, /*AccessRange=*/{0, 0, 0},
            /*MemoryRange=*/{0, 0, 0},
            /*AccessMode=*/getAdjustedMode({}),
            /*SYCLMemObject=*/nullptr, /*Dims=*/0, /*ElemSize=*/0,
            /*IsPlaceH=*/false,
            /*OffsetInBytes=*/0, /*IsSubBuffer=*/false, /*PropertyList=*/{}){};

  template <typename, int, access_mode> friend class host_accessor;

#endif // __SYCL_DEVICE_ONLY__

private:
  friend class sycl::stream;
  friend class sycl::ext::intel::esimd::detail::AccessorPrivateProxy;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  // 4.7.6.9.1. Interface for buffer command accessors
  // value_type is defined as const DataT for read_only accessors, DataT
  // otherwise
  using value_type =
      std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;

  template <access::decorated IsDecorated>
  using accessor_ptr =
      std::conditional_t<AccessTarget == access::target::device,
                         global_ptr<value_type, IsDecorated>, value_type *>;

  using iterator = typename detail::accessor_iterator<value_type, AdjustedDim>;
  using const_iterator =
      typename detail::accessor_iterator<const value_type, AdjustedDim>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  /// If creating a host_accessor this checks to see if the underlying memory
  /// object is currently in use by a command_graph, and throws if it is.
  void throwIfUsedByGraph() const {
#ifndef __SYCL_DEVICE_ONLY__
    if (IsHostBuf && AccessorBaseHost::isMemoryObjectUsedByGraph()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Host accessors cannot be created for buffers "
                            "which are currently in use by a command graph.");
    }
#endif
  }

  // The list of accessor constructors with their arguments
  // -------+---------+-------+----+-----+--------------
  // Dimensions = 0
  // -------+---------+-------+----+-----+--------------
  // buffer |         |       |    |     | property_list
  // buffer | handler |       |    |     | property_list
  // -------+---------+-------+----+-----+--------------
  // Dimensions >= 1
  // -------+---------+-------+----+-----+--------------
  // buffer |         |       |    |     | property_list
  // buffer |         |       |    | tag | property_list
  // buffer | handler |       |    |     | property_list
  // buffer | handler |       |    | tag | property_list
  // buffer |         | range |    |     | property_list
  // buffer |         | range |    | tag | property_list
  // buffer | handler | range |    |     | property_list
  // buffer | handler | range |    | tag | property_list
  // buffer |         | range | id |     | property_list
  // buffer |         | range | id | tag | property_list
  // buffer | handler | range | id |     | property_list
  // buffer | handler | range | id | tag | property_list
  // -------+---------+-------+----+-----+--------------

public:
  // implicit conversion between const / non-const types for read only accessors
  template <typename DataT_,
            typename = std::enable_if_t<
                IsAccessReadOnly && !std::is_same_v<DataT_, DataT> &&
                std::is_same_v<std::remove_const_t<DataT_>,
                               std::remove_const_t<DataT>>>>
  accessor(const accessor<DataT_, Dimensions, AccessMode, AccessTarget,
                          IsPlaceholder, PropertyListT> &other)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(other.impl), MData(other.MData) {
#else
      : accessor(other.impl) {
#endif // __SYCL_DEVICE_ONLY__
  }

  // implicit conversion from read_write T accessor to read only T (const)
  // accessor
  template <typename DataT_, access::mode AccessMode_,
            typename = std::enable_if_t<
                (AccessMode_ == access_mode::read_write) && IsAccessReadOnly &&
                std::is_same_v<std::remove_const_t<DataT_>,
                               std::remove_const_t<DataT>>>>
  accessor(const accessor<DataT_, Dimensions, AccessMode_, AccessTarget,
                          IsPlaceholder, PropertyListT> &other)
#ifdef __SYCL_DEVICE_ONLY__
      : impl(other.impl), MData(other.MData) {
#else
      : accessor(other.impl) {
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                std::is_same_v<T, DataT> && Dims == 0 &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))> * =
                nullptr>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), detail::GetZeroDimAccessRange(BufferRef),
             BufferRef.get_range()) {
    (void)PropertyList;
    (void)CodeLoc;
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(
                detail::GetZeroDimAccessRange(BufferRef)),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), AdjustedDim, sizeof(DataT),
            IsPlaceH, BufferRef.OffsetInBytes, BufferRef.IsSubBuffer,
            PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (!AccessorBaseHost::isPlaceholder())
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
#endif
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                // VS2019 can't compile sycl/test/regression/bit_cast_win.cpp
                // if std::is_same_v is used here.
                std::is_same<T, DataT>::value && Dims == 0 &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))> * =
                nullptr>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), detail::GetZeroDimAccessRange(BufferRef),
             BufferRef.get_range()) {
    (void)PropertyList;
    (void)CodeLoc;
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(
                detail::GetZeroDimAccessRange(BufferRef)),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), AdjustedDim, sizeof(DataT),
            IsPlaceH, BufferRef.OffsetInBytes, BufferRef.IsSubBuffer,
            PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (!AccessorBaseHost::isPlaceholder())
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
#endif
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = typename std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                std::is_same_v<T, DataT> && (Dims == 0) &&
                (IsGlobalBuf || IsHostBuf || IsConstantBuf || IsHostTask)>>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), detail::GetZeroDimAccessRange(BufferRef),
             BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(
                detail::GetZeroDimAccessRange(BufferRef)),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = typename std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                std::is_same_v<T, DataT> && (Dims == 0) &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), detail::GetZeroDimAccessRange(BufferRef),
             BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(
                detail::GetZeroDimAccessRange(BufferRef)),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<Dimensions>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            IsPlaceH, BufferRef.OffsetInBytes, BufferRef.IsSubBuffer,
            PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (!AccessorBaseHost::isPlaceholder())
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<Dimensions>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            IsPlaceH, BufferRef.OffsetInBytes, BufferRef.IsSubBuffer,
            PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (!AccessorBaseHost::isPlaceholder())
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, TagT,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    initHostAcc();
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      TagT, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      TagT, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                 CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                 CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, TagT,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                 CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                 CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), IsPlaceH, BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (!AccessorBaseHost::isPlaceholder())
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    if (BufferRef.isOutOfBounds(AccessOffset, AccessRange,
                                BufferRef.get_range()))
      throw sycl::invalid_object_error(
          "accessor with requested offset and range would exceed the bounds of "
          "the buffer",
          PI_ERROR_INVALID_VALUE);

    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsHostBuf || IsHostTask || (IsGlobalBuf || IsConstantBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), IsPlaceH, BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (!AccessorBaseHost::isPlaceholder())
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    if (BufferRef.isOutOfBounds(AccessOffset, AccessRange,
                                BufferRef.get_range()))
      throw sycl::invalid_object_error(
          "accessor with requested offset and range would exceed the bounds of "
          "the buffer",
          PI_ERROR_INVALID_VALUE);

    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, TagT, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, AccessOffset, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, AccessOffset, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (BufferRef.isOutOfBounds(AccessOffset, AccessRange,
                                BufferRef.get_range()))
      throw sycl::invalid_object_error(
          "accessor with requested offset and range would exceed the bounds of "
          "the buffer",
          PI_ERROR_INVALID_VALUE);

    initHostAcc();
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
    (void)CodeLoc;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    throwIfUsedByGraph();
    preScreenAccessor(PropertyList);
    if (BufferRef.isOutOfBounds(AccessOffset, AccessRange,
                                BufferRef.get_range()))
      throw sycl::invalid_object_error(
          "accessor with requested offset and range would exceed the bounds of "
          "the buffer",
          PI_ERROR_INVALID_VALUE);

    initHostAcc();
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = std::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset, TagT,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, AccessOffset,
                 PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = std::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                (IsGlobalBuf || IsConstantBuf || IsHostBuf || IsHostTask)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset, TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, AccessOffset,
                 PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename... NewPropsT>
  accessor(
      const accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder,
                     ext::oneapi::accessor_property_list<NewPropsT...>> &Other,
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(Other.impl), MData(Other.MData)
#else
      : detail::AccessorBaseHost(Other), MAccData(Other.MAccData)
#endif
  {
    static_assert(detail::IsCxPropertyList<PropertyListT>::value,
                  "Conversion is only available for accessor_property_list");
    static_assert(
        PropertyListT::template areSameCompileTimeProperties<NewPropsT...>(),
        "Compile-time-constant properties must be the same");
    (void)CodeLoc;
#ifndef __SYCL_DEVICE_ONLY__
    detail::constructorNotification(getMemoryObject(), impl.get(), AccessTarget,
                                    AccessMode, CodeLoc);
#endif
  }

  void swap(accessor &other) {
    std::swap(impl, other.impl);
#ifdef __SYCL_DEVICE_ONLY__
    std::swap(MData, other.MData);
#else
    std::swap(MAccData, other.MAccData);
#endif
  }

  bool is_placeholder() const {
#ifdef __SYCL_DEVICE_ONLY__
    return false;
#else
    return detail::AccessorBaseHost::isPlaceholder();
#endif
  }

  size_t get_size() const { return getAccessRange().size() * sizeof(DataT); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_type size() const noexcept { return getAccessRange().size(); }

  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  size_type max_size() const noexcept {
    return empty() ? 0 : (std::numeric_limits<difference_type>::max)();
  }

  bool empty() const noexcept { return size() == 0; }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<Dims == Dimensions && (Dims > 0)>>
  range<Dimensions> get_range() const {
    return getRange<Dims>();
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<Dims == Dimensions && (Dims > 0)>>
  id<Dimensions> get_offset() const {
    return getOffset<Dims>();
  }

  template <int Dims = Dimensions, typename RefT = RefType,
            typename = std::enable_if_t<Dims == 0 &&
                                        (IsAccessAnyWrite || IsAccessReadOnly)>>
  operator reference() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return *(getQualifiedPtr() + LinearIndex);
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<AccessMode != access_mode::atomic &&
                                        !IsAccessReadOnly && Dims == 0>>
  const accessor &operator=(const value_type &Other) const {
    *getQualifiedPtr() = Other;
    return *this;
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<AccessMode != access_mode::atomic &&
                                        !IsAccessReadOnly && Dims == 0>>
  const accessor &operator=(value_type &&Other) const {
    *getQualifiedPtr() = std::move(Other);
    return *this;
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<(Dims > 0) &&
                                        (IsAccessAnyWrite || IsAccessReadOnly)>>
  reference operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions>
  operator typename std::enable_if_t<Dims == 0 &&
                                         AccessMode == access::mode::atomic,
#ifdef __ENABLE_USM_ADDR_SPACE__
                                     atomic<DataT>
#else
                                     atomic<DataT, AS>
#endif
                                     >() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }

  template <int Dims = Dimensions>
  typename std::enable_if_t<(Dims > 0) && AccessMode == access::mode::atomic,
                            atomic<DataT, AS>>
  operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }

  template <int Dims = Dimensions>
  typename std::enable_if_t<Dims == 1 && AccessMode == access::mode::atomic,
                            atomic<DataT, AS>>
  operator[](size_t Index) const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>(Index));
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }
  template <int Dims = Dimensions, typename = std::enable_if_t<(Dims > 1)>>
  auto operator[](size_t Index) const {
    return AccessorSubscript<Dims - 1>(*this, Index);
  }

  template <access::target AccessTarget_ = AccessTarget,
            typename = std::enable_if_t<
                (AccessTarget_ == access::target::host_buffer) ||
                (AccessTarget_ == access::target::host_task)>>
  std::add_pointer_t<value_type> get_pointer() const noexcept {
    return getPointerAdjusted();
  }

  template <
      access::target AccessTarget_ = AccessTarget,
      typename = std::enable_if_t<(AccessTarget_ == access::target::device)>>
  __SYCL2020_DEPRECATED(
      "accessor::get_pointer() is deprecated, please use get_multi_ptr()")
  global_ptr<DataT> get_pointer() const noexcept {
    return global_ptr<DataT>(
        const_cast<typename detail::DecoratedType<DataT, AS>::type *>(
            getPointerAdjusted()));
  }

  template <access::target AccessTarget_ = AccessTarget,
            typename = std::enable_if_t<AccessTarget_ ==
                                        access::target::constant_buffer>>
  constant_ptr<DataT> get_pointer() const {
    return constant_ptr<DataT>(getPointerAdjusted());
  }

  template <access::decorated IsDecorated>
  accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
    return accessor_ptr<IsDecorated>(getPointerAdjusted());
  }

  // accessor::has_property for runtime properties is only available in host
  // code. This restriction is not listed in the core spec and will be added in
  // future versions.
  template <typename Property>
  typename std::enable_if_t<
      !ext::oneapi::is_compile_time_property<Property>::value, bool>
  has_property() const noexcept {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template has_property<Property>();
#else
    return false;
#endif
  }

  // accessor::get_property for runtime properties is only available in host
  // code. This restriction is not listed in the core spec and will be added in
  // future versions.
  template <typename Property,
            typename = typename std::enable_if_t<
                !ext::oneapi::is_compile_time_property<Property>::value>>
  Property get_property() const {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template get_property<Property>();
#else
    return Property();
#endif
  }

  template <typename Property>
  static constexpr bool has_property(
      typename std::enable_if_t<
          ext::oneapi::is_compile_time_property<Property>::value> * = 0) {
    return PropertyListT::template has_property<Property>();
  }

  template <typename Property>
  static constexpr auto get_property(
      typename std::enable_if_t<
          ext::oneapi::is_compile_time_property<Property>::value> * = 0) {
    return PropertyListT::template get_property<Property>();
  }

  bool operator==(const accessor &Rhs) const { return impl == Rhs.impl; }
  bool operator!=(const accessor &Rhs) const { return !(*this == Rhs); }

  iterator begin() const noexcept {
    return iterator::getBegin(
        get_pointer(),
        detail::convertToArrayOfN<AdjustedDim, 1>(getMemoryRange()),
        getRange<AdjustedDim>(), getOffset<AdjustedDim>());
  }

  iterator end() const noexcept {
    return iterator::getEnd(
        get_pointer(),
        detail::convertToArrayOfN<AdjustedDim, 1>(getMemoryRange()),
        getRange<AdjustedDim>(), getOffset<AdjustedDim>());
  }

  const_iterator cbegin() const noexcept {
    return const_iterator::getBegin(
        get_pointer(),
        detail::convertToArrayOfN<AdjustedDim, 1>(getMemoryRange()),
        getRange<AdjustedDim>(), getOffset<AdjustedDim>());
  }

  const_iterator cend() const noexcept {
    return const_iterator::getEnd(
        get_pointer(),
        detail::convertToArrayOfN<AdjustedDim, 1>(getMemoryRange()),
        getRange<AdjustedDim>(), getOffset<AdjustedDim>());
  }

  reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

private:
  template <int Dims, typename = std::enable_if_t<(Dims > 0)>>
  range<Dims> getRange() const {
    return detail::convertToArrayOfN<AdjustedDim, 1>(getAccessRange());
  }

  template <int Dims = Dimensions, typename = std::enable_if_t<(Dims > 0)>>
  id<Dims> getOffset() const {
    static_assert(
        !(PropertyListT::template has_property<
            sycl::ext::oneapi::property::no_offset>()),
        "Accessor has no_offset property, get_offset() can not be used");
    return detail::convertToArrayOfN<Dims, 0>(getOffset());
  }

#ifdef __SYCL_DEVICE_ONLY__
  size_t getTotalOffset() const noexcept {
    size_t TotalOffset = 0;
    detail::loop<Dimensions>([&, this](size_t I) {
      TotalOffset = TotalOffset * impl.MemRange[I];
      if constexpr (!(PropertyListT::template has_property<
                        sycl::ext::oneapi::property::no_offset>())) {
        TotalOffset += impl.Offset[I];
      }
    });

    return TotalOffset;
  }
#endif

  // supporting function for get_pointer()
  // MData has been preadjusted with offset for faster access with []
  // but for get_pointer() we must return the original pointer.
  // On device, getQualifiedPtr() returns MData, so we need to backjust it.
  // On host, getQualifiedPtr() does not return MData, no need to adjust.
  auto getPointerAdjusted() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return getQualifiedPtr() - getTotalOffset();
#else
    return getQualifiedPtr();
#endif
  }

  void preScreenAccessor(const PropertyListT &PropertyList) {
    // check that no_init property is compatible with access mode
    if (PropertyList.template has_property<property::no_init>() &&
        AccessMode == access::mode::read) {
      throw sycl::invalid_object_error(
          "accessor would cannot be both read_only and no_init",
          PI_ERROR_INVALID_VALUE);
    }
  }

  template <typename BufT, typename... PropTypes>
  void adjustAccPropsInBuf(BufT &Buffer) {
    if constexpr (PropertyListT::template has_property<
                      sycl::ext::intel::property::buffer_location>()) {
      auto location = (PropertyListT::template get_property<
                           sycl::ext::intel::property::buffer_location>())
                          .get_location();
      property_list PropList{
          sycl::property::buffer::detail::buffer_location(location)};
      Buffer.addOrReplaceAccessorProperties(PropList);
    } else {
      deleteAccPropsFromBuf(Buffer);
    }
  }

  template <typename BufT> void deleteAccPropsFromBuf(BufT &Buffer) {
    Buffer.deleteAccProps(
        sycl::detail::PropWithDataKind::AccPropBufferLocation);
  }
};

template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>)
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device,
                access::placeholder::true_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device,
                access::placeholder::true_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type1>(),
                detail::deduceAccessTarget<Type1, Type1>(target::device),
                access::placeholder::true_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type1>(),
                detail::deduceAccessTarget<Type1, Type1>(target::device),
                access::placeholder::true_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type2>(),
                detail::deduceAccessTarget<Type1, Type2>(target::device),
                access::placeholder::true_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type2>(),
                detail::deduceAccessTarget<Type1, Type2>(target::device),
                access::placeholder::true_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type2, Type3>(),
                detail::deduceAccessTarget<Type2, Type3>(target::device),
                access::placeholder::true_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type2, Type3>(),
                detail::deduceAccessTarget<Type2, Type3>(target::device),
                access::placeholder::true_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename Type4>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3, Type4)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type3, Type4>(),
                detail::deduceAccessTarget<Type3, Type4>(target::device),
                access::placeholder::true_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename Type4, typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3, Type4,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type3, Type4>(),
                detail::deduceAccessTarget<Type3, Type4>(target::device),
                access::placeholder::true_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &)
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, access::mode::read_write, target::device,
                access::placeholder::false_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type1>(),
                detail::deduceAccessTarget<Type1, Type1>(target::device),
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type1>(),
                detail::deduceAccessTarget<Type1, Type1>(target::device),
                access::placeholder::false_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1, Type2)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type2>(),
                detail::deduceAccessTarget<Type1, Type2>(target::device),
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1, Type2,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type1, Type2>(),
                detail::deduceAccessTarget<Type1, Type2>(target::device),
                access::placeholder::false_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1, Type2, Type3)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type2, Type3>(),
                detail::deduceAccessTarget<Type2, Type3>(target::device),
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1, Type2, Type3,
         const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type2, Type3>(),
                detail::deduceAccessTarget<Type2, Type3>(target::device),
                access::placeholder::false_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename Type4>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1, Type2, Type3,
         Type4)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type3, Type4>(),
                detail::deduceAccessTarget<Type3, Type4>(target::device),
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename Type4, typename... PropsT>
accessor(buffer<DataT, Dimensions, AllocatorT>, handler &, Type1, Type2, Type3,
         Type4, const ext::oneapi::accessor_property_list<PropsT...> &)
    -> accessor<DataT, Dimensions, detail::deduceAccessMode<Type3, Type4>(),
                detail::deduceAccessTarget<Type3, Type4>(target::device),
                access::placeholder::false_t,
                ext::oneapi::accessor_property_list<PropsT...>>;

/// Local accessor
///
/// \ingroup sycl_api_acc
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __SYCL_SPECIAL_CLASS local_accessor_base :
#ifndef __SYCL_DEVICE_ONLY__
    public detail::LocalAccessorBaseHost,
#endif
    public detail::accessor_common<DataT, Dimensions, AccessMode,
                                   access::target::local, IsPlaceholder> {
protected:
  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;

  using AccessorCommonT =
      detail::accessor_common<DataT, Dimensions, AccessMode,
                              access::target::local, IsPlaceholder>;

  using AccessorCommonT::AS;

  // Cannot do "using AccessorCommonT::Flag" as it doesn't work with g++ as host
  // compiler, for some reason.
  static constexpr bool IsAccessAnyWrite = AccessorCommonT::IsAccessAnyWrite;
  static constexpr bool IsAccessReadOnly = AccessorCommonT::IsAccessReadOnly;
  static constexpr bool IsConst = AccessorCommonT::IsConst;

  template <int Dims>
  using AccessorSubscript =
      typename AccessorCommonT::template AccessorSubscript<
          Dims,
          local_accessor_base<DataT, Dimensions, AccessMode, IsPlaceholder>>;

  using ConcreteASPtrType = typename detail::DecoratedType<DataT, AS>::type *;

  using RefType = detail::const_if_const_AS<AS, DataT> &;
  using PtrType = detail::const_if_const_AS<AS, DataT> *;

#ifdef __SYCL_DEVICE_ONLY__
  detail::LocalAccessorBaseDevice<AdjustedDim> impl;

  sycl::range<AdjustedDim> &getSize() { return impl.MemRange; }
  const sycl::range<AdjustedDim> &getSize() const { return impl.MemRange; }

  void __init(ConcreteASPtrType Ptr, range<AdjustedDim> AccessRange,
              range<AdjustedDim>, id<AdjustedDim>) {
    MData = Ptr;
    detail::loop<AdjustedDim>(
        [&, this](size_t I) { getSize()[I] = AccessRange[I]; });
  }

  // __init variant used by the device compiler for ESIMD kernels.
  // TODO: In ESIMD accessors usage is limited for now - access range, mem
  // range and offset are not supported.
  void __init_esimd(ConcreteASPtrType Ptr) {
    MData = Ptr;
    detail::loop<AdjustedDim>([&, this](size_t I) { getSize()[I] = 0; });
  }

public:
  // Default constructor for objects later initialized with __init member.
  local_accessor_base()
      : impl(detail::InitializedVal<AdjustedDim, range>::template get<0>()) {}

protected:
  ConcreteASPtrType getQualifiedPtr() const { return MData; }

  ConcreteASPtrType MData;

#else
public:
  local_accessor_base()
      : detail::LocalAccessorBaseHost{/*Size*/ sycl::range<3>{0, 0, 0},
                                      /*Dims*/ 0, /*ElemSize*/ sizeof(DataT)} {}

protected:
  local_accessor_base(const detail::LocalAccessorImplPtr &Impl)
      : detail::LocalAccessorBaseHost{Impl} {}

  char padding[sizeof(detail::LocalAccessorBaseDevice<AdjustedDim>) +
               sizeof(PtrType) - sizeof(detail::LocalAccessorBaseHost)];
  using detail::LocalAccessorBaseHost::getSize;

  PtrType getQualifiedPtr() const {
    return reinterpret_cast<PtrType>(LocalAccessorBaseHost::getPtr());
  }

  void *getPtr() { return detail::LocalAccessorBaseHost::getPtr(); }
  void *getPtr() const { return detail::LocalAccessorBaseHost::getPtr(); }
  const range<3> &getSize() const {
    return detail::LocalAccessorBaseHost::getSize();
  }
  range<3> &getSize() { return detail::LocalAccessorBaseHost::getSize(); }

  // The function references helper methods required by GDB pretty-printers
  void GDBMethodsAnchor() {
#ifndef NDEBUG
    const auto *this_const = this;
    (void)getSize();
    (void)this_const->getSize();
    (void)getPtr();
    (void)this_const->getPtr();
#endif
  }

#endif // __SYCL_DEVICE_ONLY__

  // Method which calculates linear offset for the ID using Range and Offset.
  template <int Dims = AdjustedDim> size_t getLinearIndex(id<Dims> Id) const {
    size_t Result = 0;
    detail::loop<Dims>(
        [&, this](size_t I) { Result = Result * getSize()[I] + Id[I]; });
    return Result;
  }

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <typename DataT_, int Dimensions_> friend class local_accessor;

public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 0>>
  local_accessor_base(handler &, const detail::code_location CodeLoc =
                                     detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(range<AdjustedDim>{1}) {
    (void)CodeLoc;
  }
#else
      : LocalAccessorBaseHost(range<3>{1, 1, 1}, AdjustedDim, sizeof(DataT)) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

        template <int Dims = Dimensions, typename = std::enable_if_t<Dims == 0>>
        local_accessor_base(handler &, const property_list &propList,
                            const detail::code_location CodeLoc =
                                detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(range<AdjustedDim>{1}) {
    (void)propList;
    (void)CodeLoc;
  }
#else
      : LocalAccessorBaseHost(range<3>{1, 1, 1}, AdjustedDim, sizeof(DataT),
                              propList) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <int Dims = Dimensions, typename = std::enable_if_t<(Dims > 0)>>
  local_accessor_base(
      range<Dimensions> AllocationSize, handler &,
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AllocationSize) {
    (void)CodeLoc;
  }
#else
      : LocalAccessorBaseHost(detail::convertToArrayOfN<3, 1>(AllocationSize),
                              AdjustedDim, sizeof(DataT)) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

        template <int Dims = Dimensions,
                  typename = std::enable_if_t<(Dims > 0)>>
        local_accessor_base(range<Dimensions> AllocationSize, handler &,
                            const property_list &propList,
                            const detail::code_location CodeLoc =
                                detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AllocationSize) {
    (void)propList;
    (void)CodeLoc;
  }
#else
      : LocalAccessorBaseHost(detail::convertToArrayOfN<3, 1>(AllocationSize),
                              AdjustedDim, sizeof(DataT), propList) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  size_t get_size() const { return getSize().size() * sizeof(DataT); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return getSize().size(); }

  template <int Dims = Dimensions, typename = std::enable_if_t<(Dims > 0)>>
  range<Dims> get_range() const {
    return detail::convertToArrayOfN<Dims, 1>(getSize());
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<Dims == 0 &&
                                        (IsAccessAnyWrite || IsAccessReadOnly)>>
  operator RefType() const {
    return *getQualifiedPtr();
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<(Dims > 0) &&
                                        (IsAccessAnyWrite || IsAccessReadOnly)>>
  RefType operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<Dims == 1 &&
                                        (IsAccessAnyWrite || IsAccessReadOnly)>>
  RefType operator[](size_t Index) const {
    return getQualifiedPtr()[Index];
  }

  template <int Dims = Dimensions>
  operator typename std::enable_if_t<
      Dims == 0 && AccessMode == access::mode::atomic, atomic<DataT, AS>>()
      const {
    return atomic<DataT, AS>(
        multi_ptr<DataT, AS, access::decorated::yes>(getQualifiedPtr()));
  }

  template <int Dims = Dimensions>
  typename std::enable_if_t<(Dims > 0) && AccessMode == access::mode::atomic,
                            atomic<DataT, AS>>
  operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }

  template <int Dims = Dimensions>
  typename std::enable_if_t<Dims == 1 && AccessMode == access::mode::atomic,
                            atomic<DataT, AS>>
  operator[](size_t Index) const {
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + Index));
  }

  template <int Dims = Dimensions, typename = std::enable_if_t<(Dims > 1)>>
  typename AccessorCommonT::template AccessorSubscript<
      Dims - 1,
      local_accessor_base<DataT, Dimensions, AccessMode, IsPlaceholder>>
  operator[](size_t Index) const {
    return AccessorSubscript<Dims - 1>(*this, Index);
  }

  bool operator==(const local_accessor_base &Rhs) const {
    return impl == Rhs.impl;
  }
  bool operator!=(const local_accessor_base &Rhs) const {
    return !(*this == Rhs);
  }
};

// TODO: Remove deprecated specialization once no longer needed
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __SYCL_EBO __SYCL_SPECIAL_CLASS accessor<
    DataT, Dimensions, AccessMode, access::target::local, IsPlaceholder>
    : public local_accessor_base<DataT, Dimensions, AccessMode, IsPlaceholder>,
      public detail::OwnerLessBase<
          accessor<DataT, Dimensions, AccessMode, access::target::local,
                   IsPlaceholder>> {

  using local_acc =
      local_accessor_base<DataT, Dimensions, AccessMode, IsPlaceholder>;

  static_assert(
      !local_acc::IsConst || local_acc::IsAccessReadOnly,
      "A const qualified DataT is only allowed for a read-only accessor");

  // Use base classes constructors
  using local_acc::local_acc;

public:
  local_ptr<DataT> get_pointer() const {
    return local_ptr<DataT>(local_acc::getQualifiedPtr());
  }

#ifdef __SYCL_DEVICE_ONLY__

  // __init needs to be defined within the class not through inheritance.
  // Map this function to inherited func.
  void __init(typename local_acc::ConcreteASPtrType Ptr,
              range<local_acc::AdjustedDim> AccessRange,
              range<local_acc::AdjustedDim> range,
              id<local_acc::AdjustedDim> id) {
    local_acc::__init(Ptr, AccessRange, range, id);
  }

  // __init variant used by the device compiler for ESIMD kernels.
  // TODO: In ESIMD accessors usage is limited for now - access range, mem
  // range and offset are not supported.
  void __init_esimd(typename local_acc::ConcreteASPtrType Ptr) {
    local_acc::__init_esimd(Ptr);
  }

public:
  // Default constructor for objects later initialized with __init member.
  accessor() {
    local_acc::impl = detail::InitializedVal<local_acc::AdjustedDim,
                                             range>::template get<0>();
  }

#else
private:
  accessor(const detail::AccessorImplPtr &Impl) : local_acc{Impl} {}
#endif
};

template <typename DataT, int Dimensions = 1>
class __SYCL_EBO __SYCL_SPECIAL_CLASS __SYCL_TYPE(local_accessor) local_accessor
    : public local_accessor_base<DataT, Dimensions,
                                 detail::accessModeFromConstness<DataT>(),
                                 access::placeholder::false_t>,
      public detail::OwnerLessBase<local_accessor<DataT, Dimensions>> {

  using local_acc =
      local_accessor_base<DataT, Dimensions,
                          detail::accessModeFromConstness<DataT>(),
                          access::placeholder::false_t>;

  static_assert(
      !local_acc::IsConst || local_acc::IsAccessReadOnly,
      "A const qualified DataT is only allowed for a read-only accessor");

  // Use base classes constructors
  using local_acc::local_acc;

#ifdef __SYCL_DEVICE_ONLY__

  // __init needs to be defined within the class not through inheritance.
  // Map this function to inherited func.
  void __init(typename local_acc::ConcreteASPtrType Ptr,
              range<local_acc::AdjustedDim> AccessRange,
              range<local_acc::AdjustedDim> range,
              id<local_acc::AdjustedDim> id) {
    local_acc::__init(Ptr, AccessRange, range, id);
  }

  // __init variant used by the device compiler for ESIMD kernels.
  // TODO: In ESIMD accessors usage is limited for now - access range, mem
  // range and offset are not supported.
  void __init_esimd(typename local_acc::ConcreteASPtrType Ptr) {
    local_acc::__init_esimd(Ptr);
  }

public:
  // Default constructor for objects later initialized with __init member.
  local_accessor() {
    local_acc::impl = detail::InitializedVal<local_acc::AdjustedDim,
                                             range>::template get<0>();
  }

#else
  local_accessor(const detail::AccessorImplPtr &Impl) : local_acc{Impl} {}
#endif

  // implicit conversion between non-const read-write accessor to const
  // read-only accessor
public:
  template <typename DataT_,
            typename = std::enable_if_t<
                std::is_const_v<DataT> &&
                std::is_same_v<DataT_, std::remove_const_t<DataT>>>>
  local_accessor(const local_accessor<DataT_, Dimensions> &other) {
    local_acc::impl = other.impl;
#ifdef __SYCL_DEVICE_ONLY__
    local_acc::MData = other.MData;
#endif
  }

  using value_type = DataT;
  using iterator = value_type *;
  using const_iterator = const value_type *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  template <access::decorated IsDecorated>
  using accessor_ptr = local_ptr<value_type, IsDecorated>;

  template <typename DataT_>
  bool operator==(const local_accessor<DataT_, Dimensions> &Rhs) const {
    return local_acc::impl == Rhs.impl;
  }

  template <typename DataT_>
  bool operator!=(const local_accessor<DataT_, Dimensions> &Rhs) const {
    return !(*this == Rhs);
  }

  void swap(local_accessor &other) { std::swap(this->impl, other.impl); }

  size_type byte_size() const noexcept { return this->size() * sizeof(DataT); }

  size_type max_size() const noexcept {
    return empty() ? 0 : (std::numeric_limits<difference_type>::max)();
  }

  bool empty() const noexcept { return this->size() == 0; }

  iterator begin() const noexcept {
    if constexpr (Dimensions == 0)
      return local_acc::getQualifiedPtr();
    else
      return &this->operator[](id<Dimensions>());
  }
  iterator end() const noexcept {
    if constexpr (Dimensions == 0)
      return begin() + 1;
    else
      return begin() + this->size();
  }

  const_iterator cbegin() const noexcept { return const_iterator(begin()); }
  const_iterator cend() const noexcept { return const_iterator(end()); }

  reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  __SYCL2020_DEPRECATED(
      "local_accessor::get_pointer() is deprecated, please use get_multi_ptr()")
  local_ptr<DataT> get_pointer() const noexcept {
    return local_ptr<DataT>(local_acc::getQualifiedPtr());
  }

  template <access::decorated IsDecorated>
  accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
    return accessor_ptr<IsDecorated>(local_acc::getQualifiedPtr());
  }

  template <typename Property> bool has_property() const noexcept {
#ifndef __SYCL_DEVICE_ONLY__
    return this->getPropList().template has_property<Property>();
#else
    return false;
#endif
  }

  template <typename Property> Property get_property() const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->getPropList().template get_property<Property>();
#else
    return Property();
#endif
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<!std::is_const_v<DataT> && Dims == 0>>
  const local_accessor &operator=(const value_type &Other) const {
    *local_acc::getQualifiedPtr() = Other;
    return *this;
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<!std::is_const_v<DataT> && Dims == 0>>
  const local_accessor &operator=(value_type &&Other) const {
    *local_acc::getQualifiedPtr() = std::move(Other);
    return *this;
  }

private:
  friend class sycl::ext::intel::esimd::detail::AccessorPrivateProxy;
};

/// Image accessors.
///
/// Available only when accessTarget == access::target::image.
///
/// \ingroup sycl_api_acc
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __SYCL_EBO __SYCL_SPECIAL_CLASS __SYCL_TYPE(accessor) accessor<
    DataT, Dimensions, AccessMode, access::target::image, IsPlaceholder>
    : public detail::image_accessor<DataT, Dimensions, AccessMode,
                                    access::target::image, IsPlaceholder>,
      public detail::OwnerLessBase<
          accessor<DataT, Dimensions, AccessMode, access::target::image,
                   IsPlaceholder>> {
private:
  accessor(const detail::AccessorImplPtr &Impl)
      : detail::image_accessor<DataT, Dimensions, AccessMode,
                               access::target::image, IsPlaceholder>{Impl} {}

public:
  template <typename AllocatorT>
  accessor(sycl::image<Dimensions, AllocatorT> &Image,
           handler &CommandGroupHandler)
      : detail::image_accessor<DataT, Dimensions, AccessMode,
                               access::target::image, IsPlaceholder>(
            Image, CommandGroupHandler, Image.getElementSize()) {
#ifndef __SYCL_DEVICE_ONLY__
    detail::associateWithHandler(CommandGroupHandler, this,
                                 access::target::image);
#endif
  }

  template <typename AllocatorT>
  accessor(sycl::image<Dimensions, AllocatorT> &Image,
           handler &CommandGroupHandler, const property_list &propList)
      : detail::image_accessor<DataT, Dimensions, AccessMode,
                               access::target::image, IsPlaceholder>(
            Image, CommandGroupHandler, Image.getElementSize()) {
    (void)propList;
#ifndef __SYCL_DEVICE_ONLY__
    detail::associateWithHandler(CommandGroupHandler, this,
                                 access::target::image);
#endif
  }
#ifdef __SYCL_DEVICE_ONLY__
private:
  using OCLImageTy =
      typename detail::opencl_image_type<Dimensions, AccessMode,
                                         access::target::image>::type;

  // Front End requires this method to be defined in the accessor class.
  // It does not call the base class's init method.
  void __init(OCLImageTy Image) { this->imageAccessorInit(Image); }

  // __init variant used by the device compiler for ESIMD kernels.
  void __init_esimd(OCLImageTy Image) { this->imageAccessorInit(Image); }

public:
  // Default constructor for objects later initialized with __init member.
  accessor() = default;
#endif
};

/// Host image accessor.
///
/// Available only when accessTarget == access::target::host_image.
///
/// \sa image
///
/// \ingroup sycl_api_acc
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __SYCL_EBO accessor<DataT, Dimensions, AccessMode,
                          access::target::host_image, IsPlaceholder>
    : public detail::image_accessor<DataT, Dimensions, AccessMode,
                                    access::target::host_image, IsPlaceholder>,
      public detail::OwnerLessBase<
          accessor<DataT, Dimensions, AccessMode, access::target::host_image,
                   IsPlaceholder>> {
public:
  template <typename AllocatorT>
  accessor(sycl::image<Dimensions, AllocatorT> &Image)
      : detail::image_accessor<DataT, Dimensions, AccessMode,
                               access::target::host_image, IsPlaceholder>(
            Image, Image.getElementSize()) {}

  template <typename AllocatorT>
  accessor(sycl::image<Dimensions, AllocatorT> &Image,
           const property_list &propList)
      : detail::image_accessor<DataT, Dimensions, AccessMode,
                               access::target::host_image, IsPlaceholder>(
            Image, Image.getElementSize()) {
    (void)propList;
  }
};

/// Image array accessor.
///
/// Available only when accessTarget == access::target::image_array and
/// dimensions < 3.
///
/// \sa image
///
/// \ingroup sycl_api_acc
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::placeholder IsPlaceholder>
class __SYCL_EBO __SYCL_SPECIAL_CLASS __SYCL_TYPE(accessor) accessor<
    DataT, Dimensions, AccessMode, access::target::image_array, IsPlaceholder>
    : public detail::image_accessor<DataT, Dimensions + 1, AccessMode,
                                    access::target::image, IsPlaceholder>,
      public detail::OwnerLessBase<
          accessor<DataT, Dimensions, AccessMode, access::target::image_array,
                   IsPlaceholder>> {
#ifdef __SYCL_DEVICE_ONLY__
private:
  using OCLImageTy =
      typename detail::opencl_image_type<Dimensions + 1, AccessMode,
                                         access::target::image>::type;

  // Front End requires this method to be defined in the accessor class.
  // It does not call the base class's init method.
  void __init(OCLImageTy Image) { this->imageAccessorInit(Image); }

  // __init variant used by the device compiler for ESIMD kernels.
  void __init_esimd(OCLImageTy Image) { this->imageAccessorInit(Image); }

public:
  // Default constructor for objects later initialized with __init member.
  accessor() = default;
#endif
public:
  template <typename AllocatorT>
  accessor(sycl::image<Dimensions + 1, AllocatorT> &Image,
           handler &CommandGroupHandler)
      : detail::image_accessor<DataT, Dimensions + 1, AccessMode,
                               access::target::image, IsPlaceholder>(
            Image, CommandGroupHandler, Image.getElementSize()) {
#ifndef __SYCL_DEVICE_ONLY__
    detail::associateWithHandler(CommandGroupHandler, this,
                                 access::target::image_array);
#endif
  }

  template <typename AllocatorT>
  accessor(sycl::image<Dimensions + 1, AllocatorT> &Image,
           handler &CommandGroupHandler, const property_list &propList)
      : detail::image_accessor<DataT, Dimensions + 1, AccessMode,
                               access::target::image, IsPlaceholder>(
            Image, CommandGroupHandler, Image.getElementSize()) {
    (void)propList;
#ifndef __SYCL_DEVICE_ONLY__
    detail::associateWithHandler(CommandGroupHandler, this,
                                 access::target::image_array);
#endif
  }

  detail::__image_array_slice__<DataT, Dimensions, AccessMode, IsPlaceholder>
  operator[](size_t Index) const {
    return detail::__image_array_slice__<DataT, Dimensions, AccessMode,
                                         IsPlaceholder>(*this, Index);
  }
};

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode = access_mode::read_write>
class __SYCL_EBO host_accessor
    : public accessor<DataT, Dimensions, AccessMode, target::host_buffer,
                      access::placeholder::false_t> {
protected:
  using AccessorT = accessor<DataT, Dimensions, AccessMode, target::host_buffer,
                             access::placeholder::false_t>;

  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;
  constexpr static bool IsAccessReadOnly = AccessMode == access::mode::read;

  template <typename T, int Dims>
  struct IsSameAsBuffer
      : std::bool_constant<std::is_same_v<T, DataT> && (Dims > 0) &&
                           (Dims == Dimensions)> {};

  void
  __init(typename accessor<DataT, Dimensions, AccessMode, target::host_buffer,
                           access::placeholder::false_t>::ConcreteASPtrType Ptr,
         range<AdjustedDim> AccessRange, range<AdjustedDim> MemRange,
         id<AdjustedDim> Offset) {
    AccessorT::__init(Ptr, AccessRange, MemRange, Offset);
  }

#ifndef __SYCL_DEVICE_ONLY__
  host_accessor(const detail::AccessorImplPtr &Impl)
      : accessor<DataT, Dimensions, AccessMode, target::host_buffer,
                 access::placeholder::false_t>{Impl} {}

  template <class Obj>
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
#endif // __SYCL_DEVICE_ONLY__

public:
  host_accessor() : AccessorT() {}

  // The list of host_accessor constructors with their arguments
  // -------+---------+-------+----+----------+--------------
  // Dimensions = 0
  // -------+---------+-------+----+----------+--------------
  // buffer |         |       |    |          | property_list
  // buffer | handler |       |    |          | property_list
  // -------+---------+-------+----+----------+--------------
  // Dimensions >= 1
  // -------+---------+-------+----+----------+--------------
  // buffer |         |       |    |          | property_list
  // buffer |         |       |    | mode_tag | property_list
  // buffer | handler |       |    |          | property_list
  // buffer | handler |       |    | mode_tag | property_list
  // buffer |         | range |    |          | property_list
  // buffer |         | range |    | mode_tag | property_list
  // buffer | handler | range |    |          | property_list
  // buffer | handler | range |    | mode_tag | property_list
  // buffer |         | range | id |          | property_list
  // buffer |         | range | id | mode_tag | property_list
  // buffer | handler | range | id |          | property_list
  // buffer | handler | range | id | mode_tag | property_list
  // -------+---------+-------+----+----------+--------------

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = typename std::enable_if_t<std::is_same_v<T, DataT> &&
                                                 Dims == 0>>
  host_accessor(
      buffer<T, 1, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, mode_tag_t<AccessMode>,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      mode_tag_t<AccessMode>, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      mode_tag_t<AccessMode>, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                  CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, mode_tag_t<AccessMode>,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, CommandGroupHandler, AccessRange, {},
                      PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, AccessRange, AccessOffset, PropertyList, CodeLoc) {
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, mode_tag_t<AccessMode>,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, AccessRange, AccessOffset, PropertyList,
                      CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, CommandGroupHandler, AccessRange, AccessOffset,
                  PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = std::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      mode_tag_t<AccessMode>, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, CommandGroupHandler, AccessRange, AccessOffset,
                      PropertyList, CodeLoc) {}

  template <int Dims = Dimensions,
            typename = std::enable_if_t<AccessMode != access_mode::atomic &&
                                        !IsAccessReadOnly && Dims == 0>>
  const host_accessor &
  operator=(const typename AccessorT::value_type &Other) const {
    *AccessorT::getQualifiedPtr() = Other;
    return *this;
  }

  template <int Dims = Dimensions,
            typename = std::enable_if_t<AccessMode != access_mode::atomic &&
                                        !IsAccessReadOnly && Dims == 0>>
  const host_accessor &operator=(typename AccessorT::value_type &&Other) const {
    *AccessorT::getQualifiedPtr() = std::move(Other);
    return *this;
  }

  // implicit conversion between const / non-const types for read only accessors
  template <typename DataT_,
            typename = std::enable_if_t<
                IsAccessReadOnly && !std::is_same_v<DataT_, DataT> &&
                std::is_same_v<std::remove_const_t<DataT_>,
                               std::remove_const_t<DataT>>>>
  host_accessor(const host_accessor<DataT_, Dimensions, AccessMode> &other)
#ifndef __SYCL_DEVICE_ONLY__
      : host_accessor(other.impl) {
    AccessorT::MAccData = other.MAccData;
#else
  {
    (void)other;
#endif // __SYCL_DEVICE_ONLY__
  }

  // implicit conversion from read_write T accessor to read only T (const)
  // accessor
  template <typename DataT_, access::mode AccessMode_,
            typename = std::enable_if_t<
                (AccessMode_ == access_mode::read_write) && IsAccessReadOnly &&
                std::is_same_v<DataT_, std::remove_const_t<DataT>>>>
  host_accessor(const host_accessor<DataT_, Dimensions, AccessMode_> &other)
#ifndef __SYCL_DEVICE_ONLY__
      : host_accessor(other.impl) {
    AccessorT::MAccData = other.MAccData;
#else
  {
    (void)other;
#endif // __SYCL_DEVICE_ONLY__
  }

  // host_accessor needs to explicitly define the owner_before member functions
  // as inheriting from OwnerLessBase causes base class conflicts.
  // TODO: Once host_accessor is detached from accessor, inherit from
  // OwnerLessBase instead.
#ifndef __SYCL_DEVICE_ONLY__
  bool ext_oneapi_owner_before(
      const ext::oneapi::detail::weak_object_base<host_accessor> &Other)
      const noexcept {
    return this->impl.owner_before(
        ext::oneapi::detail::getSyclWeakObjImpl(Other));
  }

  bool ext_oneapi_owner_before(const host_accessor &Other) const noexcept {
    return this->impl.owner_before(Other.impl);
  }
#else
  bool ext_oneapi_owner_before(
      const ext::oneapi::detail::weak_object_base<host_accessor> &Other)
      const noexcept;
  bool ext_oneapi_owner_before(const host_accessor &Other) const noexcept;
#endif
};

template <typename DataT, int Dimensions, typename AllocatorT>
host_accessor(buffer<DataT, Dimensions, AllocatorT>)
    -> host_accessor<DataT, Dimensions, access::mode::read_write>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1>
host_accessor(buffer<DataT, Dimensions, AllocatorT>, Type1)
    -> host_accessor<DataT, Dimensions,
                     detail::deduceAccessMode<Type1, Type1>()>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2>
host_accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2)
    -> host_accessor<DataT, Dimensions,
                     detail::deduceAccessMode<Type1, Type2>()>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3>
host_accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3)
    -> host_accessor<DataT, Dimensions,
                     detail::deduceAccessMode<Type2, Type3>()>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename Type4>
host_accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3, Type4)
    -> host_accessor<DataT, Dimensions,
                     detail::deduceAccessMode<Type3, Type4>()>;

template <typename DataT, int Dimensions, typename AllocatorT, typename Type1,
          typename Type2, typename Type3, typename Type4, typename Type5>
host_accessor(buffer<DataT, Dimensions, AllocatorT>, Type1, Type2, Type3, Type4,
              Type5) -> host_accessor<DataT, Dimensions,
                                      detail::deduceAccessMode<Type4, Type5>()>;

// SYCL 2020 image accessors

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget = image_target::device>
class __SYCL_EBO unsampled_image_accessor :
#ifndef __SYCL_DEVICE_ONLY__
    private detail::UnsampledImageAccessorBaseHost,
#endif // __SYCL_DEVICE_ONLY__
    public detail::OwnerLessBase<
        unsampled_image_accessor<DataT, Dimensions, AccessMode, AccessTarget>> {
  static_assert(std::is_same_v<DataT, int4> || std::is_same_v<DataT, uint4> ||
                    std::is_same_v<DataT, float4> ||
                    std::is_same_v<DataT, half4>,
                "The data type of an image accessor must be only int4, "
                "uint4, float4 or half4 from SYCL namespace");
  static_assert(AccessMode == access_mode::read ||
                    AccessMode == access_mode::write,
                "Access mode must be either read or write.");

#ifdef __SYCL_DEVICE_ONLY__
  char MPadding[sizeof(detail::UnsampledImageAccessorBaseHost)];
#else
  using host_base_class = detail::UnsampledImageAccessorBaseHost;
#endif // __SYCL_DEVICE_ONLY__

public:
  using value_type = typename std::conditional<AccessMode == access_mode::read,
                                               const DataT, DataT>::type;
  using reference = value_type &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  unsampled_image_accessor(
      unsampled_image<Dimensions, AllocatorT> &ImageRef,
      handler &CommandGroupHandlerRef, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
  {
    (void)ImageRef;
    (void)CommandGroupHandlerRef;
    (void)PropList;
    (void)CodeLoc;
  }
#else
      : host_base_class(detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                        AccessMode, detail::getSyclObjImpl(ImageRef).get(),
                        Dimensions, ImageRef.getElementSize(),
                        {ImageRef.getRowPitch(), ImageRef.getSlicePitch(), 0},
                        ImageRef.getChannelType(), ImageRef.getChannelOrder(),
                        PropList) {
    device Device = detail::getDeviceFromHandler(CommandGroupHandlerRef);
    // Avoid aspect::image warning.
    aspect ImageAspect = aspect::image;
    if (AccessTarget == image_target::device && !Device.has(ImageAspect))
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Device associated with command group handler does not have "
          "aspect::image.");

    detail::unsampledImageConstructorNotification(
        detail::getSyclObjImpl(ImageRef).get(), this->impl.get(), AccessTarget,
        AccessMode, (const void *)typeid(DataT).name(), sizeof(DataT), CodeLoc);
    detail::associateWithHandler(CommandGroupHandlerRef, this, AccessTarget);
    GDBMethodsAnchor();
  }
#endif // __SYCL_DEVICE_ONLY__

  /* -- common interface members -- */

  unsampled_image_accessor(const unsampled_image_accessor &Rhs) = default;

  unsampled_image_accessor(unsampled_image_accessor &&Rhs) = default;

  unsampled_image_accessor &
  operator=(const unsampled_image_accessor &Rhs) = default;

  unsampled_image_accessor &operator=(unsampled_image_accessor &&Rhs) = default;

  ~unsampled_image_accessor() = default;

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const unsampled_image_accessor &Rhs) const;
#else
  bool operator==(const unsampled_image_accessor &Rhs) const {
    return Rhs.impl == impl;
  }
#endif // __SYCL_DEVICE_ONLY__

  bool operator!=(const unsampled_image_accessor &Rhs) const {
    return !(Rhs == *this);
  }

  /* -- property interface members -- */

  size_t size() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    // Currently not reachable on device.
    return 0;
#else
    return host_base_class::getSize().size();
#endif // __SYCL_DEVICE_ONLY__
  }

  /* Available only when: AccessMode == access_mode::read
  if Dimensions == 1, CoordT = int
  if Dimensions == 2, CoordT = int2
  if Dimensions == 3, CoordT = int4 */
  template <typename CoordT,
            typename = std::enable_if_t<AccessMode == access_mode::read &&
                                        detail::IsValidUnsampledCoord2020DataT<
                                            Dimensions, CoordT>::value>>
  DataT read(const CoordT &Coords) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    // Currently not reachable on device.
    std::ignore = Coords;
    return {0, 0, 0, 0};
#else
    return host_base_class::read<DataT>(Coords);
#endif // __SYCL_DEVICE_ONLY__
  }

  /* Available only when: AccessMode == access_mode::write
  if Dimensions == 1, CoordT = int
  if Dimensions == 2, CoordT = int2
  if Dimensions == 3, CoordT = int4 */
  template <typename CoordT,
            typename = std::enable_if_t<AccessMode == access_mode::write &&
                                        detail::IsValidUnsampledCoord2020DataT<
                                            Dimensions, CoordT>::value>>
  void write(const CoordT &Coords, const DataT &Color) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Currently not reachable on device.
    std::ignore = Coords;
    std::ignore = Color;
#else
    host_base_class::write<DataT>(Coords, Color);
#endif // __SYCL_DEVICE_ONLY__
  }

private:
  unsampled_image_accessor(const detail::UnsampledImageAccessorImplPtr &Impl)
#ifndef __SYCL_DEVICE_ONLY__
      : host_base_class{Impl}
#endif // __SYCL_DEVICE_ONLY__
  {
    std::ignore = Impl;
  }

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write)>
class __SYCL_EBO host_unsampled_image_accessor
    : private detail::UnsampledImageAccessorBaseHost,
      public detail::OwnerLessBase<
          host_unsampled_image_accessor<DataT, Dimensions, AccessMode>> {
  static_assert(std::is_same_v<DataT, int4> || std::is_same_v<DataT, uint4> ||
                    std::is_same_v<DataT, float4> ||
                    std::is_same_v<DataT, half4>,
                "The data type of an image accessor must be only int4, "
                "uint4, float4 or half4 from SYCL namespace");

  using base_class = detail::UnsampledImageAccessorBaseHost;

public:
  using value_type = typename std::conditional<AccessMode == access_mode::read,
                                               const DataT, DataT>::type;
  using reference = value_type &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  host_unsampled_image_accessor(
      unsampled_image<Dimensions, AllocatorT> &ImageRef,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : base_class(detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                   AccessMode, detail::getSyclObjImpl(ImageRef).get(),
                   Dimensions, ImageRef.getElementSize(),
                   {ImageRef.getRowPitch(), ImageRef.getSlicePitch(), 0},
                   ImageRef.getChannelType(), ImageRef.getChannelOrder(),
                   PropList) {
    addHostUnsampledImageAccessorAndWait(base_class::impl.get());

    detail::unsampledImageConstructorNotification(
        detail::getSyclObjImpl(ImageRef).get(), this->impl.get(), std::nullopt,
        AccessMode, (const void *)typeid(DataT).name(), sizeof(DataT), CodeLoc);
  }

  /* -- common interface members -- */

  host_unsampled_image_accessor(const host_unsampled_image_accessor &Rhs) =
      default;

  host_unsampled_image_accessor(host_unsampled_image_accessor &&Rhs) = default;

  host_unsampled_image_accessor &
  operator=(const host_unsampled_image_accessor &Rhs) = default;

  host_unsampled_image_accessor &
  operator=(host_unsampled_image_accessor &&Rhs) = default;

  ~host_unsampled_image_accessor() = default;

  bool operator==(const host_unsampled_image_accessor &Rhs) const {
    return Rhs.impl == impl;
  }
  bool operator!=(const host_unsampled_image_accessor &Rhs) const {
    return !(Rhs == *this);
  }

  /* -- property interface members -- */

  size_t size() const noexcept { return base_class::getSize().size(); }

  /* Available only when: (AccessMode == access_mode::read ||
                           AccessMode == access_mode::read_write)
  if Dimensions == 1, CoordT = int
  if Dimensions == 2, CoordT = int2
  if Dimensions == 3, CoordT = int4 */
  template <
      typename CoordT,
      typename = std::enable_if_t<
          (AccessMode == access_mode::read ||
           AccessMode == access_mode::read_write) &&
          detail::IsValidUnsampledCoord2020DataT<Dimensions, CoordT>::value>>
  DataT read(const CoordT &Coords) const noexcept
#ifdef __SYCL_DEVICE_ONLY__
      ;
#else
  {
    // Host implementation is only available in host code. Device is not allowed
    // to use host_unsampled_image_accessor.
    return base_class::read<DataT>(Coords);
  }
#endif

  /* Available only when: (AccessMode == access_mode::write ||
                           AccessMode == access_mode::read_write)
  if Dimensions == 1, CoordT = int
  if Dimensions == 2, CoordT = int2
  if Dimensions == 3, CoordT = int4 */
  template <
      typename CoordT,
      typename = std::enable_if_t<
          (AccessMode == access_mode::write ||
           AccessMode == access_mode::read_write) &&
          detail::IsValidUnsampledCoord2020DataT<Dimensions, CoordT>::value>>
  void write(const CoordT &Coords, const DataT &Color) const
#ifdef __SYCL_DEVICE_ONLY__
      ;
#else
  {
    // Host implementation is only available in host code. Device is not allowed
    // to use host_unsampled_image_accessor.
    base_class::write<DataT>(Coords, Color);
  }
#endif

private:
  host_unsampled_image_accessor(
      const detail::UnsampledImageAccessorImplPtr &Impl)
      : base_class{Impl} {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

template <typename DataT, int Dimensions,
          image_target AccessTarget = image_target::device>
class __SYCL_EBO sampled_image_accessor :
#ifndef __SYCL_DEVICE_ONLY__
    private detail::SampledImageAccessorBaseHost,
#endif // __SYCL_DEVICE_ONLY__
    public detail::OwnerLessBase<
        sampled_image_accessor<DataT, Dimensions, AccessTarget>> {
  static_assert(std::is_same_v<DataT, int4> || std::is_same_v<DataT, uint4> ||
                    std::is_same_v<DataT, float4> ||
                    std::is_same_v<DataT, half4>,
                "The data type of an image accessor must be only int4, "
                "uint4, float4 or half4 from SYCL namespace");

#ifdef __SYCL_DEVICE_ONLY__
  char MPadding[sizeof(detail::SampledImageAccessorBaseHost)];
#else
  using host_base_class = detail::SampledImageAccessorBaseHost;
#endif // __SYCL_DEVICE_ONLY__

public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  sampled_image_accessor(
      sampled_image<Dimensions, AllocatorT> &ImageRef,
      handler &CommandGroupHandlerRef, const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
  {
    (void)ImageRef;
    (void)CommandGroupHandlerRef;
    (void)PropList;
    (void)CodeLoc;
  }
#else
      : host_base_class(detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                        detail::getSyclObjImpl(ImageRef).get(), Dimensions,
                        ImageRef.getElementSize(),
                        {ImageRef.getRowPitch(), ImageRef.getSlicePitch(), 0},
                        ImageRef.getChannelType(), ImageRef.getChannelOrder(),
                        ImageRef.getSampler(), PropList) {
    device Device = detail::getDeviceFromHandler(CommandGroupHandlerRef);
    // Avoid aspect::image warning.
    aspect ImageAspect = aspect::image;
    if (AccessTarget == image_target::device && !Device.has(ImageAspect))
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::feature_not_supported),
          "Device associated with command group handler does not have "
          "aspect::image.");

    detail::sampledImageConstructorNotification(
        detail::getSyclObjImpl(ImageRef).get(), this->impl.get(), AccessTarget,
        (const void *)typeid(DataT).name(), sizeof(DataT), CodeLoc);
    detail::associateWithHandler(CommandGroupHandlerRef, this, AccessTarget);
    GDBMethodsAnchor();
  }
#endif // __SYCL_DEVICE_ONLY__

  /* -- common interface members -- */

  sampled_image_accessor(const sampled_image_accessor &Rhs) = default;

  sampled_image_accessor(sampled_image_accessor &&Rhs) = default;

  sampled_image_accessor &
  operator=(const sampled_image_accessor &Rhs) = default;

  sampled_image_accessor &operator=(sampled_image_accessor &&Rhs) = default;

  ~sampled_image_accessor() = default;

#ifdef __SYCL_DEVICE_ONLY__
  bool operator==(const sampled_image_accessor &Rhs) const;
#else
  bool operator==(const sampled_image_accessor &Rhs) const {
    return Rhs.impl == impl;
  }
#endif // __SYCL_DEVICE_ONLY__

  bool operator!=(const sampled_image_accessor &Rhs) const {
    return !(Rhs == *this);
  }

  /* -- property interface members -- */

  size_t size() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    // Currently not reachable on device.
    return 0;
#else
    return host_base_class::getSize().size();
#endif // __SYCL_DEVICE_ONLY__
  }

  /* if Dimensions == 1, CoordT = float
     if Dimensions == 2, CoordT = float2
     if Dimensions == 3, CoordT = float4 */
  template <typename CoordT,
            typename = std::enable_if_t<detail::IsValidSampledCoord2020DataT<
                Dimensions, CoordT>::value>>
  DataT read(const CoordT &Coords) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    // Currently not reachable on device.
    std::ignore = Coords;
    return {0, 0, 0, 0};
#else
    return host_base_class::read<DataT>(Coords);
#endif // __SYCL_DEVICE_ONLY__
  }

private:
  sampled_image_accessor(const detail::SampledImageAccessorImplPtr &Impl)
#ifndef __SYCL_DEVICE_ONLY__
      : host_base_class{Impl}
#endif // __SYCL_DEVICE_ONLY__
  {
    std::ignore = Impl;
  }

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

template <typename DataT, int Dimensions>
class __SYCL_EBO host_sampled_image_accessor
    : private detail::SampledImageAccessorBaseHost,
      public detail::OwnerLessBase<
          host_sampled_image_accessor<DataT, Dimensions>> {
  static_assert(std::is_same_v<DataT, int4> || std::is_same_v<DataT, uint4> ||
                    std::is_same_v<DataT, float4> ||
                    std::is_same_v<DataT, half4>,
                "The data type of an image accessor must be only int4, "
                "uint4, float4 or half4 from SYCL namespace");

  using base_class = detail::SampledImageAccessorBaseHost;

public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  host_sampled_image_accessor(
      sampled_image<Dimensions, AllocatorT> &ImageRef,
      const property_list &PropList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : base_class(detail::convertToArrayOfN<3, 1>(ImageRef.get_range()),
                   detail::getSyclObjImpl(ImageRef).get(), Dimensions,
                   ImageRef.getElementSize(),
                   {ImageRef.getRowPitch(), ImageRef.getSlicePitch(), 0},
                   ImageRef.getChannelType(), ImageRef.getChannelOrder(),
                   ImageRef.getSampler(), PropList) {
    addHostSampledImageAccessorAndWait(base_class::impl.get());

    detail::sampledImageConstructorNotification(
        detail::getSyclObjImpl(ImageRef).get(), this->impl.get(), std::nullopt,
        (const void *)typeid(DataT).name(), sizeof(DataT), CodeLoc);
  }

  /* -- common interface members -- */

  host_sampled_image_accessor(const host_sampled_image_accessor &Rhs) = default;

  host_sampled_image_accessor(host_sampled_image_accessor &&Rhs) = default;

  host_sampled_image_accessor &
  operator=(const host_sampled_image_accessor &Rhs) = default;

  host_sampled_image_accessor &
  operator=(host_sampled_image_accessor &&Rhs) = default;

  ~host_sampled_image_accessor() = default;

  bool operator==(const host_sampled_image_accessor &Rhs) const {
    return Rhs.impl == impl;
  }
  bool operator!=(const host_sampled_image_accessor &Rhs) const {
    return !(Rhs == *this);
  }

  /* -- property interface members -- */

  size_t size() const noexcept { return base_class::getSize().size(); }

  /* if Dimensions == 1, CoordT = float
     if Dimensions == 2, CoordT = float2
     if Dimensions == 3, CoordT = float4 */
  template <typename CoordT,
            typename = std::enable_if_t<detail::IsValidSampledCoord2020DataT<
                Dimensions, CoordT>::value>>
  DataT read(const CoordT &Coords) const
#ifdef __SYCL_DEVICE_ONLY__
      ;
#else
  {
    // Host implementation is only available in host code. Device is not allowed
    // to use host_sampled_image_accessor.
    return base_class::read<DataT>(Coords);
  }
#endif

private:
  host_sampled_image_accessor(const detail::SampledImageAccessorImplPtr &Impl)
      : base_class{Impl} {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace _V1
} // namespace sycl

namespace std {
template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
          sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder>
struct hash<sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                           IsPlaceholder>> {
  using AccType = sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                                 IsPlaceholder>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    (void)A;
    return 0;
#else
    // getSyclObjImpl() here returns a pointer to either AccessorImplHost
    // or LocalAccessorImplHost depending on the AccessTarget.
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
#endif
  }
};

template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
struct hash<sycl::host_accessor<DataT, Dimensions, AccessMode>> {
  using AccType = sycl::host_accessor<DataT, Dimensions, AccessMode>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    (void)A;
    return 0;
#else
    // getSyclObjImpl() here returns a pointer to AccessorImplHost.
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
#endif
  }
};

template <typename DataT, int Dimensions>
struct hash<sycl::local_accessor<DataT, Dimensions>> {
  using AccType = sycl::local_accessor<DataT, Dimensions>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    (void)A;
    return 0;
#else
    // getSyclObjImpl() here returns a pointer to LocalAccessorImplHost.
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
#endif
  }
};

template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::image_target AccessTarget>
struct hash<sycl::unsampled_image_accessor<DataT, Dimensions, AccessMode,
                                           AccessTarget>> {
  using AccType = sycl::unsampled_image_accessor<DataT, Dimensions, AccessMode,
                                                 AccessTarget>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    (void)A;
    return 0;
#else
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
#endif
  }
};

template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
struct hash<
    sycl::host_unsampled_image_accessor<DataT, Dimensions, AccessMode>> {
  using AccType =
      sycl::host_unsampled_image_accessor<DataT, Dimensions, AccessMode>;

  size_t operator()(const AccType &A) const {
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
  }
};

template <typename DataT, int Dimensions, sycl::image_target AccessTarget>
struct hash<sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>> {
  using AccType = sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>;

  size_t operator()(const AccType &A) const {
#ifdef __SYCL_DEVICE_ONLY__
    // Hash is not supported on DEVICE. Just return 0 here.
    (void)A;
    return 0;
#else
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
#endif
  }
};

template <typename DataT, int Dimensions>
struct hash<sycl::host_sampled_image_accessor<DataT, Dimensions>> {
  using AccType = sycl::host_sampled_image_accessor<DataT, Dimensions>;

  size_t operator()(const AccType &A) const {
    auto AccImplPtr = sycl::detail::getSyclObjImpl(A);
    return hash<decltype(AccImplPtr)>()(AccImplPtr);
  }
};

} // namespace std
