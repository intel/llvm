//==------------ accessor.hpp - SYCL standard header file ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_types.hpp>
#include <sycl/atomic.hpp>
#include <sycl/buffer.hpp>
#include <sycl/detail/accessor_iterator.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/handler_proxy.hpp>
#include <sycl/detail/image_accessor_util.hpp>
#include <sycl/detail/image_ocl_types.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/accessor_property_list.hpp>
#include <sycl/ext/oneapi/weak_object_base.hpp>
#include <sycl/id.hpp>
#include <sycl/image.hpp>
#include <sycl/pointers.hpp>
#include <sycl/properties/accessor_properties.hpp>
#include <sycl/properties/buffer_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/property_list_conversion.hpp>
#include <sycl/sampler.hpp>

#include <iterator>
#include <type_traits>

#include <utility>

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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

// To ensure loop unrolling is done when processing dimensions.
template <size_t... Inds, class F>
void dim_loop_impl(std::integer_sequence<size_t, Inds...>, F &&f) {
  (f(Inds), ...);
}

template <size_t count, class F> void dim_loop(F &&f) {
  dim_loop_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
}

void __SYCL_EXPORT constructorNotification(void *BufferObj, void *AccessorObj,
                                           access::target Target,
                                           access::mode Mode,
                                           const code_location &CodeLoc);

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

__SYCL_EXPORT device getDeviceFromHandler(handler &CommandGroupHandlerRef);

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT = ext::oneapi::accessor_property_list<>>
class accessor_common {
protected:
  constexpr static bool IsPlaceH = IsPlaceholder == access::placeholder::true_t;
  constexpr static access::address_space AS = TargetToAS<AccessTarget>::AS;

  constexpr static bool IsHostBuf = AccessTarget == access::target::host_buffer;

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

    template <int CurDims = SubDims,
              typename = detail::enable_if_t<(CurDims > 1)>>
    auto operator[](size_t Index) {
      MIDs[Dims - CurDims] = Index;
      return AccessorSubscript<CurDims - 1, AccType>(MAccessor, MIDs);
    }

    template <int CurDims = SubDims,
              typename = detail::enable_if_t<CurDims == 1 && IsAccessAnyWrite>>
    RefType operator[](size_t Index) const {
      MIDs[Dims - CurDims] = Index;
      return MAccessor[MIDs];
    }

    template <int CurDims = SubDims>
    typename detail::enable_if_t<CurDims == 1 && IsAccessAtomic,
                                 atomic<DataT, AS>>
    operator[](size_t Index) const {
      MIDs[Dims - CurDims] = Index;
      return MAccessor[MIDs];
    }

    template <int CurDims = SubDims,
              typename = detail::enable_if_t<CurDims == 1 && IsAccessReadOnly>>
    ConstRefType operator[](size_t Index) const {
      MIDs[Dims - SubDims] = Index;
      return MAccessor[MIDs];
    }
  };
};

template <typename MayBeTag1, typename MayBeTag2>
constexpr access::mode deduceAccessMode() {
  // property_list = {} is not properly detected by deduction guide,
  // when parameter is passed without curly braces: access(buffer, no_init)
  // thus simplest approach is to check 2 last arguments for being a tag
  if constexpr (std::is_same<MayBeTag1,
                             mode_tag_t<access::mode::read>>::value ||
                std::is_same<MayBeTag2,
                             mode_tag_t<access::mode::read>>::value) {
    return access::mode::read;
  }

  if constexpr (std::is_same<MayBeTag1,
                             mode_tag_t<access::mode::write>>::value ||
                std::is_same<MayBeTag2,
                             mode_tag_t<access::mode::write>>::value) {
    return access::mode::write;
  }

  if constexpr (
      std::is_same<MayBeTag1,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value ||
      std::is_same<MayBeTag2,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value) {
    return access::mode::read;
  }

  return access::mode::read_write;
}

template <typename MayBeTag1, typename MayBeTag2>
constexpr access::target deduceAccessTarget(access::target defaultTarget) {
  if constexpr (
      std::is_same<MayBeTag1,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value ||
      std::is_same<MayBeTag2,
                   mode_target_tag_t<access::mode::read,
                                     access::target::constant_buffer>>::value) {
    return access::target::constant_buffer;
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
  AccessorBaseHost(id<3> Offset, range<3> AccessRange, range<3> MemoryRange,
                   access::mode AccessMode, void *SYCLMemObject, int Dims,
                   int ElemSize, int OffsetInBytes = 0,
                   bool IsSubBuffer = false,
                   const property_list &PropertyList = {});

public:
  id<3> &getOffset();
  range<3> &getAccessRange();
  range<3> &getMemoryRange();
  void *getPtr();
  unsigned int getElemSize() const;

  const id<3> &getOffset() const;
  const range<3> &getAccessRange() const;
  const range<3> &getMemoryRange() const;
  void *getPtr() const;

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
  friend decltype(Obj::impl) getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  LocalAccessorImplPtr impl;
};

template <int Dim, typename T> struct IsValidCoordDataT;
template <typename T> struct IsValidCoordDataT<1, T> {
  constexpr static bool value =
      detail::is_contained<T, detail::type_list<cl_int, cl_float>>::type::value;
};
template <typename T> struct IsValidCoordDataT<2, T> {
  constexpr static bool value =
      detail::is_contained<T,
                           detail::type_list<cl_int2, cl_float2>>::type::value;
};
template <typename T> struct IsValidCoordDataT<3, T> {
  constexpr static bool value =
      detail::is_contained<T,
                           detail::type_list<cl_int4, cl_float4>>::type::value;
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

  static_assert(std::is_same<DataT, cl_int4>::value ||
                    std::is_same<DataT, cl_uint4>::value ||
                    std::is_same<DataT, cl_float4>::value ||
                    std::is_same<DataT, cl_half4>::value,
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

  template <typename Param>
  void checkDeviceFeatureSupported(const device &Device) {
    if (!Device.get_info<Param>())
      throw feature_not_supported("Images are not supported by this device.",
                                  PI_ERROR_INVALID_OPERATION);
  }

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
    throw runtime_error("image::getRangeInternal() is not implemented for host",
                        PI_ERROR_INVALID_OPERATION);
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
  image_accessor() : MImageObj() {}
#endif

  // Available only when: accessTarget == access::target::host_image
  // template <typename AllocatorT>
  // accessor(image<dimensions, AllocatorT> &imageRef);
  template <
      typename AllocatorT, int Dims = Dimensions,
      typename = detail::enable_if_t<(Dims > 0 && Dims <= 3) && IsHostImageAcc>>
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
                         Dimensions, ImageElementSize),
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
  template <
      typename AllocatorT, int Dims = Dimensions,
      typename = detail::enable_if_t<(Dims > 0 && Dims <= 3) && IsImageAcc>>
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
                         Dimensions, ImageElementSize),
        MImageCount(ImageRef.size()),
        MImgChannelOrder(ImageRef.getChannelOrder()),
        MImgChannelType(ImageRef.getChannelType()) {
    checkDeviceFeatureSupported<info::device::image_support>(
        getDeviceFromHandler(CommandGroupHandlerRef));
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

  template <int Dims = Dimensions, typename = detail::enable_if_t<Dims == 1>>
  range<1> get_range() const {
    cl_int Range = getRangeInternal();
    return range<1>(Range);
  }
  template <int Dims = Dimensions, typename = detail::enable_if_t<Dims == 2>>
  range<2> get_range() const {
    cl_int2 Range = getRangeInternal();
    return range<2>(Range[0], Range[1]);
  }
  template <int Dims = Dimensions, typename = detail::enable_if_t<Dims == 3>>
  range<3> get_range() const {
    cl_int3 Range = getRangeInternal();
    return range<3>(Range[0], Range[1], Range[2]);
  }

#else
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); };
  size_t size() const noexcept { return MImageCount; };

  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 0)>>
  range<Dims> get_range() const {
    return detail::convertToArrayOfN<Dims, 1>(getAccessRange());
  }

#endif

  // Available only when:
  // (accessTarget == access::target::image && accessMode == access::mode::read)
  // || (accessTarget == access::target::host_image && ( accessMode ==
  // access::mode::read || accessMode == access::mode::read_write))
  template <typename CoordT, int Dims = Dimensions,
            typename = detail::enable_if_t<
                (Dims > 0) && (IsValidCoordDataT<Dims, CoordT>::value) &&
                (detail::is_genint<CoordT>::value) &&
                ((IsImageAcc && IsImageAccessReadOnly) ||
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
            typename = detail::enable_if_t<
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
  template <typename CoordT, int Dims = Dimensions,
            typename = detail::enable_if_t<
                (Dims > 0) && (detail::is_genint<CoordT>::value) &&
                (IsValidCoordDataT<Dims, CoordT>::value) &&
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
            typename = detail::enable_if_t<
                (Dims > 0) && (IsValidCoordDataT<Dims, CoordT>::value)>>
  DataT read(const CoordT &Coords) const {
    return MBaseAcc.read(getAdjustedCoords(Coords));
  }

  template <typename CoordT, int Dims = Dimensions,
            typename = detail::enable_if_t<
                (Dims > 0) && IsValidCoordDataT<Dims, CoordT>::value>>
  DataT read(const CoordT &Coords, const sampler &Smpl) const {
    return MBaseAcc.read(getAdjustedCoords(Coords), Smpl);
  }

  template <typename CoordT, int Dims = Dimensions,
            typename = detail::enable_if_t<
                (Dims > 0) && IsValidCoordDataT<Dims, CoordT>::value>>
  void write(const CoordT &Coords, const DataT &Color) const {
    return MBaseAcc.write(getAdjustedCoords(Coords), Color);
  }

#ifdef __SYCL_DEVICE_ONLY__
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return get_range<Dimensions>().size(); }

  template <int Dims = Dimensions, typename = detail::enable_if_t<Dims == 1>>
  range<1> get_range() const {
    cl_int2 Count = MBaseAcc.getRangeInternal();
    return range<1>(Count.x());
  }
  template <int Dims = Dimensions, typename = detail::enable_if_t<Dims == 2>>
  range<2> get_range() const {
    cl_int3 Count = MBaseAcc.getRangeInternal();
    return range<2>(Count.x(), Count.y());
  }

#else

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept {
    return MBaseAcc.MImageCount / MBaseAcc.getAccessRange()[Dimensions];
  }

  template <int Dims = Dimensions,
            typename = detail::enable_if_t<(Dims == 1 || Dims == 2)>>
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
                 AccessTarget == access::target::host_buffer),
                "Expected buffer type");

  static_assert((AccessTarget == access::target::global_buffer ||
                 AccessTarget == access::target::host_buffer) ||
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
  template <int Dims>
  using AccessorSubscript =
      typename AccessorCommonT::template AccessorSubscript<Dims>;

  using ConcreteASPtrType = typename detail::DecoratedType<DataT, AS>::type *;

  using RefType = detail::const_if_const_AS<AS, DataT> &;
  using ConstRefType = const DataT &;
  using PtrType = detail::const_if_const_AS<AS, DataT> *;

  template <int Dims = Dimensions> size_t getLinearIndex(id<Dims> Id) const {

    size_t Result = 0;
    detail::dim_loop<Dims>([&, this](size_t I) {
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
      : std::bool_constant<std::is_same<T, DataT>::value && (Dims > 0) &&
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

  // TODO replace usages with getQualifiedPtr
  const ConcreteASPtrType getNativeImageObj() const { return MData; }

  void __init(ConcreteASPtrType Ptr, range<AdjustedDim> AccessRange,
              range<AdjustedDim> MemRange, id<AdjustedDim> Offset) {
    MData = Ptr;
    detail::dim_loop<AdjustedDim>([&, this](size_t I) {
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
  // TODO In ESIMD accessors usage is limited for now - access range, mem
  // range and offset are not supported.
  void __init_esimd(ConcreteASPtrType Ptr) { MData = Ptr; }

  ConcreteASPtrType getQualifiedPtr() const { return MData; }

  template <typename DataT_, int Dimensions_, access::mode AccessMode_,
            access::target AccessTarget_, access::placeholder IsPlaceholder_,
            typename PropertyListT_>
  friend class accessor;

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

  id<3> &getOffset() {
    if constexpr (IsHostBuf)
      return MAccData->MOffset;
    else
      return AccessorBaseHost::getOffset();
  }

  range<3> &getAccessRange() { return AccessorBaseHost::getAccessRange(); }
  range<3> &getMemoryRange() {
    if constexpr (IsHostBuf)
      return MAccData->MMemoryRange;
    else
      return AccessorBaseHost::getMemoryRange();
  }
  void *getPtr() { return AccessorBaseHost::getPtr(); }

  const id<3> &getOffset() const {
    if constexpr (IsHostBuf)
      return MAccData->MOffset;
    else
      return AccessorBaseHost::getOffset();
  }
  const range<3> &getAccessRange() const {
    return AccessorBaseHost::getAccessRange();
  }
  const range<3> &getMemoryRange() const {
    if constexpr (IsHostBuf)
      return MAccData->MMemoryRange;
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

  PtrType getQualifiedPtr() const {
    if constexpr (IsHostBuf)
      return reinterpret_cast<PtrType>(MAccData->MData);
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
            /*OffsetInBytes=*/0, /*IsSubBuffer=*/false, /*PropertyList=*/{}){};

#endif // __SYCL_DEVICE_ONLY__

private:
  friend class sycl::stream;
  friend class sycl::ext::intel::esimd::detail::AccessorPrivateProxy;

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  // 4.7.6.9.1. Interface for buffer command accessors
  // value_type is defined as const DataT for read_only accessors, DataT
  // otherwise
  using value_type = typename std::conditional<AccessMode == access_mode::read,
                                               const DataT, DataT>::type;
  using reference = DataT &;
  using const_reference = const DataT &;

  using iterator = typename detail::accessor_iterator<value_type, Dimensions>;
  using const_iterator =
      typename detail::accessor_iterator<const value_type, Dimensions>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;

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
  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                std::is_same<T, DataT>::value && Dims == 0 &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))> * = nullptr>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), range<1>{1}, BufferRef.get_range()) {
    (void)PropertyList;
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0}, detail::convertToArrayOfN<3, 1>(range<1>{1}),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), AdjustedDim, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    if (!IsPlaceH)
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
            typename detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                std::is_same<T, DataT>::value && Dims == 0 &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))> * = nullptr>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), range<1>{1}, BufferRef.get_range()) {
    (void)PropertyList;
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0}, detail::convertToArrayOfN<3, 1>(range<1>{1}),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), AdjustedDim, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    if (!IsPlaceH)
      addHostAccessorAndWait(AccessorBaseHost::impl.get());
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
#endif
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = typename detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                std::is_same<T, DataT>::value && (Dims == 0) &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), range<1>{1}, BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0}, detail::convertToArrayOfN<3, 1>(range<1>{1}),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
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
            typename = typename detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                std::is_same<T, DataT>::value && (Dims == 0) &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
  accessor(
      buffer<T, 1, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), range<1>{1}, BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0}, detail::convertToArrayOfN<3, 1>(range<1>{1}),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    detail::associateWithHandler(CommandGroupHandler, this, AccessTarget);
    initHostAcc();
    detail::constructorNotification(detail::getSyclObjImpl(BufferRef).get(),
                                    detail::AccessorBaseHost::impl.get(),
                                    AccessTarget, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<Dimensions>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    if (!IsPlaceH)
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
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<Dimensions>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    if (!IsPlaceH)
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, TagT,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, TagT,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
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
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(id<AdjustedDim>(), BufferRef.get_range(), BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(
            /*Offset=*/{0, 0, 0},
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
            getAdjustedMode(PropertyList),
            detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
            BufferRef.OffsetInBytes, BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                !IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      TagT, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                !IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT,
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                IsPlaceH && (IsGlobalBuf || IsConstantBuf)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      TagT, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                IsPlaceH && (IsGlobalBuf || IsConstantBuf)>>
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                 CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename... PropTypes,
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                !IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
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
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                !IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    if (BufferRef.isOutOfBounds(AccessOffset, AccessRange,
                                BufferRef.get_range()))
      throw sycl::invalid_object_error(
          "accessor with requested offset and range would exceed the bounds of "
          "the buffer",
          PI_ERROR_INVALID_VALUE);

    if (!IsPlaceH)
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
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                ((!IsPlaceH && IsHostBuf) ||
                 (IsPlaceH && (IsGlobalBuf || IsConstantBuf)))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset,
      const ext::oneapi::accessor_property_list<PropTypes...> &PropertyList =
          {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
    if (BufferRef.isOutOfBounds(AccessOffset, AccessRange,
                                BufferRef.get_range()))
      throw sycl::invalid_object_error(
          "accessor with requested offset and range would exceed the bounds of "
          "the buffer",
          PI_ERROR_INVALID_VALUE);

    if (!IsPlaceH)
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                IsPlaceH && (IsGlobalBuf || IsConstantBuf)>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, TagT, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : accessor(BufferRef, AccessRange, AccessOffset, PropertyList, CodeLoc) {
    adjustAccPropsInBuf(BufferRef);
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename TagT, typename... PropTypes,
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                IsPlaceH && (IsGlobalBuf || IsConstantBuf)>>
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
  accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AccessOffset, AccessRange, BufferRef.get_range()) {
    (void)CommandGroupHandler;
    (void)PropertyList;
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
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
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value &&
                (!IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf))>>
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
  }
#else
      : AccessorBaseHost(detail::convertToArrayOfN<3, 0>(AccessOffset),
                         detail::convertToArrayOfN<3, 1>(AccessRange),
                         detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                         getAdjustedMode(PropertyList),
                         detail::getSyclObjImpl(BufferRef).get(), Dimensions,
                         sizeof(DataT), BufferRef.OffsetInBytes,
                         BufferRef.IsSubBuffer, PropertyList) {
    preScreenAccessor(BufferRef.size(), PropertyList);
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
            typename = detail::enable_if_t<
                detail::IsRunTimePropertyListT<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                !IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
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
            typename = detail::enable_if_t<
                detail::IsCxPropertyList<PropertyListT>::value &&
                IsSameAsBuffer<T, Dims>::value && IsValidTag<TagT>::value &&
                !IsPlaceH && (IsGlobalBuf || IsConstantBuf || IsHostBuf)>>
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
      : impl(Other.impl)
#else
      : detail::AccessorBaseHost(Other)
#endif
  {
    static_assert(detail::IsCxPropertyList<PropertyListT>::value,
                  "Conversion is only available for accessor_property_list");
    static_assert(
        PropertyListT::template areSameCompileTimeProperties<NewPropsT...>(),
        "Compile-time-constant properties must be the same");
#ifndef __SYCL_DEVICE_ONLY__
    detail::constructorNotification(getMemoryObject(), impl.get(), AccessTarget,
                                    AccessMode, CodeLoc);
#endif
  }

  void swap(accessor &other) {
    std::swap(impl, other.impl);
#ifndef __SYCL_DEVICE_ONLY__
    std::swap(MAccData, other.MAccData);
#endif
  }

  constexpr bool is_placeholder() const { return IsPlaceH; }

  size_t get_size() const { return getAccessRange().size() * sizeof(DataT); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept { return getAccessRange().size(); }

  size_t byte_size() const noexcept { return size() * sizeof(DataT); }

  size_t max_size() const noexcept {
    return empty() ? 0 : (std::numeric_limits<difference_type>::max)();
  }

  bool empty() const noexcept { return size() == 0; }

  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 0)>>
  range<Dimensions> get_range() const {
    return detail::convertToArrayOfN<Dimensions, 1>(getAccessRange());
  }

  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 0)>>
  id<Dimensions> get_offset() const {
    static_assert(
        !(PropertyListT::template has_property<
            sycl::ext::oneapi::property::no_offset>()),
        "Accessor has no_offset property, get_offset() can not be used");
    return detail::convertToArrayOfN<Dimensions, 0>(getOffset());
  }

  template <int Dims = Dimensions, typename RefT = RefType,
            typename = detail::enable_if_t<Dims == 0 && IsAccessAnyWrite &&
                                           !std::is_const<RefT>::value>>
  operator RefType() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return *(getQualifiedPtr() + LinearIndex);
  }

  template <int Dims = Dimensions,
            typename = detail::enable_if_t<Dims == 0 && IsAccessReadOnly>>
  operator ConstRefType() const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return *(getQualifiedPtr() + LinearIndex);
  }

  template <int Dims = Dimensions,
            typename = detail::enable_if_t<(Dims > 0) && IsAccessAnyWrite>>
  RefType operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions>
  typename detail::enable_if_t<(Dims > 0) && IsAccessReadOnly, ConstRefType>
  operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions>
  operator typename detail::enable_if_t<Dims == 0 &&
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
  typename detail::enable_if_t<(Dims > 0) && AccessMode == access::mode::atomic,
                               atomic<DataT, AS>>
  operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }

  template <int Dims = Dimensions>
  typename detail::enable_if_t<Dims == 1 && AccessMode == access::mode::atomic,
                               atomic<DataT, AS>>
  operator[](size_t Index) const {
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>(Index));
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }
  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 1)>>
  auto operator[](size_t Index) const {
    return AccessorSubscript<Dims - 1>(*this, Index);
  }

  template <access::target AccessTarget_ = AccessTarget,
            typename = detail::enable_if_t<AccessTarget_ ==
                                           access::target::host_buffer>>
  DataT *get_pointer() const {
    return getPointerAdjusted();
  }

  template <
      access::target AccessTarget_ = AccessTarget,
      typename = detail::enable_if_t<AccessTarget_ == access::target::device>>
  global_ptr<DataT> get_pointer() const {
    return global_ptr<DataT>(getPointerAdjusted());
  }

  template <access::target AccessTarget_ = AccessTarget,
            typename = detail::enable_if_t<AccessTarget_ ==
                                           access::target::constant_buffer>>
  constant_ptr<DataT> get_pointer() const {
    return constant_ptr<DataT>(getPointerAdjusted());
  }

  // accessor::has_property for runtime properties is only available in host
  // code. This restriction is not listed in the core spec and will be added in
  // future versions.
  template <typename Property>
  typename sycl::detail::enable_if_t<
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
            typename = typename sycl::detail::enable_if_t<
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
        detail::convertToArrayOfN<Dimensions, 1>(getMemoryRange()), get_range(),
        get_offset());
  }

  iterator end() const noexcept {
    return iterator::getEnd(
        get_pointer(),
        detail::convertToArrayOfN<Dimensions, 1>(getMemoryRange()), get_range(),
        get_offset());
  }

  const_iterator cbegin() const noexcept {
    return const_iterator::getBegin(
        get_pointer(),
        detail::convertToArrayOfN<Dimensions, 1>(getMemoryRange()), get_range(),
        get_offset());
  }

  const_iterator cend() const noexcept {
    return const_iterator::getEnd(
        get_pointer(),
        detail::convertToArrayOfN<Dimensions, 1>(getMemoryRange()), get_range(),
        get_offset());
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
#ifdef __SYCL_DEVICE_ONLY__
  size_t getTotalOffset() const {
    size_t TotalOffset = 0;
    detail::dim_loop<Dimensions>([&, this](size_t I) {
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
  auto getPointerAdjusted() const {
#ifdef __SYCL_DEVICE_ONLY__
    return getQualifiedPtr() - getTotalOffset();
#else
    return getQualifiedPtr();
#endif
  }

  void preScreenAccessor(const size_t elemInBuffer,
                         const PropertyListT &PropertyList) {
    // check device accessor buffer size
    if (!IsHostBuf && elemInBuffer == 0)
      throw sycl::invalid_object_error(
          "SYCL buffer size is zero. To create a device accessor, SYCL "
          "buffer size must be greater than zero.",
          PI_ERROR_INVALID_VALUE);

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
  using AccessorCommonT::IsAccessAnyWrite;
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
    detail::dim_loop<AdjustedDim>(
        [&, this](size_t I) { getSize()[I] = AccessRange[I]; });
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
    for (int I = 0; I < Dims; ++I)
      Result = Result * getSize()[I] + Id[I];
    return Result;
  }

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  template <int Dims = Dimensions, typename = detail::enable_if_t<Dims == 0>>
  local_accessor_base(handler &, const detail::code_location CodeLoc =
                                     detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(range<AdjustedDim>{1}){}
#else
      : LocalAccessorBaseHost(range<3>{1, 1, 1}, AdjustedDim, sizeof(DataT)) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

        template <int Dims = Dimensions,
                  typename = detail::enable_if_t<Dims == 0>>
        local_accessor_base(handler &, const property_list &propList,
                            const detail::code_location CodeLoc =
                                detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(range<AdjustedDim>{1}) {
    (void)propList;
  }
#else
      : LocalAccessorBaseHost(range<3>{1, 1, 1}, AdjustedDim, sizeof(DataT),
                              propList) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 0)>>
  local_accessor_base(
      range<Dimensions> AllocationSize, handler &,
      const detail::code_location CodeLoc = detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AllocationSize){}
#else
      : LocalAccessorBaseHost(detail::convertToArrayOfN<3, 1>(AllocationSize),
                              AdjustedDim, sizeof(DataT)) {
    detail::constructorNotification(nullptr, LocalAccessorBaseHost::impl.get(),
                                    access::target::local, AccessMode, CodeLoc);
    GDBMethodsAnchor();
  }
#endif

        template <int Dims = Dimensions,
                  typename = detail::enable_if_t<(Dims > 0)>>
        local_accessor_base(range<Dimensions> AllocationSize, handler &,
                            const property_list &propList,
                            const detail::code_location CodeLoc =
                                detail::code_location::current())
#ifdef __SYCL_DEVICE_ONLY__
      : impl(AllocationSize) {
    (void)propList;
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

  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 0)>>
  range<Dims> get_range() const {
    return detail::convertToArrayOfN<Dims, 1>(getSize());
  }

  template <int Dims = Dimensions,
            typename = detail::enable_if_t<Dims == 0 && IsAccessAnyWrite>>
  operator RefType() const {
    return *getQualifiedPtr();
  }

  template <int Dims = Dimensions,
            typename = detail::enable_if_t<(Dims > 0) && IsAccessAnyWrite>>
  RefType operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return getQualifiedPtr()[LinearIndex];
  }

  template <int Dims = Dimensions,
            typename = detail::enable_if_t<Dims == 1 && IsAccessAnyWrite>>
  RefType operator[](size_t Index) const {
    return getQualifiedPtr()[Index];
  }

  template <int Dims = Dimensions>
  operator typename detail::enable_if_t<
      Dims == 0 && AccessMode == access::mode::atomic, atomic<DataT, AS>>()
      const {
    return atomic<DataT, AS>(
        multi_ptr<DataT, AS, access::decorated::yes>(getQualifiedPtr()));
  }

  template <int Dims = Dimensions>
  typename detail::enable_if_t<(Dims > 0) && AccessMode == access::mode::atomic,
                               atomic<DataT, AS>>
  operator[](id<Dimensions> Index) const {
    const size_t LinearIndex = getLinearIndex(Index);
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + LinearIndex));
  }

  template <int Dims = Dimensions>
  typename detail::enable_if_t<Dims == 1 && AccessMode == access::mode::atomic,
                               atomic<DataT, AS>>
  operator[](size_t Index) const {
    return atomic<DataT, AS>(multi_ptr<DataT, AS, access::decorated::yes>(
        getQualifiedPtr() + Index));
  }

  template <int Dims = Dimensions, typename = detail::enable_if_t<(Dims > 1)>>
  typename AccessorCommonT::template AccessorSubscript<
      Dims - 1,
      local_accessor_base<DataT, Dimensions, AccessMode, IsPlaceholder>>
  operator[](size_t Index) const {
    return AccessorSubscript<Dims - 1>(*this, Index);
  }

  local_ptr<DataT> get_pointer() const {
    return local_ptr<DataT>(getQualifiedPtr());
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
    : public local_accessor_base<DataT, Dimensions, access::mode::read_write,
                                 access::placeholder::false_t>,
      public detail::OwnerLessBase<local_accessor<DataT, Dimensions>> {

  using local_acc =
      local_accessor_base<DataT, Dimensions, access::mode::read_write,
                          access::placeholder::false_t>;

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

public:
  // Default constructor for objects later initialized with __init member.
  local_accessor() {
    local_acc::impl = detail::InitializedVal<local_acc::AdjustedDim,
                                             range>::template get<0>();
  }

#else
  local_accessor(const detail::AccessorImplPtr &Impl) : local_acc{Impl} {}
#endif

public:
  using value_type = DataT;
  using iterator = value_type *;
  using const_iterator = const value_type *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;

  void swap(local_accessor &other) { std::swap(this->impl, other.impl); }

  size_t byte_size() const noexcept { return this->size() * sizeof(DataT); }

  size_t max_size() const noexcept {
    return empty() ? 0 : (std::numeric_limits<difference_type>::max)();
  }

  bool empty() const noexcept { return this->size() == 0; }

  iterator begin() const noexcept {
    return &this->operator[](id<Dimensions>());
  }
  iterator end() const noexcept { return begin() + this->size(); }

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
                      access::placeholder::false_t>,
      public detail::OwnerLessBase<
          host_accessor<DataT, Dimensions, AccessMode>> {
protected:
  using AccessorT = accessor<DataT, Dimensions, AccessMode, target::host_buffer,
                             access::placeholder::false_t>;

  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;

  template <typename T, int Dims>
  struct IsSameAsBuffer
      : std::bool_constant<std::is_same<T, DataT>::value && (Dims > 0) &&
                           (Dims == Dimensions)> {};

  void
  __init(typename accessor<DataT, Dimensions, AccessMode, target::host_buffer,
                           access::placeholder::false_t>::ConcreteASPtrType Ptr,
         range<AdjustedDim> AccessRange, range<AdjustedDim> MemRange,
         id<AdjustedDim> Offset) {
    AccessorT::__init(Ptr, AccessRange, MemRange, Offset);
  }

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
            typename = typename detail::enable_if_t<
                std::is_same<T, DataT>::value && Dims == 0>>
  host_accessor(
      buffer<T, 1, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, mode_tag_t<AccessMode>,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      mode_tag_t<AccessMode>, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, CommandGroupHandler, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      mode_tag_t<AccessMode>, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, AccessRange, {}, PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, CommandGroupHandler, AccessRange, {}, PropertyList,
                  CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, mode_tag_t<AccessMode>,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, CommandGroupHandler, AccessRange, {},
                      PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, AccessRange, AccessOffset, PropertyList, CodeLoc) {
  }

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, range<Dimensions> AccessRange,
      id<Dimensions> AccessOffset, mode_tag_t<AccessMode>,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, AccessRange, AccessOffset, PropertyList,
                      CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : AccessorT(BufferRef, CommandGroupHandler, AccessRange, AccessOffset,
                  PropertyList, CodeLoc) {}

  template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>::value>>
  host_accessor(
      buffer<T, Dims, AllocatorT> &BufferRef, handler &CommandGroupHandler,
      range<Dimensions> AccessRange, id<Dimensions> AccessOffset,
      mode_tag_t<AccessMode>, const property_list &PropertyList = {},
      const detail::code_location CodeLoc = detail::code_location::current())
      : host_accessor(BufferRef, CommandGroupHandler, AccessRange, AccessOffset,
                      PropertyList, CodeLoc) {}
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
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

} // namespace std
