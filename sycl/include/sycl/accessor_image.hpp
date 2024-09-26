#pragma once

#include <sycl/accessor.hpp>
#include <sycl/detail/image_accessor_util.hpp>
#include <sycl/device.hpp>
#include <sycl/image.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/detail/image_ocl_types.hpp>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {
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

void __SYCL_EXPORT unsampledImageConstructorNotification(
    void *ImageObj, void *AccessorObj,
    const std::optional<image_target> &Target, access::mode Mode,
    const void *Type, uint32_t ElemSize, const code_location &CodeLoc);

void __SYCL_EXPORT sampledImageConstructorNotification(
    void *ImageObj, void *AccessorObj,
    const std::optional<image_target> &Target, const void *Type,
    uint32_t ElemSize, const code_location &CodeLoc);

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
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

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
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

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
                (detail::is_genint_v<CoordT>) &&
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
      typename = std::enable_if_t<(Dims > 0) && (detail::is_genint_v<CoordT>) &&
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

  template <typename CoordT, typename CoordElemType = get_elem_type_t<CoordT>>
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
  template <typename Property> bool has_property() const noexcept {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template has_property<Property>();
#else
    return false;
#endif
  }
  template <typename Property> Property get_property() const {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template get_property<Property>();
#else
    return Property();
#endif
  }

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
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

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
  template <typename Property> bool has_property() const noexcept {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template has_property<Property>();
#else
    return false;
#endif
  }
  template <typename Property> Property get_property() const {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template get_property<Property>();
#else
    return Property();
#endif
  }

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
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

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
  template <typename Property> bool has_property() const noexcept {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template has_property<Property>();
#else
    return false;
#endif
  }
  template <typename Property> Property get_property() const {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template get_property<Property>();
#else
    return Property();
#endif
  }

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
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

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
  template <typename Property> bool has_property() const noexcept {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template has_property<Property>();
#else
    return false;
#endif
  }
  template <typename Property> Property get_property() const {
#ifndef __SYCL_DEVICE_ONLY__
    return getPropList().template get_property<Property>();
#else
    return Property();
#endif
  }

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
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace _V1
} // namespace sycl

namespace std {
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
