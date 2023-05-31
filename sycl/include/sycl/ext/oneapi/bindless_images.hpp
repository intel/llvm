//==----------- bindless_images.hpp --- SYCL bindless images ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/queue.hpp>

#include <sycl/ext/oneapi/bindless_images_descriptor.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/ext/oneapi/bindless_images_memory.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
/// Opaque unsampled image handle type.
struct unsampled_image_handle {
  using raw_handle_type = unsigned long;
  raw_handle_type raw_handle;
};
/// Opaque sampled image handle type.
struct sampled_image_handle {
  using raw_handle_type = unsigned long;
  raw_handle_type raw_handle;
};

/**
 *  @brief   Allocate image memory based on image_descriptor
 *
 *  @param   syclContext The context in which we create our memory handle
 *  @param   desc The image descriptor
 *  @return  Memory handle to allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle alloc_image_mem(const sycl::context &syclContext,
                                               const image_descriptor &desc);

/**
 *  @brief   Free image memory
 *
 *  @param   syclContext The context in which we created our memory handle
 *  @param   handle Memory handle to allocated memory on the device
 */
__SYCL_EXPORT void free_image_mem(const sycl::context &syclContext,
                                  image_mem_handle handle);

/**
 *  @brief   Allocate mipmap memory based on image_descriptor
 *
 *  @param   syclContext The context in which we create our memory handle
 *  @param   desc The image descriptor
 *  @return  Memory handle to allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle alloc_mipmap_mem(
    const sycl::context &syclContext, const image_descriptor &desc);

/**
 *  @brief   Free mipmap memory
 *
 *  @param   syclContext The context in which we created our memory handle
 *  @param   handle The mipmap memory handle
 */
__SYCL_EXPORT void free_mipmap_mem(const sycl::context &syclContext,
                                   image_mem_handle handle);

/**
 *  @brief   Retrieve the memory handle to an individual mipmap image
 *
 *  @param   syclContext The context in which we created our memory handle
 *  @param   mipMem The memory handle to the mipmapped array
 *  @param   level The requested level of the mipmap
 *  @return  Memory handle to the individual mipmap image
 */
__SYCL_EXPORT image_mem_handle get_mip_level(const sycl::context &syclContext,
                                             const image_mem &mipMem,
                                             const unsigned int level);

/**
 *  @brief   Import external memory taking an external memory handle (the type
 *           of which is dependent on the OS & external API) and return an
 *           interop memory handle
 *
 *  @tparam  ExternalMemHandleType Handle type describing external memory handle
 *  @param   syclContext The context in which we create our interop memory
 *           handle
 *  @param   externalMem External memory descriptor
 *  @return  Interop memory handle to the external memory
 */
template <typename ExternalMemHandleType>
__SYCL_EXPORT interop_mem_handle import_external_memory(
    const sycl::context &syclContext,
    external_mem_descriptor<ExternalMemHandleType> externalMem);

/**
 *  @brief   Maps an interop memory handle to an image memory handle (which may
 *           have a device optimized memory layout)
 *
 *  @param   syclContext The conext in which we create our image memory handle
 *  @param   memHandle   Interop memory handle
 *  @param   desc        The image descriptor
 *  @return  Memory handle to externally allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle map_external_memory_array(
    const sycl::context &syclContext, interop_mem_handle memHandle,
    const image_descriptor &desc);

/**
 *  @brief   Import external semaphore taking an external semaphore handle (the
 *           type of which is dependent on the OS & external API)
 *
 *  @tparam  ExternalSemaphoreHandleType Handle type describing external
 *           semaphore handle
 *  @param   syclContext The context in which we create our interop semaphore
 *           handle
 *  @param   externalSemaphoreDesc External semaphore descriptor
 *  @return  Interop semaphore handle to the external semaphore
 */
template <typename ExternalSemaphoreHandleType>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    const sycl::context &syclContext,
    external_semaphore_descriptor<ExternalSemaphoreHandleType>
        externalSemaphoreDesc);

/**
 *  @brief   Instruct the queue with a non-blocking wait on an external
 *           semaphore
 *
 *  @param   queue The queue instructed to wait
 *  @param   semaphoreHandle The interop semaphore handle to wait on
 */
__SYCL_EXPORT sycl::event
wait_external_semaphore(const sycl::queue &queue,
                        interop_semaphore_handle semaphoreHandle);

/**
 *  @brief   Instruct the queue to signal the external semaphore once all
 *           previous commands have completed execution
 *
 *  @param   queue The queue instructed to signal
 *  @param   semaphoreHandle The interop semaphore handle to signal
 */
__SYCL_EXPORT sycl::event
signal_external_semaphore(const sycl::queue &queue,
                          interop_semaphore_handle semaphoreHandle);

/**
 *  @brief   Destroy the external semaphore handle
 *
 *  @param   syclContext The context in which the interop semaphore handle was
 *           created
 *  @param   semaphoreHandle The interop semaphore handle to destroy
 */
__SYCL_EXPORT void
destroy_external_semaphore(const sycl::context &syclContext,
                           interop_semaphore_handle semaphoreHandle);

/**
 *  @brief   Release external memory
 *
 *  @param   syclContext The context in which the interop memory handle was
 * created
 *  @param   interopHandle The interop memory handle to release
 */
__SYCL_EXPORT void release_external_memory(const sycl::context &syclContext,
                                           interop_mem_handle interopHandle);

/**
 *  @brief   Create an image and return the device image handle
 *
 *  @param   syclContext The context in which we create our image handle
 *  @param   memHandle Device memory handle wrapper for allocated image memory
 *  @param   desc The image descriptor
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT unsampled_image_handle
create_image(const sycl::context &syclContext, image_mem &memHandle,
             const image_descriptor &desc);

/**
 *  @brief   Create an image and return the device image handle
 *
 *  @param   syclContext The context in which we create our image handle
 *  @param   memHandle Device memory handle for allocated image memory
 *  @param   desc The image descriptor
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT unsampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             const image_descriptor &desc);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   syclContext The context in which we create our image handle
 *  @param   imgMem Device memory pointer to allocated image memory
 *  @param   pitch The allocation pitch value
 *  @param   sampler SYCL sampler to sample the image
 *  @param   desc The image descriptor
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, void *imgMem, size_t pitch,
             const sampler &sampler, const image_descriptor &desc);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   syclContext The context in which we create our image handle
 *  @param   memHandle Device memory handle wrapper for allocated image memory
 *  @param   sampler SYCL sampler to sample the image
 *  @param   desc The image descriptor
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, image_mem &memHandle,
             const sampler &sampler, const image_descriptor &desc);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   syclContext The context in which we create our image handle
 *  @param   memHandle Device memory handle for allocated image memory
 *  @param   sampler SYCL sampler to sample the image
 *  @param   desc The image descriptor
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             const sampler &sampler, const image_descriptor &desc);

/**
 *  @brief   Destroy an unsampled image handle. Does not free memory backing the
 *           handle
 *
 *  @param   syclContext The context in which we created our image handle
 *  @param   imageHandle The unsampled image handle to destroy
 **/
__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        unsampled_image_handle &imageHandle);

/**
 *  @brief   Destroy a sampled image handle. Does not free memory backing the
 *           handle
 *
 *  @param   syclContext The context in which we created our image handle
 *  @param   imageHandle The sampled image handle to destroy
 **/
__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        sampled_image_handle &imageHandle);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   widthInBytes The width of the image in bytes
 *  @param   height The height of the image in elements
 *  @param   elementSizeBytes Number of bytes of a singular image element
 *  @param   queue The queue
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const queue &queue);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   widthInBytes The width of the image in bytes
 *  @param   height The height of the image in elements
 *  @param   elementSizeBytes Number of bytes of a singular image element
 *  @param   device The device
 *  @param   syclContext The context
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const device &device,
                                         const sycl::context &syclContext);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   desc The image descriptor
 *  @param   queue The queue
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         const image_descriptor &desc,
                                         const queue &queue);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   desc The image descriptor
 *  @param   device The device
 *  @param   syclContext The context
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         const image_descriptor &desc,
                                         const device &device,
                                         const sycl::context &syclContext);

/**
 *  @brief   Get the range that describes the image's dimensions
 *
 *  @param   context The context in which we created our image memory handle
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @return  sycl range describing image's dimensions
 */
sycl::range<3> get_image_range(const sycl::context &context,
                               const image_mem_handle memHandle);

/**
 *  @brief   Get the channel type that describes the image memory
 *
 *  @param   context The context in which we created our image memory handle
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @return  sycl image channel type that describes the image
 */
sycl::image_channel_type
get_image_channel_type(const sycl::context &context,
                       const image_mem_handle memHandle);

/**
 *  @brief   Get the number of channels that describes the image memory
 *
 *  @param   context The context in which we created our image memory handle
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @return  The number of channels describing the image
 */
unsigned int get_image_num_channels(const sycl::context &context,
                                    const image_mem_handle memHandle);

/**
 *  @brief   If any exist, get the backend specific flags used on creation of
 *           the image memory handle
 *
 *  @param   context The context in which we created our image memory handle
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @return  The backend specific flags used to create the image memory
 */
unsigned int get_image_flags(const sycl::context &context,
                             const image_mem_handle memHandle);

namespace detail {
// Get the number of coordinates
template <typename CoordT> constexpr size_t coord_size() {
  if constexpr (std::is_scalar<CoordT>::value) {
    return 1;
  } else {
    return CoordT::size();
  }
}
} // namespace detail

/**
 *  @brief   Read an unsampled image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  CoordT The input coordinate type. e.g. int, int2, or int4 for
 *           1D, 2D, and 3D respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to fetch image data
 *  @return  Image data
 *
 *  __NVPTX__: Name mangling info
 *             Cuda surfaces require integer coords (by bytes)
 *             Cuda textures require float coords (by element or normalized)
 *             The name mangling should therefore not interfere with one
 *             another
 */
template <typename DataT, typename CoordT>
DataT read_image(const unsampled_image_handle &imageHandle,
                 const CoordT &coords) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageRead<DataT, uint64_t, CoordT>(imageHandle.raw_handle,
                                                        coords);
#else
    // TODO: add SPIRV part for unsampled image read
#endif
#else
    assert(false); // Bindless images not yet implemented on host
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate to be have 1, 2, or 4 components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

/**
 *  @brief   Read a sampled image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  CoordT The input coordinate type. e.g. float, float2, or float4 for
 *           1D, 2D, and 3D respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to fetch image data
 *  @return  Sampled image data
 *
 *  __NVPTX__: Name mangling info
 *             Cuda surfaces require integer coords (by bytes)
 *             Cuda textures require float coords (by element or normalized)
 *             The name mangling should therefore not interfere with one
 *             another
 */
template <typename DataT, typename CoordT>
DataT read_image(const sampled_image_handle &imageHandle,
                 const CoordT &coords) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageRead<DataT, uint64_t, CoordT>(imageHandle.raw_handle,
                                                        coords);
#else
    // TODO: add SPIRV part for sampled image read
#endif
#else
    assert(false); // Bindless images not yet implemented on host.
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate to be have 1, 2, or 4 components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

/**
 *  @brief   Read a mipmap image using its handle with LOD filtering
 *
 *  @tparam  DataT The return type
 *  @tparam  CoordT The input coordinate type. e.g. float, float2, or float4 for
 *           1D, 2D, and 3D respectively
 *  @param   imageHandle The mipmap image handle
 *  @param   coords The coordinates at which to fetch mipmap image data
 *  @param   level The mipmap level at which to sample
 *  @return  Mipmap image data with LOD filtering
 */
template <typename DataT, typename CoordT>
DataT read_image(const sampled_image_handle &imageHandle, const CoordT &coords,
                 const float level) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageReadLod<DataT, uint64_t, CoordT>(
        imageHandle.raw_handle, coords, level);
#else
    // TODO: add SPIRV for mipmap level read
#endif
#else
    assert(false); // Bindless images not yet implemented on host
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate to be have 1, 2, or 4 components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

/**
 *  @brief   Read a mipmap image using its handle with anisotropic filtering
 *
 *  @tparam  DataT The return type
 *  @tparam  CoordT The input coordinate type. e.g. float, float2, or float4 for
 *           1D, 2D, and 3D respectively
 *  @param   imageHandle The mipmap image handle
 *  @param   coords The coordinates at which to fetch mipmap image data
 *  @param   dX Screen space gradient in the x dimension
 *  @param   dY Screen space gradient in the y dimension
 *  @return  Mipmap image data with anisotropic filtering
 */
template <typename DataT, typename CoordT>
DataT read_image(const sampled_image_handle &imageHandle, const CoordT &coords,
                 const CoordT &dX, const CoordT &dY) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageReadGrad<DataT, uint64_t, CoordT>(
        imageHandle.raw_handle, coords, dX, dY);
#else
    // TODO: add SPIRV part for mipmap grad read
#endif
#else
    assert(false); // Bindless images not yet implemented on host
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate and gradient to have 1, 2, or 4 "
                  "components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

/**
 *  @brief   Write to an unsampled image using its handle
 *
 *  @tparam  DataT The data type to write
 *  @tparam  CoordT The input coordinate type. e.g. int, int2, or int4 for
 *           1D, 2D, and 3D respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to write image data
 */
template <typename DataT, typename CoordT>
void write_image(const unsampled_image_handle &imageHandle,
                 const CoordT &Coords, const DataT &Color) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    __invoke__ImageWrite<uint64_t, CoordT, DataT>(
        (uint64_t)imageHandle.raw_handle, Coords, Color);
#else
    // TODO: add SPIRV part for unsampled image write
#endif
#else
    assert(false); // Bindless images not yet implemented on host
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate to be have 1, 2, or 4 components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
