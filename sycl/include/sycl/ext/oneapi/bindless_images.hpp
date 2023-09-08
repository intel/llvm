//==----------- bindless_images.hpp --- SYCL bindless images ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>                               // for context
#include <sycl/detail/export.hpp>                         // for __SYCL_EXPORT
#include <sycl/detail/pi.h>                               // for pi_uint64
#include <sycl/device.hpp>                                // for device
#include <sycl/ext/oneapi/bindless_images_descriptor.hpp> // for image_desc...
#include <sycl/ext/oneapi/bindless_images_interop.hpp>    // for interop_me...
#include <sycl/ext/oneapi/bindless_images_memory.hpp>     // for image_mem_...
#include <sycl/ext/oneapi/bindless_images_sampler.hpp>    // for bindless_i...
#include <sycl/image.hpp>                                 // for image_chan...
#include <sycl/queue.hpp>                                 // for queue
#include <sycl/range.hpp>                                 // for range

#include <assert.h>    // for assert
#include <stddef.h>    // for size_t
#include <type_traits> // for is_scalar

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

/// Opaque unsampled image handle type.
struct unsampled_image_handle {
  using raw_handle_type = pi_uint64;
  raw_handle_type raw_handle;
};
/// Opaque sampled image handle type.
struct sampled_image_handle {
  using raw_handle_type = pi_uint64;
  raw_handle_type raw_handle;
};

/**
 *  @brief   Allocate image memory based on image_descriptor
 *
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we create our memory handle
 *  @param   syclContext The context in which we create our memory handle
 *  @return  Memory handle to allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle
alloc_image_mem(const image_descriptor &desc, const sycl::device &syclDevice,
                const sycl::context &syclContext);

/**
 *  @brief   Allocate image memory based on image_descriptor
 *
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue in which we create our memory handle
 *  @return  Memory handle to allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle alloc_image_mem(const image_descriptor &desc,
                                               const sycl::queue &syclQueue);

/**
 *  @brief   Free image memory
 *
 *  @param   handle Memory handle to allocated memory on the device
 *  @param   syclDevice The device in which we create our memory handle
 *  @param   syclContext The context in which we created our memory handle
 */
__SYCL_EXPORT void free_image_mem(image_mem_handle handle,
                                  const sycl::device &syclDevice,
                                  const sycl::context &syclContext);

/**
 *  @brief   Free image memory
 *
 *  @param   handle Memory handle to allocated memory on the device
 *  @param   syclQueue The queue in which we create our memory handle
 */
__SYCL_EXPORT void free_image_mem(image_mem_handle handle,
                                  const sycl::device &syclQueue);

/**
 *  @brief   Allocate mipmap memory based on image_descriptor
 *
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we create our memory handle
 *  @param   syclContext The context in which we create our memory handle
 *  @return  Memory handle to allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle
alloc_mipmap_mem(const image_descriptor &desc, const sycl::device &syclDevice,
                 const sycl::context &syclContext);

/**
 *  @brief   Allocate mipmap memory based on image_descriptor
 *
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue in which we create our memory handle
 *  @return  Memory handle to allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle alloc_mipmap_mem(const image_descriptor &desc,
                                                const sycl::device &syclQueue);

/**
 *  @brief   Free mipmap memory
 *
 *  @param   handle The mipmap memory handle
 *  @param   syclDevice The device in which we created our memory handle
 *  @param   syclContext The context in which we created our memory handle
 */
__SYCL_EXPORT void free_mipmap_mem(image_mem_handle handle,
                                   const sycl::device &syclDevice,
                                   const sycl::context &syclContext);

/**
 *  @brief   Free mipmap memory
 *
 *  @param   handle The mipmap memory handle
 *  @param   syclQueue The queue in which we created our memory handle
 */
__SYCL_EXPORT void free_mipmap_mem(image_mem_handle handle,
                                   const sycl::queue &syclQueue);

/**
 *  @brief   Retrieve the memory handle to an individual mipmap image
 *
 *  @param   mipMem The memory handle to the mipmapped array
 *  @param   level The requested level of the mipmap
 *  @param   syclDevice The device in which we created our memory handle
 *  @param   syclContext The context in which we created our memory handle
 *  @return  Memory handle to the individual mipmap image
 */
__SYCL_EXPORT image_mem_handle get_mip_level_mem_handle(
    const image_mem_handle mipMem, const unsigned int level,
    const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Retrieve the memory handle to an individual mipmap image
 *
 *  @param   mipMem The memory handle to the mipmapped array
 *  @param   level The requested level of the mipmap
 *  @param   syclQueue The queue in which we created our memory handle
 *  @return  Memory handle to the individual mipmap image
 */
__SYCL_EXPORT image_mem_handle get_mip_level_mem_handle(
    const image_mem_handle mipMem, const unsigned int level,
    const sycl::device &syclQueue);

/**
 *  @brief   Import external memory taking an external memory handle (the type
 *           of which is dependent on the OS & external API) and return an
 *           interop memory handle
 *
 *  @tparam  ExternalMemHandleType Handle type describing external memory handle
 *  @param   externalMem External memory descriptor
 *  @param   syclDevice The device in which we create our interop memory
 *  @param   syclContext The context in which we create our interop memory
 *           handle
 *  @return  Interop memory handle to the external memory
 */
template <typename ExternalMemHandleType>
__SYCL_EXPORT interop_mem_handle import_external_memory(
    external_mem_descriptor<ExternalMemHandleType> externalMem,
    const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Import external memory taking an external memory handle (the type
 *           of which is dependent on the OS & external API) and return an
 *           interop memory handle
 *
 *  @tparam  ExternalMemHandleType Handle type describing external memory handle
 *  @param   externalMem External memory descriptor
 *  @param   syclQueue The queue in which we create our interop memory
 *           handle
 *  @return  Interop memory handle to the external memory
 */
template <typename ExternalMemHandleType>
__SYCL_EXPORT interop_mem_handle import_external_memory(
    external_mem_descriptor<ExternalMemHandleType> externalMem,
    const sycl::queue &syclQueue);

/**
 *  @brief   Maps an interop memory handle to an image memory handle (which may
 *           have a device optimized memory layout)
 *
 *  @param   memHandle   Interop memory handle
 *  @param   desc        The image descriptor
 *  @param   syclDevice The device in which we create our image memory handle
 *  @param   syclContext The conext in which we create our image memory handle
 *  @return  Memory handle to externally allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle map_external_memory_array(
    interop_mem_handle memHandle, const image_descriptor &desc,
    const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Maps an interop memory handle to an image memory handle (which may
 *           have a device optimized memory layout)
 *
 *  @param   memHandle   Interop memory handle
 *  @param   desc        The image descriptor
 *  @param   syclQueue   The queue in which we create our image memory handle
 *  @return  Memory handle to externally allocated memory on the device
 */
__SYCL_EXPORT image_mem_handle map_external_memory_array(
    interop_mem_handle memHandle, const image_descriptor &descm,
    const sycl::queue &syclQueue);

/**
 *  @brief   Import external semaphore taking an external semaphore handle (the
 *           type of which is dependent on the OS & external API)
 *
 *  @tparam  ExternalSemaphoreHandleType Handle type describing external
 *           semaphore handle
 *  @param   externalSemaphoreDesc External semaphore descriptor
 *  @param   syclDevice The device in which we create our interop semaphore
 *           handle
 *  @param   syclContext The context in which we create our interop semaphore
 *           handle
 *  @return  Interop semaphore handle to the external semaphore
 */
template <typename ExternalSemaphoreHandleType>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<ExternalSemaphoreHandleType>
        externalSemaphoreDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Import external semaphore taking an external semaphore handle (the
 *           type of which is dependent on the OS & external API)
 *
 *  @tparam  ExternalSemaphoreHandleType Handle type describing external
 *           semaphore handle
 *  @param   externalSemaphoreDesc External semaphore descriptor
 *  @param   syclQueue The queue in which we create our interop semaphore
 *           handle
 *  @return  Interop semaphore handle to the external semaphore
 */
template <typename ExternalSemaphoreHandleType>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<ExternalSemaphoreHandleType>
        externalSemaphoreDesc,
    const sycl::queue &syclQueue);

/**
 *  @brief   Destroy the external semaphore handle
 *
 *  @param   semaphoreHandle The interop semaphore handle to destroy
 *  @param   syclDevice The device in which the interop semaphore handle was
 *           created
 *  @param   syclContext The context in which the interop semaphore handle was
 *           created
 */
__SYCL_EXPORT void
destroy_external_semaphore(interop_semaphore_handle semaphoreHandle,
                           const sycl::device &syclDevice,
                           const sycl::context &syclContext);

/**
 *  @brief   Destroy the external semaphore handle
 *
 *  @param   semaphoreHandle The interop semaphore handle to destroy
 *  @param   syclQueue The queue in which the interop semaphore handle was
 *           created
 */
__SYCL_EXPORT void
destroy_external_semaphore(interop_semaphore_handle semaphoreHandle,
                           const sycl::queue &syclQueue);

/**
 *  @brief   Release external memory
 *
 *  @param   interopHandle The interop memory handle to release
 *  @param   syclDevice The device in which the interop memory handle was
 * created
 *  @param   syclContext The context in which the interop memory handle was
 * created
 */
__SYCL_EXPORT void release_external_memory(interop_mem_handle interopHandle,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext);

/**
 *  @brief   Release external memory
 *
 *  @param   interopHandle The interop memory handle to release
 *  @param   syclQueue The queue in which the interop memory handle was
 * created
 */
__SYCL_EXPORT void release_external_memory(interop_mem_handle interopHandle,
                                           const sycl::queue &syclQueue);

/**
 *  @brief   Create an image and return the device image handle
 *
 *  @param   memHandle Device memory handle wrapper for allocated image memory
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we created our image handle
 *  @param   syclContext The context in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT unsampled_image_handle
create_image(image_mem &memHandle, const image_descriptor &desc,
             const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Create an image and return the device image handle
 *
 *  @param   memHandle Device memory handle wrapper for allocated image memory
 *  @param   desc The image descriptor
 *  @param   syclqueue The queue in which we created our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT unsampled_image_handle create_image(image_mem &memHandle,
                                                  const image_descriptor &desc,
                                                  const sycl::queue &syclQueue);

/**
 *  @brief   Create an image and return the device image handle
 *
 *  @param   memHandle Device memory handle for allocated image memory
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we created our image handle
 *  @param   syclContext The context in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT unsampled_image_handle
create_image(image_mem_handle memHandle, const image_descriptor &desc,
             const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Create an image and return the device image handle
 *
 *  @param   memHandle Device memory handle for allocated image memory
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue in which we created our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT unsampled_image_handle create_image(image_mem_handle memHandle,
                                                  const image_descriptor &desc,
                                                  const sycl::queue &syclQueue);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   imgMem Device memory pointer to allocated image memory
 *  @param   pitch The allocation pitch value
 *  @param   sampler bindless image sampler to sample the image
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we create our image handle
 *  @param   syclContext The context in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(void *imgMem, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   imgMem Device memory pointer to allocated image memory
 *  @param   pitch The allocation pitch value
 *  @param   sampler bindless image sampler used to sample the image
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(void *imgMem, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   memHandle Device memory handle wrapper for allocated image memory
 *  @param   sampler bindless image sampler used to sample the image
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we create our image handle
 *  @param   syclContext The context in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(image_mem &memHandle, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   memHandle Device memory handle wrapper for allocated image memory
 *  @param   sampler bindless image sampler used to sample the image
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(image_mem &memHandle, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   memHandle Device memory handle for allocated image memory
 *  @param   sampler bindless image sampler used to sample the image
 *  @param   desc The image descriptor
 *  @param   syclDevice The device in which we create our image handle
 *  @param   syclContext The context in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(image_mem_handle memHandle, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext);

/**
 *  @brief   Create a sampled image and return the device image handle
 *
 *  @param   memHandle Device memory handle for allocated image memory
 *  @param   sampler bindless image sampler used to sample the image
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue in which we create our image handle
 *  @return  Image handle to created image object on the device
 */
__SYCL_EXPORT sampled_image_handle
create_image(image_mem_handle memHandle, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue);

/**
 *  @brief   Destroy an unsampled image handle. Does not free memory backing the
 *           handle
 *
 *  @param   imageHandle The unsampled image handle to destroy
 *  @param   syclDevice The device in which we created our image handle
 *  @param   syclContext The context in which we created our image handle
 **/
__SYCL_EXPORT void destroy_image_handle(unsampled_image_handle &imageHandle,
                                        const sycl::device &syclDevice,
                                        const sycl::context &syclContext);

/**
 *  @brief   Destroy an unsampled image handle. Does not free memory backing the
 *           handle
 *
 *  @param   imageHandle The unsampled image handle to destroy
 *  @param   syclQueue The queue in which we created our image handle
 **/
__SYCL_EXPORT void destroy_image_handle(unsampled_image_handle &imageHandle,
                                        const sycl::queue &syclQueue);

/**
 *  @brief   Destroy a sampled image handle. Does not free memory backing the
 *           handle
 *
 *  @param   imageHandle The sampled image handle to destroy
 *  @param   syclDevice The device in which we created our image handle
 *  @param   syclContext The context in which we created our image handle
 **/
__SYCL_EXPORT void destroy_image_handle(sampled_image_handle &imageHandle,
                                        const sycl::device &syclDevice,
                                        const sycl::context &syclContext);

/**
 *  @brief   Destroy a sampled image handle. Does not free memory backing the
 *           handle
 *
 *  @param   imageHandle The sampled image handle to destroy
 *  @param   syclQueue The queue in which we created our image handle
 **/
__SYCL_EXPORT void destroy_image_handle(sampled_image_handle &imageHandle,
                                        const sycl::queue &syclQueue);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   widthInBytes The width of the image in bytes
 *  @param   height The height of the image in elements
 *  @param   elementSizeBytes Number of bytes of a singular image element
 *  @param   syclQueue The queue
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const sycl::queue &syclQueue);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   widthInBytes The width of the image in bytes
 *  @param   height The height of the image in elements
 *  @param   elementSizeBytes Number of bytes of a singular image element
 *  @param   syclDevice The device
 *  @param   syclContext The context
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const sycl::device &syclDevice,
                                         const sycl::context &syclContext);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   desc The image descriptor
 *  @param   syclQueue The queue
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         const image_descriptor &desc,
                                         const sycl::queue &syclQueue);

/**
 *  @brief   Allocate pitched USM image memory
 *
 *  @param   resultPitch The allocation pitch value
 *  @param   desc The image descriptor
 *  @param   syclDevice The device
 *  @param   syclContext The context
 *  @return  Generic pointer to allocated USM image memory
 */
__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         const image_descriptor &desc,
                                         const sycl::device &syclDevice,
                                         const sycl::context &syclContext);

/**
 *  @brief   Get the range that describes the image's dimensions
 *
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @param   syclDevice The device in which we created our image memory handle
 *  @param   syclContext The context in which we created our image memory handle
 *  @return  sycl range describing image's dimensions
 */
__SYCL_EXPORT sycl::range<3> get_image_range(const image_mem_handle memHandle,
                                             const sycl::device &syclDevice,
                                             const sycl::context &syclContext);

/**
 *  @brief   Get the range that describes the image's dimensions
 *
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @param   syclQueue The queue in which we created our image memory handle
 *  @return  sycl range describing image's dimensions
 */
__SYCL_EXPORT sycl::range<3> get_image_range(const image_mem_handle memHandle,
                                             const sycl::queue &syclQueue);

/**
 *  @brief   Get the channel type that describes the image memory
 *
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @param   syclDevice The device in which we created our image memory handle
 *  @param   syclContext The context in which we created our image memory handle
 *  @return  sycl image channel type that describes the image
 */
__SYCL_EXPORT sycl::image_channel_type
get_image_channel_type(const image_mem_handle memHandle,
                       const sycl::device &syclDevice,
                       const sycl::context &syclContext);

/**
 *  @brief   Get the channel type that describes the image memory
 *
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @param   syclQueue The queue in which we created our image memory handle
 *  @return  sycl image channel type that describes the image
 */
__SYCL_EXPORT sycl::image_channel_type
get_image_channel_type(const image_mem_handle memHandle,
                       const sycl::queue &syclQueue);

/**
 *  @brief   Get the number of channels that describes the image memory
 *
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @param   syclDevice The device in which we created our image memory handle
 *  @param   syclContext The context in which we created our image memory handle
 *  @return  The number of channels describing the image
 */
__SYCL_EXPORT unsigned int
get_image_num_channels(const image_mem_handle memHandle,
                       const sycl::device &syclDevice,
                       const sycl::context &syclContext);

/**
 *  @brief   Get the number of channels that describes the image memory
 *
 *  @param   memHandle Memory handle to allocated memory on the device
 *  @param   syclQueue The queue in which we created our image memory handle
 *  @return  The number of channels describing the image
 */
__SYCL_EXPORT unsigned int
get_image_num_channels(const image_mem_handle memHandle,
                       const sycl::queue &syclQueue);

namespace detail {
// Get the number of coordinates
template <typename CoordT> constexpr size_t coord_size() {
  if constexpr (std::is_scalar<CoordT>::value) {
    return 1;
  } else {
    return CoordT::size();
  }
}

// bit_cast Color to a type the NVPTX backend is known to accept
template <typename DataT> constexpr auto convert_color_nvptx(DataT Color) {
  constexpr size_t dataSize = sizeof(DataT);
  static_assert(
      dataSize == 1 || dataSize == 2 || dataSize == 4 || dataSize == 8 ||
          dataSize == 16,
      "Expected input data type to be of size 1, 2, 4, 8, or 16 bytes.");

  if constexpr (dataSize == 1) {
    return sycl::bit_cast<uint8_t>(Color);
  } else if constexpr (dataSize == 2) {
    return sycl::bit_cast<uint16_t>(Color);
  } else if constexpr (dataSize == 4) {
    return sycl::bit_cast<uint32_t>(Color);
  } else if constexpr (dataSize == 8) {
    return sycl::bit_cast<sycl::vec<uint32_t, 2>>(Color);
  } else { // dataSize == 16
    return sycl::bit_cast<sycl::vec<uint32_t, 4>>(Color);
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
DataT read_image(const unsampled_image_handle &imageHandle [[maybe_unused]],
                 const CoordT &coords [[maybe_unused]]) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                "Expected input coordinate to be have 1, 2, or 4 components "
                "for 1D, 2D and 3D images respectively.");

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
DataT read_image(const sampled_image_handle &imageHandle [[maybe_unused]],
                 const CoordT &coords [[maybe_unused]]) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                "Expected input coordinate to be have 1, 2, or 4 components "
                "for 1D, 2D and 3D images respectively.");

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
DataT read_image(const sampled_image_handle &imageHandle [[maybe_unused]],
                 const CoordT &coords [[maybe_unused]],
                 const float level [[maybe_unused]]) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                "Expected input coordinate to be have 1, 2, or 4 components "
                "for 1D, 2D and 3D images respectively.");

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  return __invoke__ImageReadLod<DataT, uint64_t, CoordT>(imageHandle.raw_handle,
                                                         coords, level);
#else
  // TODO: add SPIRV for mipmap level read
#endif
#else
  assert(false); // Bindless images not yet implemented on host
#endif
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
DataT read_image(const sampled_image_handle &imageHandle [[maybe_unused]],
                 const CoordT &coords [[maybe_unused]],
                 const CoordT &dX [[maybe_unused]],
                 const CoordT &dY [[maybe_unused]]) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                "Expected input coordinate and gradient to have 1, 2, or 4 "
                "components "
                "for 1D, 2D and 3D images respectively.");

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
void write_image(const unsampled_image_handle &imageHandle [[maybe_unused]],
                 const CoordT &Coords [[maybe_unused]],
                 const DataT &Color [[maybe_unused]]) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                "Expected input coordinate to be have 1, 2, or 4 components "
                "for 1D, 2D and 3D images respectively.");

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  __invoke__ImageWrite((uint64_t)imageHandle.raw_handle, Coords,
                       detail::convert_color_nvptx(Color));
#else
  // TODO: add SPIRV part for unsampled image write
#endif
#else
  assert(false); // Bindless images not yet implemented on host
#endif
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
