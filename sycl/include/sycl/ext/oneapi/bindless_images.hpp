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

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/detail/image_ocl_types.hpp> // for __invoke__*
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

/// Opaque unsampled image handle type.
struct unsampled_image_handle {
  using raw_image_handle_type = pi_uint64;

  unsampled_image_handle() : raw_handle(~0) {}

  unsampled_image_handle(raw_image_handle_type raw_image_handle)
      : raw_handle(raw_image_handle) {}

  raw_image_handle_type raw_handle;
};

/// Opaque sampled image handle type.
struct sampled_image_handle {
  using raw_image_handle_type = pi_uint64;

  sampled_image_handle() : raw_handle(~0) {}

  sampled_image_handle(raw_image_handle_type handle) : raw_handle(handle) {}

  raw_image_handle_type raw_handle;
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
 *  @param   imageType Type of image memory to be freed
 *  @param   syclDevice The device in which we create our memory handle
 *  @param   syclContext The context in which we created our memory handle
 */
__SYCL_EXPORT void free_image_mem(image_mem_handle handle, image_type imageType,
                                  const sycl::device &syclDevice,
                                  const sycl::context &syclContext);

/**
 *  @brief   Free image memory
 *
 *  @param   handle Memory handle to allocated memory on the device
 *  @param   imageType Type of image memory to be freed
 *  @param   syclQueue The queue in which we create our memory handle
 */
__SYCL_EXPORT void free_image_mem(image_mem_handle handle, image_type imageType,
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
    const image_mem_handle mipMem, unsigned int level,
    const sycl::device &syclDevice, const sycl::context &syclContext);

/**
 *  @brief   Retrieve the memory handle to an individual mipmap image
 *
 *  @param   mipMem The memory handle to the mipmapped array
 *  @param   level The requested level of the mipmap
 *  @param   syclQueue The queue in which we created our memory handle
 *  @return  Memory handle to the individual mipmap image
 */
__SYCL_EXPORT image_mem_handle
get_mip_level_mem_handle(const image_mem_handle mipMem, unsigned int level,
                         const sycl::queue &syclQueue);

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
__SYCL_EXPORT
image_mem_handle map_external_image_memory(interop_mem_handle memHandle,
                                           const image_descriptor &desc,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext);

/**
 *  @brief   Maps an interop memory handle to an image memory handle (which may
 *           have a device optimized memory layout)
 *
 *  @param   memHandle   Interop memory handle
 *  @param   desc        The image descriptor
 *  @param   syclQueue   The queue in which we create our image memory handle
 *  @return  Memory handle to externally allocated memory on the device
 */
__SYCL_EXPORT
image_mem_handle map_external_image_memory(interop_mem_handle memHandle,
                                           const image_descriptor &desc,
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
 *  @brief   Release the external semaphore
 *
 *  @param   semaphoreHandle The interop semaphore handle to destroy
 *  @param   syclDevice The device in which the interop semaphore handle was
 *           created
 *  @param   syclContext The context in which the interop semaphore handle was
 *           created
 */
__SYCL_EXPORT void
release_external_semaphore(interop_semaphore_handle semaphoreHandle,
                           const sycl::device &syclDevice,
                           const sycl::context &syclContext);

/**
 *  @brief   Release the external semaphore
 *
 *  @param   semaphoreHandle The interop semaphore handle to destroy
 *  @param   syclQueue The queue in which the interop semaphore handle was
 *           created
 */
__SYCL_EXPORT void
release_external_semaphore(interop_semaphore_handle semaphoreHandle,
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

// is sycl::vec
template <typename T> struct is_vec {
  static constexpr bool value = false;
};
template <typename T, int N> struct is_vec<sycl::vec<T, N>> {
  static constexpr bool value = true;
};
template <typename T> inline constexpr bool is_vec_v = is_vec<T>::value;

// Get the number of coordinates
template <typename CoordT> constexpr size_t coord_size() {
  if constexpr (std::is_scalar_v<CoordT>) {
    return 1;
  } else {
    return CoordT::size();
  }
}

// bit_cast Color to a type the backend is known to accept
template <typename DataT> constexpr auto convert_color(DataT Color) {
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

// assert coords or elements of coords is of an integer type
template <typename CoordT> constexpr void assert_unsampled_coords() {
  if constexpr (std::is_scalar_v<CoordT>) {
    static_assert(std::is_same_v<CoordT, int>,
                  "Expected integer coordinate data type");
  } else {
    static_assert(is_vec_v<CoordT>, "Expected sycl::vec coordinates");
    static_assert(std::is_same_v<typename CoordT::element_type, int>,
                  "Expected integer coordinates data type");
  }
}

template <typename CoordT> constexpr bool are_floating_coords() {
  if constexpr (is_vec_v<CoordT>) {
    return std::is_same_v<typename CoordT::element_type, float>;
  } else {
    return std::is_same_v<CoordT, float>;
  }
}

template <typename CoordT> constexpr bool are_integer_coords() {
  if constexpr (is_vec_v<CoordT>) {
    return std::is_same_v<typename CoordT::element_type, int>;
  } else {
    return std::is_same_v<CoordT, int>;
  }
}

template <typename CoordT> constexpr void assert_coords_type() {
  static_assert(are_floating_coords<CoordT>() || are_integer_coords<CoordT>(),
                "Expected coordinates to be of `float` or `int` type, or "
                "vectors of these types.");
}

// assert coords or elements of coords is of a float type
template <typename CoordT> constexpr void assert_sample_coords() {
  if constexpr (std::is_scalar_v<CoordT>) {
    static_assert(std::is_same_v<CoordT, float>,
                  "Expected float coordinate data type");
  } else {
    static_assert(is_vec_v<CoordT>, "Expected sycl::vec coordinates");
    static_assert(std::is_same_v<typename CoordT::element_type, float>,
                  "Expected float coordinates data type");
  }
}

// assert coords or elements of coords is of a int type
template <typename CoordT> constexpr void assert_fetch_coords() {
  if constexpr (std::is_scalar_v<CoordT>) {
    static_assert(std::is_same_v<CoordT, int>,
                  "Expected int coordinate data type");
  } else {
    static_assert(is_vec_v<CoordT>, "Expected sycl::vec coordinates");
    static_assert(std::is_same_v<typename CoordT::element_type, int>,
                  "Expected int coordinates data type");
  }
}

template <typename DataT> constexpr bool is_data_size_valid() {
  return (sizeof(DataT) == 1) || (sizeof(DataT) == 2) || (sizeof(DataT) == 4) ||
         (sizeof(DataT) == 8) || (sizeof(DataT) == 16);
}

template <typename DataT> constexpr bool is_recognized_standard_type() {
  return is_data_size_valid<DataT>() &&
         (is_vec_v<DataT> || std::is_scalar_v<DataT> ||
          std::is_floating_point_v<DataT> || std::is_same_v<DataT, sycl::half>);
}

#ifdef __SYCL_DEVICE_ONLY__

// Image types used for generating SPIR-V
template <int NDims>
using OCLImageTyRead =
    typename sycl::detail::opencl_image_type<NDims, sycl::access::mode::read,
                                             sycl::access::target::image>::type;

template <int NDims>
using OCLImageTyWrite =
    typename sycl::detail::opencl_image_type<NDims, sycl::access::mode::write,
                                             sycl::access::target::image>::type;

template <int NDims>
using OCLImageArrayTyRead = typename sycl::detail::opencl_image_type<
    NDims, sycl::access::mode::read, sycl::access::target::image_array>::type;

template <int NDims>
using OCLImageArrayTyWrite = typename sycl::detail::opencl_image_type<
    NDims, sycl::access::mode::write, sycl::access::target::image_array>::type;

template <int NDims>
using OCLSampledImageArrayTyRead =
    typename sycl::detail::sampled_opencl_image_type<
        detail::OCLImageArrayTyRead<NDims>>::type;

// Macros are required because it is not legal for a function to return
// a variable of type 'opencl_image_type'.
#if defined(__SPIR__)
#define CONVERT_HANDLE_TO_IMAGE(raw_handle, ImageType)                         \
  __spirv_ConvertHandleToImageINTEL<ImageType>(raw_handle)

#define CONVERT_HANDLE_TO_SAMPLED_IMAGE(raw_handle, NDims)                     \
  __spirv_ConvertHandleToSampledImageINTEL<                                    \
      typename sycl::detail::sampled_opencl_image_type<                        \
          detail::OCLImageTyRead<NDims>>::type>(raw_handle)

#define CONVERT_HANDLE_TO_SAMPLED_IMAGE_ARRAY(raw_handle, NDims)               \
  __spirv_ConvertHandleToSampledImageINTEL<                                    \
      typename sycl::detail::sampled_opencl_image_type<                        \
          detail::OCLImageArrayTyRead<NDims>>::type>(raw_handle)

#define FETCH_UNSAMPLED_IMAGE(DataT, raw_handle, coords)                       \
  __invoke__ImageRead<DataT>(raw_handle, coords)

#define FETCH_SAMPLED_IMAGE(DataT, raw_handle, coords)                         \
  __invoke__ImageReadLod<DataT>(raw_handle, coords, 0.f)

#define SAMPLE_IMAGE_READ(DataT, raw_handle, coords)                           \
  __invoke__ImageReadLod<DataT>(raw_handle, coords, 0.f)

#define FETCH_IMAGE_ARRAY(DataT, raw_handle, coords, arrayLayer, coordsLayer)  \
  __invoke__ImageRead<DataT>(raw_handle, coordsLayer)

#define WRITE_IMAGE_ARRAY(raw_handle, coords, arrayLayer, coordsLayer, color)  \
  __invoke__ImageWrite(raw_handle, coordsLayer, color)

#define FETCH_SAMPLED_IMAGE_ARRAY(DataT, raw_handle, coords, arrayLayer,       \
                                  coordsLayer)                                 \
  __invoke__ImageReadLod<DataT>(raw_handle, coordsLayer, 0.f)

#define READ_SAMPLED_IMAGE_ARRAY(DataT, raw_handle, coords, arrayLayer,        \
                                 coordsLayer)                                  \
  __invoke__ImageReadLod<DataT>(raw_handle, coordsLayer, 0.f)

#else
#define CONVERT_HANDLE_TO_IMAGE(raw_handle, ImageType) raw_handle

#define CONVERT_HANDLE_TO_SAMPLED_IMAGE(raw_handle, NDims) raw_handle

#define CONVERT_HANDLE_TO_SAMPLED_IMAGE_ARRAY(raw_handle, NDims) raw_handle

#define FETCH_UNSAMPLED_IMAGE(DataT, raw_handle, coords)                       \
  __invoke__ImageFetch<DataT>(raw_handle, coords)

#define FETCH_SAMPLED_IMAGE(DataT, raw_handle, coords)                         \
  __invoke__SampledImageFetch<DataT>(raw_handle, coords)

#define SAMPLE_IMAGE_READ(DataT, raw_handle, coords)                           \
  __invoke__ImageRead<DataT>(raw_handle, coords)

#define FETCH_IMAGE_ARRAY(DataT, raw_handle, coords, arrayLayer, coordsLayer)  \
  __invoke__ImageArrayFetch<DataT>(raw_handle, coords, arrayLayer)

#define WRITE_IMAGE_ARRAY(raw_handle, coords, arrayLayer, coordsLayer, color)  \
  __invoke__ImageArrayWrite(raw_handle, coords, arrayLayer, color)

#define FETCH_SAMPLED_IMAGE_ARRAY(DataT, raw_handle, coords, arrayLayer,       \
                                  coordsLayer)                                 \
  __invoke__SampledImageArrayFetch<DataT>(raw_handle, coords, arrayLayer)

#define READ_SAMPLED_IMAGE_ARRAY(DataT, raw_handle, coords, arrayLayer,        \
                                 coordsLayer)                                  \
  __invoke__ImageArrayRead<DataT>(raw_handle, coords, arrayLayer)

#endif

#endif // __SYCL_DEVICE_ONLY__

} // namespace detail

/**
 *  @brief   Fetch data from an unsampled image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. int, int2, or int3 for
 *           1D, 2D, and 3D, respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to fetch image data
 *  @return  Image data
 *
 *  __NVPTX__: Name mangling info
 *             Cuda surfaces require integer coords (by bytes)
 *             Cuda textures require float coords (by element or normalized)
 *             for sampling, and integer coords (by bytes) for fetching
 *             The name mangling should therefore not interfere with one
 *             another
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT fetch_image(const unsampled_image_handle &imageHandle [[maybe_unused]],
                  const CoordT &coords [[maybe_unused]]) {
  detail::assert_fetch_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 3,
                "Expected input coordinate to be have 1, 2, or 3 components "
                "for 1D, 2D and 3D images, respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return FETCH_UNSAMPLED_IMAGE(
        DataT,
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageTyRead<coordSize>),
        coords);

  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to read a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(FETCH_UNSAMPLED_IMAGE(
        HintT,
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageTyRead<coordSize>),
        coords));
  }
#else
  assert(false); // Bindless images not yet implemented on host
#endif
}

/**
 *  @brief   Fetch data from a sampled image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. int, int2, or int3 for
 *           1D, 2D, and 3D, respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to fetch image data
 *  @return  Fetched image data
 *
 *  __NVPTX__: Name mangling info
 *             Cuda surfaces require integer coords (by bytes)
 *             Cuda textures require float coords (by element or normalized)
 *             for sampling, and integer coords (by bytes) for fetching
 *             The name mangling should therefore not interfere with one
 *             another
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT fetch_image(const sampled_image_handle &imageHandle [[maybe_unused]],
                  const CoordT &coords [[maybe_unused]]) {
  detail::assert_fetch_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 3,
                "Expected input coordinate to be have 1, 2, or 3 components "
                "for 1D, 2D and 3D images, respectively.");
  static_assert(sizeof(HintT) == sizeof(DataT),
                "When trying to read a user-defined type, HintT must be of "
                "the same size as the user-defined DataT.");
  static_assert(detail::is_recognized_standard_type<HintT>(),
                "HintT must always be a recognized standard type");

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return FETCH_SAMPLED_IMAGE(
        DataT,
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords);
  } else {
    return sycl::bit_cast<DataT>(FETCH_SAMPLED_IMAGE(
        HintT,
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords));
  }
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

/**
 *  @brief   Sample data from a sampled image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. float, float2, or float3 for
 *           1D, 2D, and 3D, respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to sample image data
 *  @return  Sampled image data
 *
 *  __NVPTX__: Name mangling info
 *             Cuda surfaces require integer coords (by bytes)
 *             Cuda textures require float coords (by element or normalized)
 *             for sampling, and integer coords (by bytes) for fetching
 *             The name mangling should therefore not interfere with one
 *             another
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT sample_image(const sampled_image_handle &imageHandle [[maybe_unused]],
                   const CoordT &coords [[maybe_unused]]) {
  detail::assert_sample_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 3,
                "Expected input coordinate to be have 1, 2, or 3 components "
                "for 1D, 2D and 3D images, respectively.");
  static_assert(sizeof(HintT) == sizeof(DataT),
                "When trying to read a user-defined type, HintT must be of "
                "the same size as the user-defined DataT.");
  static_assert(detail::is_recognized_standard_type<HintT>(),
                "HintT must always be a recognized standard type");

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return SAMPLE_IMAGE_READ(
        DataT,
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords);
  } else {
    return sycl::bit_cast<DataT>(SAMPLE_IMAGE_READ(
        HintT,
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords));
  }
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

/**
 *  @brief   Sample a mipmap image using its handle with LOD filtering
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. float, float2, or float3 for
 *           1D, 2D, and 3D, respectively
 *  @param   imageHandle The mipmap image handle
 *  @param   coords The coordinates at which to sample mipmap image data
 *  @param   level The mipmap level at which to sample
 *  @return  Mipmap image data with LOD filtering
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT sample_mipmap(const sampled_image_handle &imageHandle [[maybe_unused]],
                    const CoordT &coords [[maybe_unused]],
                    const float level [[maybe_unused]]) {
  detail::assert_sample_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 3,
                "Expected input coordinate to be have 1, 2, or 3 components "
                "for 1D, 2D and 3D images, respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return __invoke__ImageReadLod<DataT>(
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords, level);
  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to read a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(__invoke__ImageReadLod<HintT>(
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords, level));
  }
#else
  assert(false); // Bindless images not yet implemented on host
#endif
}

/**
 *  @brief   Sample a mipmap image using its handle with anisotropic filtering
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. float, float2, or float3 for
 *           1D, 2D, and 3D, respectively
 *  @param   imageHandle The mipmap image handle
 *  @param   coords The coordinates at which to sample mipmap image data
 *  @param   dX Screen space gradient in the x dimension
 *  @param   dY Screen space gradient in the y dimension
 *  @return  Mipmap image data with anisotropic filtering
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT sample_mipmap(const sampled_image_handle &imageHandle [[maybe_unused]],
                    const CoordT &coords [[maybe_unused]],
                    const CoordT &dX [[maybe_unused]],
                    const CoordT &dY [[maybe_unused]]) {
  detail::assert_sample_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 3,
                "Expected input coordinates and gradients to have 1, 2, or 3 "
                "components for 1D, 2D, and 3D images, respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return __invoke__ImageReadGrad<DataT>(
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords, dX, dY);
  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to read a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(__invoke__ImageReadGrad<HintT>(
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, coordSize),
        coords, dX, dY));
  }
#else
  assert(false); // Bindless images not yet implemented on host
#endif
}

/**
 *  @brief   Fetch data from an unsampled image array using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. int or int2 for 1D or 2D,
 *           respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to fetch image data
 *  @param   arrayLayer The image array layer at which to fetch
 *  @return  Image data
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT fetch_image_array(const unsampled_image_handle &imageHandle
                        [[maybe_unused]],
                        const CoordT &coords [[maybe_unused]],
                        unsigned int arrayLayer [[maybe_unused]]) {
  detail::assert_unsampled_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2,
                "Expected input coordinate to be have 1 or 2 components for 1D "
                "and 2D images respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  sycl::vec<int, coordSize + 1> coordsLayer{coords, arrayLayer};
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return FETCH_IMAGE_ARRAY(
        DataT,
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageArrayTyRead<coordSize>),
        coords, arrayLayer, coordsLayer);
  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to fetch a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(FETCH_IMAGE_ARRAY(
        HintT,
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageArrayTyRead<coordSize>),
        coords, arrayLayer, coordsLayer));
  }
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

/**
 *  @brief   Fetch data from an unsampled cubemap image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to fetch image data (int2 only)
 *  @param   face The cubemap face at which to fetch
 *  @return  Image data
 */
template <typename DataT, typename HintT = DataT>
DataT fetch_cubemap(const unsampled_image_handle &imageHandle,
                    const int2 &coords, unsigned int face) {
  return fetch_image_array<DataT, HintT>(imageHandle, coords, face);
}

/**
 *  @brief   Sample a cubemap image using its handle
 *
 *  @tparam  DataT The return type
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *
 *  @param   imageHandle The image handle
 *  @param   dirVec The direction vector at which to sample image data (float3
 *           only)
 *  @return  Image data
 */
template <typename DataT, typename HintT = DataT>
DataT sample_cubemap(const sampled_image_handle &imageHandle [[maybe_unused]],
                     const sycl::float3 &dirVec [[maybe_unused]]) {
  [[maybe_unused]] constexpr size_t NDims = 2;

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return __invoke__ImageReadCubemap<DataT, uint64_t>(
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, NDims), dirVec);
  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to read a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(__invoke__ImageReadCubemap<HintT, uint64_t>(
        CONVERT_HANDLE_TO_SAMPLED_IMAGE(imageHandle.raw_handle, NDims),
        dirVec));
  }
#else
  assert(false); // Bindless images not yet implemented on host
#endif
}

/**
 *  @brief   Fetch data from a sampled image array using its handle.
 *
 *  @tparam  DataT The return type.
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. int or int2 for 1D or 2D,
 *           respectively.
 *  @param   imageHandle The image handle.
 *  @param   coords The coordinates at which to fetch image data.
 *  @param   arrayLayer The image array layer at which to fetch.
 *  @return  Image data.
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT fetch_image_array(const sampled_image_handle &imageHandle
                        [[maybe_unused]],
                        const CoordT &coords [[maybe_unused]],
                        unsigned int arrayLayer [[maybe_unused]]) {
  detail::assert_unsampled_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2,
                "Expected input coordinate to be have 1 or 2 components for 1D "
                "and 2D images respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  sycl::vec<int, coordSize + 1> coordsLayer{coords, arrayLayer};
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return FETCH_SAMPLED_IMAGE_ARRAY(DataT,
                                     CONVERT_HANDLE_TO_SAMPLED_IMAGE_ARRAY(
                                         imageHandle.raw_handle, coordSize),
                                     coords, arrayLayer, coordsLayer);
  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to fetch a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(
        FETCH_SAMPLED_IMAGE_ARRAY(HintT,
                                  CONVERT_HANDLE_TO_SAMPLED_IMAGE_ARRAY(
                                      imageHandle.raw_handle, coordSize),
                                  coords, arrayLayer, coordsLayer));
  }
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

/**
 *  @brief   Sample data from a sampled image array using its handle.
 *
 *  @tparam  DataT The return type.
 *  @tparam  HintT A hint type that can be used to select for a specialized
 *           backend intrinsic when a user-defined type is passed as `DataT`.
 *           HintT should be a `sycl::vec` type, `sycl::half` type, or POD type.
 *           HintT must also have the same size as DataT.
 *  @tparam  CoordT The input coordinate type. e.g. int or int2 for 1D or 2D,
 *           respectively.
 *  @param   imageHandle The image handle.
 *  @param   coords The coordinates at which to fetch image data.
 *  @param   arrayLayer The image array layer at which to fetch.
 *  @return  Image data.
 */
template <typename DataT, typename HintT = DataT, typename CoordT>
DataT sample_image_array(const sampled_image_handle &imageHandle
                         [[maybe_unused]],
                         const CoordT &coords [[maybe_unused]],
                         unsigned int arrayLayer [[maybe_unused]]) {
  detail::assert_sample_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2,
                "Expected input coordinate to be have 1 or 2 components for 1D "
                "and 2D images respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  sycl::vec<float, coordSize + 1> coordsLayer{coords, arrayLayer};
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    return READ_SAMPLED_IMAGE_ARRAY(DataT,
                                    CONVERT_HANDLE_TO_SAMPLED_IMAGE_ARRAY(
                                        imageHandle.raw_handle, coordSize),
                                    coords, arrayLayer, coordsLayer);
  } else {
    static_assert(sizeof(HintT) == sizeof(DataT),
                  "When trying to fetch a user-defined type, HintT must be of "
                  "the same size as the user-defined DataT.");
    static_assert(detail::is_recognized_standard_type<HintT>(),
                  "HintT must always be a recognized standard type");
    return sycl::bit_cast<DataT>(
        READ_SAMPLED_IMAGE_ARRAY(HintT,
                                 CONVERT_HANDLE_TO_SAMPLED_IMAGE_ARRAY(
                                     imageHandle.raw_handle, coordSize),
                                 coords, arrayLayer, coordsLayer));
  }
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

/**
 *  @brief   Write to an unsampled image using its handle
 *
 *  @tparam  DataT The data type to write
 *  @tparam  CoordT The input coordinate type. e.g. int, int2, or int3 for
 *           1D, 2D, and 3D, respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to write image data
 *  @param   color The data to write
 */
template <typename DataT, typename CoordT>
void write_image(unsampled_image_handle imageHandle [[maybe_unused]],
                 const CoordT &coords [[maybe_unused]],
                 const DataT &color [[maybe_unused]]) {
  detail::assert_unsampled_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2 || coordSize == 3,
                "Expected input coordinate to be have 1, 2, or 3 components "
                "for 1D, 2D and 3D images, respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    __invoke__ImageWrite(
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageTyWrite<coordSize>),
        coords, color);
  } else {
    // Convert DataT to a supported backend write type when user-defined type is
    // passed
    __invoke__ImageWrite(
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageTyWrite<coordSize>),
        coords, detail::convert_color(color));
  }
#else
  assert(false); // Bindless images not yet implemented on host
#endif
}

/**
 *  @brief   Write to an unsampled image array using its handle
 *
 *  @tparam  DataT The data type to write
 *  @tparam  CoordT The input coordinate type. e.g. int or int2 for 1D or 2D,
 *           respectively
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to write image data
 *  @param   arrayLayer The image array layer at which to write
 *  @param   color The data to write
 */
template <typename DataT, typename CoordT>
void write_image_array(unsampled_image_handle imageHandle [[maybe_unused]],
                       const CoordT &coords [[maybe_unused]],
                       unsigned int arrayLayer [[maybe_unused]],
                       const DataT &color [[maybe_unused]]) {
  detail::assert_unsampled_coords<CoordT>();
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  static_assert(coordSize == 1 || coordSize == 2,
                "Expected input coordinate to be have 1 or 2 components for 1D "
                "and 2D images respectively.");

#ifdef __SYCL_DEVICE_ONLY__
  sycl::vec<int, coordSize + 1> coordsLayer{coords, arrayLayer};
  if constexpr (detail::is_recognized_standard_type<DataT>()) {
    WRITE_IMAGE_ARRAY(
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageArrayTyWrite<coordSize>),
        coords, arrayLayer, coordsLayer, color);
  } else {
    // Convert DataT to a supported backend write type when user-defined type is
    // passed
    WRITE_IMAGE_ARRAY(
        CONVERT_HANDLE_TO_IMAGE(imageHandle.raw_handle,
                                detail::OCLImageArrayTyWrite<coordSize>),
        coords, arrayLayer, coordsLayer, detail::convert_color(color));
  }
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

/**
 *  @brief   Write to an unsampled cubemap using its handle
 *
 *  @tparam  DataT The data type to write
 *
 *  @param   imageHandle The image handle
 *  @param   coords The coordinates at which to write image data (int2 only)
 *  @param   face The cubemap face at which to write
 *  @param   color The data to write
 */
template <typename DataT>
void write_cubemap(unsampled_image_handle imageHandle, const sycl::int2 &coords,
                   int face, const DataT &color) {
  return write_image_array(imageHandle, coords, face, color);
}

} // namespace ext::oneapi::experimental

inline event queue::ext_oneapi_copy(
    const void *Src, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) { CGH.ext_oneapi_copy(Src, Dest, DestImgDesc); },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_copy(Src, SrcOffset, SrcExtent, Dest, DestOffset,
                            DestImgDesc, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, Dest, DestImgDesc);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent, event DepEvent,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, SrcOffset, SrcExtent, Dest, DestOffset,
                            DestImgDesc, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    const std::vector<event> &DepEvents, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, Dest, DestImgDesc);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, sycl::range<3> SrcExtent,
    ext::oneapi::experimental::image_mem_handle Dest, sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DestImgDesc,
    sycl::range<3> CopyExtent, const std::vector<event> &DepEvents,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, SrcOffset, SrcExtent, Dest, DestOffset,
                            DestImgDesc, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) { CGH.ext_oneapi_copy(Src, Dest, SrcImgDesc); },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset, sycl::range<3> DestExtent,
    sycl::range<3> CopyExtent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_copy(Src, SrcOffset, SrcImgDesc, Dest, DestOffset,
                            DestExtent, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, Dest, SrcImgDesc);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset, sycl::range<3> DestExtent,
    sycl::range<3> CopyExtent, event DepEvent,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, SrcOffset, SrcImgDesc, Dest, DestOffset,
                            DestExtent, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc,
    const std::vector<event> &DepEvents, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, Dest, SrcImgDesc);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    sycl::range<3> SrcOffset,
    const ext::oneapi::experimental::image_descriptor &SrcImgDesc, void *Dest,
    sycl::range<3> DestOffset, sycl::range<3> DestExtent,
    sycl::range<3> CopyExtent, const std::vector<event> &DepEvents,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, SrcOffset, SrcImgDesc, Dest, DestOffset,
                            DestExtent, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_copy(Src, Dest, DeviceImgDesc, DeviceRowPitch);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, void *Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, sycl::range<3> HostExtent, sycl::range<3> CopyExtent,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_copy(Src, SrcOffset, Dest, DestOffset, DeviceImgDesc,
                            DeviceRowPitch, HostExtent, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, event DepEvent,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, Dest, DeviceImgDesc, DeviceRowPitch);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &ImageDesc,
    event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, Dest, ImageDesc);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &ImageDesc,
    const std::vector<event> &DepEvents, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, Dest, ImageDesc);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const ext::oneapi::experimental::image_mem_handle Src,
    ext::oneapi::experimental::image_mem_handle Dest,
    const ext::oneapi::experimental::image_descriptor &ImageDesc,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) { CGH.ext_oneapi_copy(Src, Dest, ImageDesc); },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, void *Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, sycl::range<3> HostExtent, sycl::range<3> CopyExtent,
    event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy(Src, SrcOffset, Dest, DestOffset, DeviceImgDesc,
                            DeviceRowPitch, HostExtent, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, void *Dest,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, const std::vector<event> &DepEvents,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, Dest, DeviceImgDesc, DeviceRowPitch);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_copy(
    const void *Src, sycl::range<3> SrcOffset, void *Dest,
    sycl::range<3> DestOffset,
    const ext::oneapi::experimental::image_descriptor &DeviceImgDesc,
    size_t DeviceRowPitch, sycl::range<3> HostExtent, sycl::range<3> CopyExtent,
    const std::vector<event> &DepEvents, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy(Src, SrcOffset, Dest, DestOffset, DeviceImgDesc,
                            DeviceRowPitch, HostExtent, CopyExtent);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    const std::vector<event> &DepEvents, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t WaitValue, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle, WaitValue);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t WaitValue, event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle, WaitValue);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_wait_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t WaitValue, const std::vector<event> &DepEvents,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_wait_external_semaphore(SemaphoreHandle, WaitValue);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    event DepEvent, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    const std::vector<event> &DepEvents, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t SignalValue, const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle, SignalValue);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t SignalValue, event DepEvent,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle, SignalValue);
      },
      CodeLoc);
}

inline event queue::ext_oneapi_signal_external_semaphore(
    sycl::ext::oneapi::experimental::interop_semaphore_handle SemaphoreHandle,
    uint64_t SignalValue, const std::vector<event> &DepEvents,
    const detail::code_location &CodeLoc) {
  detail::tls_code_loc_t TlsCodeLocCapture(CodeLoc);
  return submit(
      [&](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_signal_external_semaphore(SemaphoreHandle, SignalValue);
      },
      CodeLoc);
}

} // namespace _V1
} // namespace sycl
