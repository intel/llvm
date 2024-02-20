//==----------- bindless_images.hpp --- SYCL bindless images ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/sampler.hpp>

#include <detail/context_impl.hpp>
#include <detail/image_impl.hpp>
#include <detail/plugin_printers.hpp>
#include <detail/queue_impl.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

void populate_pi_structs(const image_descriptor &desc, pi_image_desc &piDesc,
                         pi_image_format &piFormat, size_t pitch = 0) {
  piDesc = {};
  piDesc.image_width = desc.width;
  piDesc.image_height = desc.height;
  piDesc.image_depth = desc.depth;
  piDesc.image_type = desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  piDesc.image_row_pitch = pitch;
  piDesc.image_array_size = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = desc.num_levels;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  piFormat = {};
  piFormat.image_channel_data_type =
      sycl::detail::convertChannelType(desc.channel_type);
  piFormat.image_channel_order =
      sycl::detail::convertChannelOrder(desc.channel_order);
}

detail::image_mem_impl::image_mem_impl(const image_descriptor &desc,
                                       const device &syclDevice,
                                       const context &syclContext)
    : descriptor(desc), syclDevice(syclDevice), syclContext(syclContext) {
  handle = alloc_image_mem(desc, syclDevice, syclContext);
}

detail::image_mem_impl::~image_mem_impl() {
  free_image_mem(this->get_handle(), this->get_descriptor().type,
                 this->get_device(), this->get_context());
}

__SYCL_EXPORT
image_mem::image_mem(const image_descriptor &desc,
                     const sycl::device &syclDevice,
                     const sycl::context &syclContext) {
  impl =
      std::make_shared<detail::image_mem_impl>(desc, syclDevice, syclContext);
}

__SYCL_EXPORT
image_mem::image_mem(const image_descriptor &desc, const sycl::queue &syclQueue)
    : image_mem(desc, syclQueue.get_device(), syclQueue.get_context()) {}

__SYCL_EXPORT sycl::range<3> image_mem::get_range() const {
  auto desc = impl->get_descriptor();
  return {desc.width, desc.height, desc.depth};
}

__SYCL_EXPORT sycl::image_channel_type image_mem::get_channel_type() const {
  return impl->get_descriptor().channel_type;
}

__SYCL_EXPORT sycl::image_channel_order image_mem::get_channel_order() const {
  return impl->get_descriptor().channel_order;
}

__SYCL_EXPORT unsigned int image_mem::get_num_channels() const {
  return sycl::detail::getImageNumberChannels(
      impl->get_descriptor().channel_order);
}

__SYCL_EXPORT image_type image_mem::get_type() const {
  return impl->get_descriptor().type;
}

__SYCL_EXPORT image_mem_handle
image_mem::get_mip_level_mem_handle(const unsigned int level) const {
  return ext::oneapi::experimental::get_mip_level_mem_handle(
      impl->get_handle(), level, impl->get_device(), impl->get_context());
}

__SYCL_EXPORT void destroy_image_handle(unsampled_image_handle &imageHandle,
                                        const sycl::device &syclDevice,
                                        const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_image_handle piImageHandle = imageHandle.raw_handle;

  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextMemUnsampledImageHandleDestroy>(
      C, Device, piImageHandle);
}

__SYCL_EXPORT void destroy_image_handle(unsampled_image_handle &imageHandle,
                                        const sycl::queue &syclQueue) {
  destroy_image_handle(imageHandle, syclQueue.get_device(),
                       syclQueue.get_context());
}

__SYCL_EXPORT void destroy_image_handle(sampled_image_handle &imageHandle,
                                        const sycl::device &syclDevice,
                                        const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_image_handle piImageHandle = imageHandle.raw_handle;

  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextMemSampledImageHandleDestroy>(
      C, Device, piImageHandle);
}

__SYCL_EXPORT void destroy_image_handle(sampled_image_handle &imageHandle,
                                        const sycl::queue &syclQueue) {
  destroy_image_handle(imageHandle, syclQueue.get_device(),
                       syclQueue.get_context());
}

__SYCL_EXPORT image_mem_handle
alloc_image_mem(const image_descriptor &desc, const sycl::device &syclDevice,
                const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  if (desc.type == image_type::mipmap) {
    // Mipmaps must have more than one level
    if (desc.num_levels <= 1)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Mipmap number of levels must be 2 or more");
  } else if (desc.type == image_type::standard) {
    // Non-mipmap images must have only 1 level
    if (desc.num_levels != 1)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Image number of levels must be 1");
  } else {
    // Not an image to allocate
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid image type to allocate");
  }

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  image_mem_handle retHandle;

  // Call impl.
  Plugin->call<sycl::errc::memory_allocation,
               sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, Device, &piFormat, &piDesc, &retHandle.raw_handle);

  return retHandle;
}

__SYCL_EXPORT image_mem_handle alloc_image_mem(const image_descriptor &desc,
                                               const sycl::queue &syclQueue) {
  return alloc_image_mem(desc, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT_DEPRECATED("Distinct mipmap allocs are deprecated. "
                         "Instead use alloc_image_mem().")
image_mem_handle alloc_mipmap_mem(const image_descriptor &desc,
                                  const sycl::device &syclDevice,
                                  const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  // Mipmaps must have more than one level
  if (desc.num_levels <= 1)
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Mipmap number of levels must be 2 or more");

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  // Call impl.
  image_mem_handle retHandle;
  Plugin->call<sycl::errc::memory_allocation,
               sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, Device, &piFormat, &piDesc, &retHandle.raw_handle);

  return retHandle;
}

__SYCL_EXPORT_DEPRECATED("Distinct mipmap allocs are deprecated. "
                         "Instead use alloc_image_mem().")
image_mem_handle alloc_mipmap_mem(const image_descriptor &desc,
                                  const sycl::queue &syclQueue) {
  return alloc_mipmap_mem(desc, syclQueue.get_device(),
                          syclQueue.get_context());
}

__SYCL_EXPORT image_mem_handle get_mip_level_mem_handle(
    const image_mem_handle mipMem, unsigned int level,
    const sycl::device &syclDevice, const sycl::context &syclContext) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  // Call impl.
  image_mem_handle individual_image;
  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextMemMipmapGetLevel>(
      C, Device, mipMem.raw_handle, level, &individual_image.raw_handle);

  return individual_image;
}

__SYCL_EXPORT image_mem_handle
get_mip_level_mem_handle(const image_mem_handle mipMem, unsigned int level,
                         const sycl::queue &syclQueue) {
  return get_mip_level_mem_handle(mipMem, level, syclQueue.get_device(),
                                  syclQueue.get_context());
}

__SYCL_EXPORT void free_image_mem(image_mem_handle memHandle,
                                  image_type imageType,
                                  const sycl::device &syclDevice,
                                  const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  if (memHandle.raw_handle != nullptr) {
    if (imageType == image_type::mipmap) {
      Plugin->call<sycl::errc::memory_allocation,
                   sycl::detail::PiApiKind::piextMemMipmapFree>(
          C, Device, memHandle.raw_handle);
    } else if (imageType == image_type::standard) {
      Plugin->call<sycl::errc::memory_allocation,
                   sycl::detail::PiApiKind::piextMemImageFree>(
          C, Device, memHandle.raw_handle);
    } else {
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Invalid image type to free");
    }
  }
}

__SYCL_EXPORT void free_image_mem(image_mem_handle memHandle,
                                  image_type imageType,
                                  const sycl::queue &syclQueue) {
  free_image_mem(memHandle, imageType, syclQueue.get_device(),
                 syclQueue.get_context());
}

__SYCL_EXPORT_DEPRECATED("Distinct image frees are deprecated. "
                         "Instead use overload that accepts image_type.")
void free_image_mem(image_mem_handle memHandle, const sycl::device &syclDevice,
                    const sycl::context &syclContext) {
  return free_image_mem(memHandle, image_type::standard, syclDevice,
                        syclContext);
}

__SYCL_EXPORT_DEPRECATED("Distinct image frees are deprecated. "
                         "Instead use overload that accepts image_type.")
void free_image_mem(image_mem_handle memHandle, const sycl::queue &syclQueue) {
  free_image_mem(memHandle, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT_DEPRECATED(
    "Distinct mipmap frees are deprecated. "
    "Instead use free_image_mem() that accepts image_type.")
void free_mipmap_mem(image_mem_handle memoryHandle,
                     const sycl::device &syclDevice,
                     const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  Plugin->call<sycl::errc::memory_allocation,
               sycl::detail::PiApiKind::piextMemMipmapFree>(
      C, Device, memoryHandle.raw_handle);
}

__SYCL_EXPORT_DEPRECATED(
    "Distinct mipmap frees are deprecated. "
    "Instead use free_image_mem() that accepts image_type.")
void free_mipmap_mem(image_mem_handle memoryHandle,
                     const sycl::queue &syclQueue) {
  free_mipmap_mem(memoryHandle, syclQueue.get_device(),
                  syclQueue.get_context());
}

__SYCL_EXPORT unsampled_image_handle
create_image(image_mem &imgMem, const image_descriptor &desc,
             const sycl::device &syclDevice, const sycl::context &syclContext) {
  return create_image(imgMem.get_handle(), desc, syclDevice, syclContext);
}

__SYCL_EXPORT unsampled_image_handle
create_image(image_mem &imgMem, const image_descriptor &desc,
             const sycl::queue &syclQueue) {
  return create_image(imgMem.get_handle(), desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

__SYCL_EXPORT unsampled_image_handle
create_image(image_mem_handle memHandle, const image_descriptor &desc,
             const sycl::device &syclDevice, const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  // Call impl.
  pi_image_handle piImageHandle;
  pi_mem piImage;
  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextMemUnsampledImageCreate>(
      C, Device, memHandle.raw_handle, &piFormat, &piDesc, &piImage,
      &piImageHandle);

  return unsampled_image_handle{piImageHandle};
}

__SYCL_EXPORT unsampled_image_handle
create_image(image_mem_handle memHandle, const image_descriptor &desc,
             const sycl::queue &syclQueue) {
  return create_image(memHandle, desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

__SYCL_EXPORT sampled_image_handle
create_image(image_mem_handle memHandle, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext) {
  return create_image(memHandle.raw_handle, 0 /*pitch*/, sampler, desc,
                      syclDevice, syclContext);
}

__SYCL_EXPORT sampled_image_handle
create_image(image_mem_handle memHandle, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue) {
  return create_image(memHandle, sampler, desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

__SYCL_EXPORT sampled_image_handle
create_image(image_mem &imgMem, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext) {
  return create_image(imgMem.get_handle().raw_handle, 0 /*pitch*/, sampler,
                      desc, syclDevice, syclContext);
}

__SYCL_EXPORT sampled_image_handle
create_image(image_mem &imgMem, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue) {
  return create_image(imgMem.get_handle().raw_handle, 0 /*pitch*/, sampler,
                      desc, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT sampled_image_handle
create_image(void *devPtr, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  const pi_sampler_properties sProps[] = {
      PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS,
      static_cast<pi_sampler_properties>(sampler.coordinate),
      PI_SAMPLER_PROPERTIES_ADDRESSING_MODE,
      static_cast<pi_sampler_properties>(sampler.addressing[0]),
      PI_SAMPLER_PROPERTIES_ADDRESSING_MODE,
      static_cast<pi_sampler_properties>(sampler.addressing[1]),
      PI_SAMPLER_PROPERTIES_ADDRESSING_MODE,
      static_cast<pi_sampler_properties>(sampler.addressing[2]),
      PI_SAMPLER_PROPERTIES_FILTER_MODE,
      static_cast<pi_sampler_properties>(sampler.filtering),
      PI_SAMPLER_PROPERTIES_MIP_FILTER_MODE,
      static_cast<pi_sampler_properties>(sampler.mipmap_filtering),
      0};

  pi_sampler piSampler = {};
  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextBindlessImageSamplerCreate>(
      C, sProps, sampler.min_mipmap_level_clamp, sampler.max_mipmap_level_clamp,
      sampler.max_anisotropy, &piSampler);

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat, pitch);

  // Call impl.
  pi_mem piImage;
  pi_image_handle piImageHandle;
  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextMemSampledImageCreate>(
      C, Device, devPtr, &piFormat, &piDesc, piSampler, &piImage,
      &piImageHandle);

  return sampled_image_handle{piImageHandle};
}

__SYCL_EXPORT sampled_image_handle
create_image(void *devPtr, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue) {
  return create_image(devPtr, pitch, sampler, desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

template <>
__SYCL_EXPORT interop_mem_handle import_external_memory<external_mem_fd>(
    external_mem_descriptor<external_mem_fd> externalMem,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  pi_interop_mem_handle piInteropMem;
  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemImportOpaqueFD>(
      C, Device, externalMem.size_in_bytes,
      externalMem.external_handle.file_descriptor, &piInteropMem);

  return interop_mem_handle{piInteropMem};
}

template <>
__SYCL_EXPORT interop_mem_handle import_external_memory<external_mem_fd>(
    external_mem_descriptor<external_mem_fd> externalMem,
    const sycl::queue &syclQueue) {
  return import_external_memory<external_mem_fd>(
      externalMem, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT
image_mem_handle map_external_image_memory(interop_mem_handle memHandle,
                                           const image_descriptor &desc,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  pi_interop_mem_handle piInteropMem{memHandle.raw_handle};

  image_mem_handle retHandle;
  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemMapExternalArray>(
      C, Device, &piFormat, &piDesc, piInteropMem, &retHandle.raw_handle);

  return image_mem_handle{retHandle};
}

__SYCL_EXPORT
image_mem_handle map_external_image_memory(interop_mem_handle memHandle,
                                           const image_descriptor &desc,
                                           const sycl::queue &syclQueue) {
  return map_external_image_memory(memHandle, desc, syclQueue.get_device(),
                                   syclQueue.get_context());
}

__SYCL_EXPORT_DEPRECATED("map_external_memory_array is deprecated."
                         "use map_external_image_memory")
image_mem_handle map_external_memory_array(interop_mem_handle memHandle,
                                           const image_descriptor &desc,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext) {
  return map_external_image_memory(memHandle, desc, syclDevice, syclContext);
}

__SYCL_EXPORT_DEPRECATED("map_external_memory_array is deprecated."
                         "use map_external_image_memory")
image_mem_handle map_external_memory_array(interop_mem_handle memHandle,
                                           const image_descriptor &desc,
                                           const sycl::queue &syclQueue) {
  return map_external_memory_array(memHandle, desc, syclQueue.get_device(),
                                   syclQueue.get_context());
}

__SYCL_EXPORT void release_external_memory(interop_mem_handle interopMem,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemReleaseInterop>(
      C, Device, (pi_interop_mem_handle)interopMem.raw_handle);
}

__SYCL_EXPORT void release_external_memory(interop_mem_handle interopMem,
                                           const sycl::queue &syclQueue) {
  release_external_memory(interopMem, syclQueue.get_device(),
                          syclQueue.get_context());
}

template <>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<external_semaphore_fd> externalSemaphoreDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();

  pi_interop_semaphore_handle piInteropSemaphore;

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextImportExternalSemaphoreOpaqueFD>(
      C, Device, externalSemaphoreDesc.external_handle.file_descriptor,
      &piInteropSemaphore);

  return interop_semaphore_handle{piInteropSemaphore};
}

template <>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<external_semaphore_fd> externalSemaphoreDesc,
    const sycl::queue &syclQueue) {
  return import_external_semaphore(
      externalSemaphoreDesc, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT void
destroy_external_semaphore(interop_semaphore_handle semaphoreHandle,
                           const sycl::device &syclDevice,
                           const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_context C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  pi_device Device = DevImpl->getHandleRef();

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextDestroyExternalSemaphore>(
      C, Device, (pi_interop_semaphore_handle)semaphoreHandle.raw_handle);
}

__SYCL_EXPORT void
destroy_external_semaphore(interop_semaphore_handle semaphoreHandle,
                           const sycl::queue &syclQueue) {
  destroy_external_semaphore(semaphoreHandle, syclQueue.get_device(),
                             syclQueue.get_context());
}

__SYCL_EXPORT sycl::range<3> get_image_range(const image_mem_handle memHandle,
                                             const sycl::device &syclDevice,
                                             const sycl::context &syclContext) {
  std::ignore = syclDevice;
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  size_t Width, Height, Depth;

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_WIDTH, &Width, nullptr);

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_HEIGHT, &Height, nullptr);

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_DEPTH, &Depth, nullptr);

  return {Width, Height, Depth};
}

__SYCL_EXPORT sycl::range<3> get_image_range(const image_mem_handle memHandle,
                                             const sycl::queue &syclQueue) {
  return get_image_range(memHandle, syclQueue.get_device(),
                         syclQueue.get_context());
}

__SYCL_EXPORT sycl::image_channel_type
get_image_channel_type(const image_mem_handle memHandle,
                       const sycl::device &syclDevice,
                       const sycl::context &syclContext) {
  std::ignore = syclDevice;
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  pi_image_format PIFormat;

  Plugin->call<sycl::errc::invalid,
               sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_FORMAT, &PIFormat, nullptr);

  image_channel_type ChannelType =
      sycl::detail::convertChannelType(PIFormat.image_channel_data_type);

  return ChannelType;
}

__SYCL_EXPORT sycl::image_channel_type
get_image_channel_type(const image_mem_handle memHandle,
                       const sycl::queue &syclQueue) {
  return get_image_channel_type(memHandle, syclQueue.get_device(),
                                syclQueue.get_context());
}

__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const sycl::device &syclDevice,
                                         const sycl::context &syclContext) {
  void *RetVal = nullptr;
  if (widthInBytes == 0 || height == 0 || elementSizeBytes == 0) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                          "Cannot allocate pitched memory with zero size!");
  }

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  if (CtxImpl->is_host()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                          "Cannot allocate pitched memory on host!");
  }

  pi_context PiContext = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_device PiDevice;

  PiDevice = sycl::detail::getSyclObjImpl(syclDevice)->getHandleRef();

  Plugin->call<sycl::errc::memory_allocation,
               sycl::detail::PiApiKind::piextUSMPitchedAlloc>(
      &RetVal, resultPitch, PiContext, PiDevice, nullptr, widthInBytes, height,
      elementSizeBytes);

  return RetVal;
}

__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const sycl::queue &syclQueue) {
  return pitched_alloc_device(resultPitch, widthInBytes, height,
                              elementSizeBytes, syclQueue.get_device(),
                              syclQueue.get_context());
}

__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         const image_descriptor &desc,
                                         const sycl::queue &syclQueue) {
  return pitched_alloc_device(resultPitch, desc, syclQueue.get_device(),
                              syclQueue.get_context());
}

__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         const image_descriptor &desc,
                                         const sycl::device &syclDevice,
                                         const sycl::context &syclContext) {
  uint8_t numChannels =
      sycl::detail::getImageNumberChannels(desc.channel_order);
  unsigned int elementSizeBytes =
      sycl::detail::getImageElementSize(numChannels, desc.channel_type);

  size_t widthInBytes = desc.width * elementSizeBytes;
  size_t height = desc.height;

  return pitched_alloc_device(resultPitch, widthInBytes, height,
                              elementSizeBytes, syclDevice, syclContext);
}

__SYCL_EXPORT unsigned int
get_image_num_channels(const image_mem_handle memHandle,
                       const sycl::device &syclDevice,
                       const sycl::context &syclContext) {
  std::ignore = syclDevice;

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_image_format PIFormat;

  Plugin->call<sycl::errc::runtime,
               sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_FORMAT, &PIFormat, nullptr);

  image_channel_order Order =
      sycl::detail::convertChannelOrder(PIFormat.image_channel_order);

  return static_cast<unsigned int>(sycl::detail::getImageNumberChannels(Order));
}

__SYCL_EXPORT unsigned int
get_image_num_channels(const image_mem_handle memHandle,
                       const sycl::queue &syclQueue) {
  return get_image_num_channels(memHandle, syclQueue.get_device(),
                                syclQueue.get_context());
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
