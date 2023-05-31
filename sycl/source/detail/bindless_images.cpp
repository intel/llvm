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
#include <detail/sampler_impl.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

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
      sycl::_V1::detail::convertChannelType(desc.channel_type);
  piFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(desc.channel_order);
}

__SYCL_EXPORT
image_mem::image_mem(const sycl::context &syclContext,
                     const image_descriptor &desc)
    : handle{nullptr}, descriptor(desc), syclContext(syclContext) {
  if (desc.type == image_type::mipmap) {
    handle = alloc_mipmap_mem(syclContext, descriptor);
  } else {
    handle = alloc_image_mem(syclContext, descriptor);
  }
}

__SYCL_EXPORT
image_mem::image_mem(image_mem &&rhs) noexcept
    : handle(std::move(rhs.handle)), descriptor(std::move(rhs.descriptor)),
      syclContext(std::move(rhs.syclContext)) {
  rhs.handle = {nullptr};
  rhs.descriptor.width = 0;
  rhs.descriptor.height = 0;
  rhs.descriptor.depth = 0;
}

__SYCL_EXPORT image_mem::~image_mem() {
  if (handle.raw_handle != nullptr) {
    if (this->descriptor.type == image_type::mipmap) {
      free_mipmap_mem(syclContext, handle);
    } else {
      free_image_mem(syclContext, handle);
    }
  }
}

__SYCL_EXPORT image_mem &image_mem::operator=(image_mem &&rhs) noexcept {
  if (this != &rhs && handle.raw_handle != nullptr) {
    if (this->descriptor.type == image_type::mipmap) {
      free_mipmap_mem(syclContext, handle);
    } else {
      free_image_mem(syclContext, handle);
    }
    this->syclContext = rhs.syclContext;
    this->descriptor = rhs.descriptor;
    this->handle = rhs.handle;
    rhs.handle = {nullptr};
    rhs.descriptor.width = 0;
    rhs.descriptor.height = 0;
    rhs.descriptor.depth = 0;
  }
  return *this;
}

__SYCL_EXPORT sycl::range<3> image_mem::get_range() const {
  return {descriptor.width, descriptor.height, descriptor.depth};
}

__SYCL_EXPORT sycl::image_channel_type image_mem::get_channel_type() const {
  return descriptor.channel_type;
}

__SYCL_EXPORT sycl::image_channel_order image_mem::get_channel_order() const {
  return descriptor.channel_order;
}

__SYCL_EXPORT unsigned int image_mem::get_num_channels() const {
  return sycl::detail::getImageNumberChannels(descriptor.channel_order);
}

__SYCL_EXPORT unsigned int image_mem::get_flags() const {
  return sycl::ext::oneapi::experimental::get_image_flags(this->get_context(),
                                                          this->get_handle());
}

__SYCL_EXPORT image_type image_mem::get_type() const { return descriptor.type; }

__SYCL_EXPORT image_mem_handle
image_mem::get_mip_level(const unsigned int level) const {
  return ext::oneapi::experimental::get_mip_level(this->get_context(), *this,
                                                  level);
}

__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        unsampled_image_handle &imageHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_image_handle piImageHandle = imageHandle.raw_handle;

  Error = Plugin->call_nocheck<
      sycl::detail::PiApiKind::piextMemUnsampledImageHandleDestroy>(
      C, piImageHandle);

  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error("Failed to destroy image handle!",
                              PI_ERROR_INVALID_MEM_OBJECT);
  }
}

__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        sampled_image_handle &imageHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_image_handle piImageHandle = imageHandle.raw_handle;

  Error = Plugin->call_nocheck<
      sycl::detail::PiApiKind::piextMemSampledImageHandleDestroy>(
      C, piImageHandle);

  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error("Failed to destroy image handle!",
                              PI_ERROR_INVALID_MEM_OBJECT);
  }
}

__SYCL_EXPORT image_mem_handle alloc_image_mem(const sycl::context &syclContext,
                                               const image_descriptor &desc) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Non-mipmap images must have only 1 level
  if (desc.num_levels != 1)
    throw sycl::runtime_error("Image number of levels must be 1",
                              PI_ERROR_INVALID_MEM_OBJECT);

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  image_mem_handle retHandle;

  // Call impl.
  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, &piFormat, &piDesc, &retHandle.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::memory_allocation_error("Failed to allocate image memory!",
                                        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }

  return retHandle;
}

__SYCL_EXPORT image_mem_handle alloc_mipmap_mem(
    const sycl::context &syclContext, const image_descriptor &desc) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Mipmaps must have more than one level
  if (desc.num_levels <= 1)
    throw sycl::runtime_error("Mipmap number of levels must be 2 or more",
                              PI_ERROR_INVALID_MEM_OBJECT);

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  // Call impl.
  image_mem_handle retHandle;
  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, &piFormat, &piDesc, &retHandle.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::memory_allocation_error("Failed to allocate mipmap memory!",
                                        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }

  return retHandle;
}

__SYCL_EXPORT image_mem_handle get_mip_level(const sycl::context &syclContext,
                                             const image_mem &mipMem,
                                             unsigned int level) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Call impl.
  image_mem_handle individual_image;
  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemMipmapGetLevel>(
      C, mipMem.get_handle().raw_handle, level, &individual_image.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error("Failed to retrieve a mipmap image level!",
                              PI_ERROR_INVALID_MEM_OBJECT);
  }

  return individual_image;
}

__SYCL_EXPORT void free_image_mem(const sycl::context &syclContext,
                                  image_mem_handle memHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  auto Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageFree>(
      C, memHandle.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::memory_allocation_error("Failed to free image memory!",
                                        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }
}

__SYCL_EXPORT void free_mipmap_mem(const sycl::context &syclContext,
                                   image_mem_handle memoryHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  auto Error =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemMipmapFree>(
          C, memoryHandle.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::memory_allocation_error("Failed to free mipmap memory!",
                                        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }

  return;
}

__SYCL_EXPORT unsampled_image_handle
create_image(const sycl::context &syclContext, image_mem &imgMem,
             const image_descriptor &desc) {
  return create_image(syclContext, imgMem.get_handle(), desc);
}

__SYCL_EXPORT unsampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             const image_descriptor &desc) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  // Call impl.
  pi_image_handle piImageHandle;
  pi_mem piImage;
  Error =
      Plugin
          ->call_nocheck<sycl::detail::PiApiKind::piextMemUnsampledImageCreate>(
              C, memHandle.raw_handle, &piFormat, &piDesc, &piImage,
              &piImageHandle);

  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error("Failed to create unsampled image handle!",
                              PI_ERROR_INVALID_MEM_OBJECT);
  }

  return unsampled_image_handle{piImageHandle};
}

__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             const sampler &sampler, const image_descriptor &desc) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Obtain pi_sampler
  const std::shared_ptr<sycl::detail::sampler_impl> SamplerImpl =
      sycl::detail::getSyclObjImpl(sampler);
  pi_sampler piSampler = SamplerImpl->getOrCreateSampler(syclContext);

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  // Call impl.
  pi_image_handle piImageHandle;
  pi_mem piImage;
  Error =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemSampledImageCreate>(
          C, memHandle.raw_handle, &piFormat, &piDesc, piSampler, &piImage,
          &piImageHandle);

  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error("Failed to create sampled image!",
                              PI_ERROR_INVALID_MEM_OBJECT);
  }

  return sampled_image_handle{piImageHandle};
}

__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, image_mem &imgMem,
             const sampler &sampler, const image_descriptor &desc) {
  return create_image(syclContext, imgMem.get_handle().raw_handle, 0 /*pitch*/,
                      sampler, desc);
}

__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, void *devPtr, size_t pitch,
             const sampler &sampler, const image_descriptor &desc) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Obtain pi_sampler
  std::shared_ptr<sycl::detail::sampler_impl> SamplerImpl =
      sycl::detail::getSyclObjImpl(sampler);
  pi_sampler piSampler = SamplerImpl->getOrCreateSampler(syclContext);

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat, pitch);

  // Call impl.
  pi_mem piImage;
  pi_image_handle piImageHandle;
  Error =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemSampledImageCreate>(
          C, devPtr, &piFormat, &piDesc, piSampler, &piImage, &piImageHandle);

  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error("Failed to create sampled image handle!",
                              PI_ERROR_INVALID_MEM_OBJECT);
  }
  return sampled_image_handle{piImageHandle};
}

template <>
__SYCL_EXPORT interop_mem_handle import_external_memory<external_mem_fd>(
    const sycl::context &syclContext,
    external_mem_descriptor<external_mem_fd> externalMem) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();

  pi_result Error;

  pi_interop_mem_handle piInteropMem;
  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImportOpaqueFD>(
      C, externalMem.size_in_bytes, externalMem.external_handle.file_descriptor,
      &piInteropMem);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid external memory handle passed to `import_external_memory`");
  }

  return interop_mem_handle{piInteropMem};
}

__SYCL_EXPORT image_mem_handle map_external_memory_array(
    const sycl::context &syclContext, interop_mem_handle memHandle,
    const image_descriptor &desc) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  pi_image_format piFormat;
  populate_pi_structs(desc, piDesc, piFormat);

  pi_interop_mem_handle piInteropMem{memHandle.raw_handle};

  image_mem_handle retHandle;
  Error =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemMapExternalArray>(
          C, &piFormat, &piDesc, piInteropMem, &retHandle.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid interop memory handle passed to `map_external_memory_array`");
  }

  return image_mem_handle{retHandle};
}

__SYCL_EXPORT void release_external_memory(const sycl::context &syclContext,
                                           interop_mem_handle interopMem) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemReleaseInterop>(
      C, (pi_interop_mem_handle)interopMem.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid interop memory handle passed to `release_external_memory`");
  }
}

template <>
__SYCL_EXPORT interop_semaphore_handle
import_external_semaphore(const sycl::context &syclContext,
                          external_semaphore_descriptor<external_semaphore_fd>
                              externalSemaphoreDesc) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_context C = CtxImpl->getHandleRef();
  pi_result Error;

  pi_interop_semaphore_handle piInteropSemaphore;

  Error = Plugin->call_nocheck<
      sycl::detail::PiApiKind::piextImportExternalSemaphoreOpaqueFD>(
      C, externalSemaphoreDesc.external_handle.file_descriptor,
      &piInteropSemaphore);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid semaphore handle passed to `import_external_semaphore`");
  }

  return interop_semaphore_handle{piInteropSemaphore};
}

__SYCL_EXPORT sycl::event
wait_external_semaphore(const sycl::queue &syclQueue,
                        interop_semaphore_handle semaphoreHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclQueue.get_context());
  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(syclQueue);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_queue PiQueue = QueueImpl->getHandleRef();
  pi_result Error;
  sycl::event syclEvent;
  std::shared_ptr<sycl::detail::event_impl> EvtImpl =
      sycl::detail::getSyclObjImpl(syclEvent);
  pi_event PiEvent = EvtImpl->getHandleRef();

  Error =
      Plugin->call_nocheck<sycl::detail::PiApiKind::piextWaitExternalSemaphore>(
          PiQueue, (pi_interop_semaphore_handle)semaphoreHandle.raw_handle,
          0 /* num_events_in_wait_list */, nullptr /* event_wait_list */,
          &PiEvent);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid semaphore handle passed to `wait_external_semaphore`");
  }

  return syclEvent;
}

__SYCL_EXPORT sycl::event
signal_external_semaphore(const sycl::queue &syclQueue,
                          interop_semaphore_handle semaphoreHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclQueue.get_context());
  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(syclQueue);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_queue PiQueue = QueueImpl->getHandleRef();
  pi_result Error;
  sycl::event syclEvent;
  std::shared_ptr<sycl::detail::event_impl> EvtImpl =
      sycl::detail::getSyclObjImpl(syclEvent);
  pi_event PiEvent = EvtImpl->getHandleRef();

  Error =
      Plugin
          ->call_nocheck<sycl::detail::PiApiKind::piextSignalExternalSemaphore>(
              PiQueue, (pi_interop_semaphore_handle)semaphoreHandle.raw_handle,
              0 /* num_events_in_wait_list */, nullptr /* event_wait_list */,
              &PiEvent);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid semaphore handle passed to `signal_external_semaphore`");
  }

  return syclEvent;
}

__SYCL_EXPORT void
destroy_external_semaphore(const sycl::context &syclContext,
                           interop_semaphore_handle semaphoreHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_context C = CtxImpl->getHandleRef();
  pi_result Error;

  Error = Plugin->call_nocheck<
      sycl::detail::PiApiKind::piextDestroyExternalSemaphore>(
      C, (pi_interop_semaphore_handle)semaphoreHandle.raw_handle);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid semaphore handle passed to `destroy_external_semaphore`");
  }
}

__SYCL_EXPORT sycl::range<3> get_image_range(const sycl::context &syclContext,
                                             const image_mem_handle memHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  size_t Width, Height, Depth;

  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_WIDTH, &Width, nullptr);

  if (Error == PI_SUCCESS)
    Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
        memHandle.raw_handle, PI_IMAGE_INFO_HEIGHT, &Height, nullptr);

  if (Error == PI_SUCCESS)
    Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
        memHandle.raw_handle, PI_IMAGE_INFO_DEPTH, &Depth, nullptr);

  if (Error != PI_SUCCESS) {
    throw sycl::exception(sycl::errc::invalid,
                          "Invalid memory handle passed to `get_image_range`");
  }

  return {Width, Height, Depth};
}

__SYCL_EXPORT unsigned int get_image_flags(const sycl::context &syclContext,
                                           const image_mem_handle memHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  unsigned int Flags;

  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_FLAGS, &Flags, nullptr);
  if (Error != PI_SUCCESS) {
    throw sycl::exception(sycl::errc::invalid,
                          "Invalid memory handle passed to `get_image_flags`");
  }

  return Flags;
}

__SYCL_EXPORT sycl::image_channel_type
get_image_channel_type(const sycl::context &syclContext,
                       const image_mem_handle memHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_format PIFormat;

  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_FORMAT, &PIFormat, nullptr);
  if (Error != PI_SUCCESS) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Invalid memory handle passed to `get_image_channel_type`");
  }

  image_channel_type ChannelType =
      sycl::detail::convertChannelType(PIFormat.image_channel_data_type);

  return ChannelType;
}

__SYCL_EXPORT void *pitched_alloc_device(size_t *resultPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elementSizeBytes,
                                         const sycl::device &syclDevice,
                                         const sycl::context &syclContext) {
  void *RetVal = nullptr;
  if (widthInBytes == 0 || height == 0 || elementSizeBytes == 0) {
    throw sycl::memory_allocation_error(
        "Cannot allocate pitched memory with zero size!",
        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  if (CtxImpl->is_host()) {
    throw sycl::memory_allocation_error(
        "Cannot allocate pitched memory on host!",
        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }

  pi_context PiContext = CtxImpl->getHandleRef();
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_device PiDevice;

  PiDevice = sycl::detail::getSyclObjImpl(syclDevice)->getHandleRef();

  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextUSMPitchedAlloc>(
      &RetVal, resultPitch, PiContext, PiDevice, nullptr, widthInBytes, height,
      elementSizeBytes);

  if (Error != PI_SUCCESS) {
    throw sycl::memory_allocation_error("Failed to allocate pitched memory!",
                                        PI_ERROR_MEM_OBJECT_ALLOCATION_FAILURE);
  }

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
get_image_num_channels(const sycl::context &syclContext,
                       const image_mem_handle memHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::PluginPtr &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_format PIFormat;

  Error = Plugin->call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      memHandle.raw_handle, PI_IMAGE_INFO_FORMAT, &PIFormat, nullptr);
  if (Error != PI_SUCCESS) {
    throw sycl::runtime_error(
        "Failed to get the number of channels from image memory!",
        PI_ERROR_INVALID_MEM_OBJECT);
  }

  image_channel_order Order =
      sycl::detail::convertChannelOrder(PIFormat.image_channel_order);

  return static_cast<unsigned int>(sycl::detail::getImageNumberChannels(Order));
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
