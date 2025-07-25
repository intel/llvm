//==----------- bindless_images.hpp --- SYCL bindless images ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/sampler.hpp>

#include <detail/context_impl.hpp>
#include <detail/image_impl.hpp>
#include <detail/queue_impl.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

void populate_ur_structs(const image_descriptor &desc, ur_image_desc_t &urDesc,
                         ur_image_format_t &urFormat, size_t pitch = 0) {
  urDesc = {};
  urDesc.stype = UR_STRUCTURE_TYPE_IMAGE_DESC;
  urDesc.width = desc.width;
  urDesc.height = desc.height;
  urDesc.depth = desc.depth;

  if (desc.array_size > 1) {
    // Image array or cubemap
    urDesc.type =
        desc.type == image_type::cubemap  ? UR_MEM_TYPE_IMAGE_CUBEMAP_EXP
        : desc.type == image_type::gather ? UR_MEM_TYPE_IMAGE_GATHER_EXP
        : desc.height > 0                 ? UR_MEM_TYPE_IMAGE2D_ARRAY
                                          : UR_MEM_TYPE_IMAGE1D_ARRAY;
  } else {
    urDesc.type = desc.depth > 0 ? UR_MEM_TYPE_IMAGE3D
                                 : (desc.height > 0 ? UR_MEM_TYPE_IMAGE2D
                                                    : UR_MEM_TYPE_IMAGE1D);
  }

  urDesc.rowPitch = pitch;
  urDesc.arraySize = desc.array_size;
  urDesc.slicePitch = 0;
  urDesc.numMipLevel = desc.num_levels;
  urDesc.numSamples = 0;

  urFormat = {};
  urFormat.channelType = sycl::detail::convertChannelType(desc.channel_type);
  urFormat.channelOrder = sycl::detail::convertChannelOrder(
      sycl::ext::oneapi::experimental::detail::get_image_default_channel_order(
          desc.num_channels));
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

__SYCL_EXPORT_DEPRECATED("get_channel_order() is deprecated. "
                         "Instead use get_channel_num().")
sycl::image_channel_order image_mem::get_channel_order() const {
  return sycl::ext::oneapi::experimental::detail::
      get_image_default_channel_order(impl->get_descriptor().num_channels);
}

__SYCL_EXPORT unsigned int image_mem::get_num_channels() const {
  return impl->get_descriptor().num_channels;
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
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<
      sycl::errc::runtime,
      sycl::detail::UrApiKind::urBindlessImagesUnsampledImageHandleDestroyExp>(
      urCtx, urDevice, imageHandle.raw_handle);
}

__SYCL_EXPORT void destroy_image_handle(unsampled_image_handle &imageHandle,
                                        const sycl::queue &syclQueue) {
  destroy_image_handle(imageHandle, syclQueue.get_device(),
                       syclQueue.get_context());
}

__SYCL_EXPORT void destroy_image_handle(sampled_image_handle &imageHandle,
                                        const sycl::device &syclDevice,
                                        const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<
      sycl::errc::runtime,
      sycl::detail::UrApiKind::urBindlessImagesSampledImageHandleDestroyExp>(
      urCtx, urDevice, imageHandle.raw_handle);
}

__SYCL_EXPORT void destroy_image_handle(sampled_image_handle &imageHandle,
                                        const sycl::queue &syclQueue) {
  destroy_image_handle(imageHandle, syclQueue.get_device(),
                       syclQueue.get_context());
}

__SYCL_EXPORT image_mem_handle
alloc_image_mem(const image_descriptor &desc, const sycl::device &syclDevice,
                const sycl::context &syclContext) {
  desc.verify();

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  image_mem_handle retHandle = {};

  // Call impl.
  Adapter->call<sycl::errc::memory_allocation,
                sycl::detail::UrApiKind::urBindlessImagesImageAllocateExp>(
      urCtx, urDevice, &urFormat, &urDesc,
      reinterpret_cast<ur_exp_image_mem_native_handle_t *>(
          &retHandle.raw_handle));

  return retHandle;
}

__SYCL_EXPORT image_mem_handle alloc_image_mem(const image_descriptor &desc,
                                               const sycl::queue &syclQueue) {
  return alloc_image_mem(desc, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT image_mem_handle get_mip_level_mem_handle(
    const image_mem_handle mipMem, unsigned int level,
    const sycl::device &syclDevice, const sycl::context &syclContext) {

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  // Call impl.
  image_mem_handle individual_image = {};
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urBindlessImagesMipmapGetLevelExp>(
      urCtx, urDevice, mipMem.raw_handle, level, &individual_image.raw_handle);

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
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  if (memHandle.raw_handle != 0) {
    if (imageType == image_type::mipmap) {
      Adapter->call<sycl::errc::memory_allocation,
                    sycl::detail::UrApiKind::urBindlessImagesMipmapFreeExp>(
          urCtx, urDevice, memHandle.raw_handle);
    } else {
      Adapter->call<sycl::errc::memory_allocation,
                    sycl::detail::UrApiKind::urBindlessImagesImageFreeExp>(
          urCtx, urDevice, memHandle.raw_handle);
    }
  }
}

__SYCL_EXPORT void free_image_mem(image_mem_handle memHandle,
                                  image_type imageType,
                                  const sycl::queue &syclQueue) {
  free_image_mem(memHandle, imageType, syclQueue.get_device(),
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
  desc.verify();

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  // Call impl.
  ur_exp_image_native_handle_t urImageHandle = 0;
  Adapter
      ->call<sycl::errc::runtime,
             sycl::detail::UrApiKind::urBindlessImagesUnsampledImageCreateExp>(
          urCtx, urDevice, memHandle.raw_handle, &urFormat, &urDesc,
          &urImageHandle);

  return unsampled_image_handle{urImageHandle};
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
  return create_image(reinterpret_cast<void *>(memHandle.raw_handle),
                      0 /*pitch*/, sampler, desc, syclDevice, syclContext);
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
  return create_image(reinterpret_cast<void *>(imgMem.get_handle().raw_handle),
                      0 /*pitch*/, sampler, desc, syclDevice, syclContext);
}

__SYCL_EXPORT sampled_image_handle
create_image(image_mem &imgMem, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue) {
  return create_image(reinterpret_cast<void *>(imgMem.get_handle().raw_handle),
                      0 /*pitch*/, sampler, desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

inline ur_sampler_addressing_mode_t
translate_addressing_mode(sycl::addressing_mode Mode) {
  switch (Mode) {
  case sycl::addressing_mode::mirrored_repeat:
    return UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
  case sycl::addressing_mode::repeat:
    return UR_SAMPLER_ADDRESSING_MODE_REPEAT;
  case sycl::addressing_mode::clamp_to_edge:
    return UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
  case sycl::addressing_mode::clamp:
    return UR_SAMPLER_ADDRESSING_MODE_CLAMP;
  case sycl::addressing_mode::none:
  default:
    return UR_SAMPLER_ADDRESSING_MODE_NONE;
  }
}

inline ur_sampler_filter_mode_t
translate_filter_mode(sycl::filtering_mode Mode) {
  switch (Mode) {
  case sycl::filtering_mode::linear:
    return UR_SAMPLER_FILTER_MODE_LINEAR;
  case sycl::filtering_mode::nearest:
    return UR_SAMPLER_FILTER_MODE_NEAREST;
  }
  return UR_SAMPLER_FILTER_MODE_FORCE_UINT32;
}

inline ur_exp_sampler_cubemap_filter_mode_t
translate_cubemap_filter_mode(cubemap_filtering_mode Mode) {
  switch (Mode) {
  case cubemap_filtering_mode::disjointed:
    return UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_DISJOINTED;
  case cubemap_filtering_mode::seamless:
    return UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_SEAMLESS;
  }
  return UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_FORCE_UINT32;
}

__SYCL_EXPORT sampled_image_handle
create_image(void *devPtr, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::device &syclDevice,
             const sycl::context &syclContext) {
  desc.verify();

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_sampler_desc_t UrSamplerProps{
      UR_STRUCTURE_TYPE_SAMPLER_DESC, nullptr,
      sampler.coordinate == coordinate_normalization_mode::normalized,
      translate_addressing_mode(sampler.addressing[0]),
      translate_filter_mode(sampler.filtering)};

  ur_exp_sampler_mip_properties_t UrMipProps{
      UR_STRUCTURE_TYPE_EXP_SAMPLER_MIP_PROPERTIES,
      nullptr,
      sampler.min_mipmap_level_clamp,
      sampler.max_mipmap_level_clamp,
      sampler.max_anisotropy,
      translate_filter_mode(sampler.mipmap_filtering)};
  UrSamplerProps.pNext = &UrMipProps;

  ur_exp_sampler_addr_modes_t UrAddrModes{
      UR_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES,
      nullptr,
      {translate_addressing_mode(sampler.addressing[0]),
       translate_addressing_mode(sampler.addressing[1]),
       translate_addressing_mode(sampler.addressing[2])}};
  UrMipProps.pNext = &UrAddrModes;

  ur_exp_sampler_cubemap_properties_t UrCubemapProps{
      UR_STRUCTURE_TYPE_EXP_SAMPLER_CUBEMAP_PROPERTIES, nullptr,
      translate_cubemap_filter_mode(sampler.cubemap_filtering)};
  UrAddrModes.pNext = &UrCubemapProps;

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat, pitch);

  // Call impl.
  ur_exp_image_native_handle_t urImageHandle = 0;
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urBindlessImagesSampledImageCreateExp>(
      urCtx, urDevice,
      reinterpret_cast<ur_exp_image_mem_native_handle_t>(devPtr), &urFormat,
      &urDesc, &UrSamplerProps, &urImageHandle);

  return sampled_image_handle{urImageHandle};
}

__SYCL_EXPORT sampled_image_handle
create_image(void *devPtr, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue) {
  return create_image(devPtr, pitch, sampler, desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

namespace detail {

/**
 * Converts SYCL external_mem_handle_type to the corresponding UR type.
 *
 * Note that this function does a simple conversion
 * and doesn't check the result validity for any specific scenario.
 */
constexpr ur_exp_external_mem_type_t
to_ur_type(external_mem_handle_type externalMemHandleType) {
  switch (externalMemHandleType) {
  case external_mem_handle_type::opaque_fd:
    return UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;
  case external_mem_handle_type::dma_buf:
    return UR_EXP_EXTERNAL_MEM_TYPE_DMA_BUF;
  case external_mem_handle_type::win32_nt_handle:
    return UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT;
  case external_mem_handle_type::win32_nt_dx11_resource:
    return UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX11_RESOURCE;
  case external_mem_handle_type::win32_nt_dx12_resource:
    return UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE;
  default:
    // This ensures that all cases have to be handled
    assert(false && "Invalid memory handle type");
    return UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD; // Fallback
  }
}

} // namespace detail

template <>
__SYCL_EXPORT external_mem import_external_memory<resource_fd>(
    external_mem_descriptor<resource_fd> externalMemDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_handle_t urExternalMem = nullptr;
  ur_exp_file_descriptor_t urFileDescriptor = {};
  urFileDescriptor.stype = UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR;
  urFileDescriptor.fd = externalMemDesc.external_resource.file_descriptor;
  ur_exp_external_mem_desc_t urExternalMemDescriptor = {};
  urExternalMemDescriptor.stype = UR_STRUCTURE_TYPE_EXP_EXTERNAL_MEM_DESC;
  urExternalMemDescriptor.pNext = &urFileDescriptor;

  const auto urHandleType = detail::to_ur_type(externalMemDesc.handle_type);
  if ((urHandleType != UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD) &&
      (urHandleType != UR_EXP_EXTERNAL_MEM_TYPE_DMA_BUF)) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid memory handle type");
  }

  Adapter
      ->call<sycl::errc::invalid,
             sycl::detail::UrApiKind::urBindlessImagesImportExternalMemoryExp>(
          urCtx, urDevice, externalMemDesc.size_in_bytes, urHandleType,
          &urExternalMemDescriptor, &urExternalMem);

  return external_mem{urExternalMem};
}

template <>
__SYCL_EXPORT external_mem import_external_memory<resource_fd>(
    external_mem_descriptor<resource_fd> externalMemDesc,
    const sycl::queue &syclQueue) {
  return import_external_memory<resource_fd>(
      externalMemDesc, syclQueue.get_device(), syclQueue.get_context());
}

template <>
__SYCL_EXPORT external_mem import_external_memory<resource_win32_handle>(
    external_mem_descriptor<resource_win32_handle> externalMemDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_handle_t urExternalMem = nullptr;
  ur_exp_win32_handle_t urWin32Handle = {};
  urWin32Handle.stype = UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE;
  urWin32Handle.handle = externalMemDesc.external_resource.handle;
  ur_exp_external_mem_desc_t urExternalMemDescriptor{};
  urExternalMemDescriptor.stype = UR_STRUCTURE_TYPE_EXP_EXTERNAL_MEM_DESC;
  urExternalMemDescriptor.pNext = &urWin32Handle;

  const auto urHandleType = detail::to_ur_type(externalMemDesc.handle_type);
  if ((urHandleType != UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT) &&
      (urHandleType != UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX11_RESOURCE) &&
      (urHandleType != UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE)) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid memory handle type");
  }

  Adapter
      ->call<sycl::errc::invalid,
             sycl::detail::UrApiKind::urBindlessImagesImportExternalMemoryExp>(
          urCtx, urDevice, externalMemDesc.size_in_bytes, urHandleType,
          &urExternalMemDescriptor, &urExternalMem);

  return external_mem{urExternalMem};
}

template <>
__SYCL_EXPORT external_mem import_external_memory<resource_win32_handle>(
    external_mem_descriptor<resource_win32_handle> externalMemDesc,
    const sycl::queue &syclQueue) {
  return import_external_memory<resource_win32_handle>(
      externalMemDesc, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT
image_mem_handle map_external_image_memory(external_mem extMem,
                                           const image_descriptor &desc,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext) {
  desc.verify();

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  ur_exp_external_mem_handle_t urExternalMem{extMem.raw_handle};

  image_mem_handle retHandle = {};
  Adapter->call<sycl::errc::invalid,
                sycl::detail::UrApiKind::urBindlessImagesMapExternalArrayExp>(
      urCtx, urDevice, &urFormat, &urDesc, urExternalMem,
      &retHandle.raw_handle);

  return image_mem_handle{retHandle};
}

__SYCL_EXPORT
image_mem_handle map_external_image_memory(external_mem extMem,
                                           const image_descriptor &desc,
                                           const sycl::queue &syclQueue) {
  return map_external_image_memory(extMem, desc, syclQueue.get_device(),
                                   syclQueue.get_context());
}

__SYCL_EXPORT
void *map_external_linear_memory(external_mem extMem, uint64_t offset,
                                 uint64_t size, const sycl::device &syclDevice,
                                 const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_handle_t urExternalMem{extMem.raw_handle};

  void *retMemory = nullptr;
  Adapter->call<
      sycl::errc::invalid,
      sycl::detail::UrApiKind::urBindlessImagesMapExternalLinearMemoryExp>(
      urCtx, urDevice, offset, size, urExternalMem, &retMemory);

  return retMemory;
}

__SYCL_EXPORT
void *map_external_linear_memory(external_mem extMem, uint64_t offset,
                                 uint64_t size, const sycl::queue &syclQueue) {
  return map_external_linear_memory(
      extMem, offset, size, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT void release_external_memory(external_mem extMem,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter
      ->call<sycl::errc::invalid,
             sycl::detail::UrApiKind::urBindlessImagesReleaseExternalMemoryExp>(
          urCtx, urDevice, extMem.raw_handle);
}

__SYCL_EXPORT void release_external_memory(external_mem extMem,
                                           const sycl::queue &syclQueue) {
  release_external_memory(extMem, syclQueue.get_device(),
                          syclQueue.get_context());
}

__SYCL_EXPORT void
unmap_external_linear_memory(void *mappedLinearRegion,
                             const sycl::device &syclDevice,
                             const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<
      sycl::errc::invalid,
      sycl::detail::UrApiKind::urBindlessImagesFreeMappedLinearMemoryExp>(
      urCtx, urDevice, mappedLinearRegion);
}

__SYCL_EXPORT void unmap_external_image_memory(
    image_mem_handle mappedImageMem, image_type imageType,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  free_image_mem(mappedImageMem, imageType, syclDevice, syclContext);
}

__SYCL_EXPORT bool
supports_importing_handle_type(external_mem_handle_type externMemHandleType,
                               const sycl::device &syclDevice) {
  auto [urDevice, Adapter] = get_ur_handles(syclDevice);
  const auto urHandleType = detail::to_ur_type(externMemHandleType);
  ur_bool_t supportsExternalHandleType{0};
  Adapter->call<
      sycl::errc::invalid,
      sycl::detail::UrApiKind::urBindlessImagesSupportsImportingHandleTypeExp>(
      urDevice, urHandleType, &supportsExternalHandleType);
  return static_cast<bool>(supportsExternalHandleType);
}

template <>
__SYCL_EXPORT external_semaphore import_external_semaphore(
    external_semaphore_descriptor<resource_fd> externalSemaphoreDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_semaphore_handle_t urExternalSemaphore = nullptr;
  ur_exp_file_descriptor_t urFileDescriptor = {};
  urFileDescriptor.stype = UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR;
  urFileDescriptor.fd = externalSemaphoreDesc.external_resource.file_descriptor;
  ur_exp_external_semaphore_desc_t urExternalSemDesc = {};
  urExternalSemDesc.stype = UR_STRUCTURE_TYPE_EXP_EXTERNAL_SEMAPHORE_DESC;
  urExternalSemDesc.pNext = &urFileDescriptor;

  // This specialization handles timeline and binary semaphores which both
  // have the same underlying external semaphore handle type of opaque file
  // descriptor.

  // Select appropriate semaphore handle type.
  ur_exp_external_semaphore_type_t urHandleType;
  switch (externalSemaphoreDesc.handle_type) {
  case external_semaphore_handle_type::opaque_fd:
    urHandleType = UR_EXP_EXTERNAL_SEMAPHORE_TYPE_OPAQUE_FD;
    break;
  case external_semaphore_handle_type::timeline_fd:
    urHandleType = UR_EXP_EXTERNAL_SEMAPHORE_TYPE_TIMELINE_FD;
    break;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid semaphore handle type");
  }

  Adapter->call<
      sycl::errc::invalid,
      sycl::detail::UrApiKind::urBindlessImagesImportExternalSemaphoreExp>(
      urCtx, urDevice, urHandleType, &urExternalSemDesc, &urExternalSemaphore);

  return external_semaphore{urExternalSemaphore,
                            externalSemaphoreDesc.handle_type};
}

template <>
__SYCL_EXPORT external_semaphore import_external_semaphore(
    external_semaphore_descriptor<resource_fd> externalSemaphoreDesc,
    const sycl::queue &syclQueue) {
  return import_external_semaphore(
      externalSemaphoreDesc, syclQueue.get_device(), syclQueue.get_context());
}

template <>
__SYCL_EXPORT external_semaphore import_external_semaphore(
    external_semaphore_descriptor<resource_win32_handle> externalSemaphoreDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_semaphore_handle_t urExternalSemaphore = nullptr;
  ur_exp_win32_handle_t urWin32Handle = {};
  urWin32Handle.stype = UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE;
  urWin32Handle.handle = externalSemaphoreDesc.external_resource.handle;
  ur_exp_external_semaphore_desc_t urExternalSemDesc = {};
  urExternalSemDesc.stype = UR_STRUCTURE_TYPE_EXP_EXTERNAL_SEMAPHORE_DESC;
  urExternalSemDesc.pNext = &urWin32Handle;

  // Select appropriate semaphore handle type.
  ur_exp_external_semaphore_type_t urHandleType;
  switch (externalSemaphoreDesc.handle_type) {
  case external_semaphore_handle_type::win32_nt_handle:
    urHandleType = UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT;
    break;
  case external_semaphore_handle_type::win32_nt_dx12_fence:
    urHandleType = UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE;
    break;
  case external_semaphore_handle_type::timeline_win32_nt_handle:
    urHandleType = UR_EXP_EXTERNAL_SEMAPHORE_TYPE_TIMELINE_WIN32_NT;
    break;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid semaphore handle type");
  }

  Adapter->call<
      sycl::errc::invalid,
      sycl::detail::UrApiKind::urBindlessImagesImportExternalSemaphoreExp>(
      urCtx, urDevice, urHandleType, &urExternalSemDesc, &urExternalSemaphore);

  return external_semaphore{urExternalSemaphore,
                            externalSemaphoreDesc.handle_type};
}

template <>
__SYCL_EXPORT external_semaphore import_external_semaphore(
    external_semaphore_descriptor<resource_win32_handle> externalSemaphoreDesc,
    const sycl::queue &syclQueue) {
  return import_external_semaphore(
      externalSemaphoreDesc, syclQueue.get_device(), syclQueue.get_context());
}

__SYCL_EXPORT void
release_external_semaphore(external_semaphore externalSemaphore,
                           const sycl::device &syclDevice,
                           const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<
      sycl::errc::invalid,
      sycl::detail::UrApiKind::urBindlessImagesReleaseExternalSemaphoreExp>(
      urCtx, urDevice, externalSemaphore.raw_handle);
}

__SYCL_EXPORT void
release_external_semaphore(external_semaphore externalSemaphore,
                           const sycl::queue &syclQueue) {
  release_external_semaphore(externalSemaphore, syclQueue.get_device(),
                             syclQueue.get_context());
}

__SYCL_EXPORT sycl::range<3> get_image_range(const image_mem_handle memHandle,
                                             const sycl::device &syclDevice,
                                             const sycl::context &syclContext) {
  (void)syclDevice;
  auto [urCtx, Adapter] = get_ur_handles(syclContext);

  size_t Width = 0, Height = 0, Depth = 0;

  Adapter->call<sycl::errc::invalid,
                sycl::detail::UrApiKind::urBindlessImagesImageGetInfoExp>(
      urCtx, memHandle.raw_handle, UR_IMAGE_INFO_WIDTH, &Width, nullptr);

  Adapter->call<sycl::errc::invalid,
                sycl::detail::UrApiKind::urBindlessImagesImageGetInfoExp>(
      urCtx, memHandle.raw_handle, UR_IMAGE_INFO_HEIGHT, &Height, nullptr);

  Adapter->call<sycl::errc::invalid,
                sycl::detail::UrApiKind::urBindlessImagesImageGetInfoExp>(
      urCtx, memHandle.raw_handle, UR_IMAGE_INFO_DEPTH, &Depth, nullptr);

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
  (void)syclDevice;
  auto [urCtx, Adapter] = get_ur_handles(syclContext);

  ur_image_format_t URFormat = {};

  Adapter->call<sycl::errc::invalid,
                sycl::detail::UrApiKind::urBindlessImagesImageGetInfoExp>(
      urCtx, memHandle.raw_handle, UR_IMAGE_INFO_FORMAT, &URFormat, nullptr);

  image_channel_type ChannelType =
      sycl::detail::convertChannelType(URFormat.channelType);

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

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<sycl::errc::memory_allocation,
                sycl::detail::UrApiKind::urUSMPitchedAllocExp>(
      urCtx, urDevice, nullptr, nullptr, widthInBytes, height, elementSizeBytes,
      &RetVal, resultPitch);

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
  unsigned int elementSizeBytes =
      sycl::detail::getImageElementSize(desc.num_channels, desc.channel_type);

  size_t widthInBytes = desc.width * elementSizeBytes;
  size_t height = desc.height;

  return pitched_alloc_device(resultPitch, widthInBytes, height,
                              elementSizeBytes, syclDevice, syclContext);
}

__SYCL_EXPORT unsigned int
get_image_num_channels(const image_mem_handle memHandle,
                       const sycl::device &syclDevice,
                       const sycl::context &syclContext) {
  (void)syclDevice;
  auto [urCtx, Adapter] = get_ur_handles(syclContext);
  ur_image_format_t URFormat = {};

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urBindlessImagesImageGetInfoExp>(
      urCtx, memHandle.raw_handle, UR_IMAGE_INFO_FORMAT, &URFormat, nullptr);

  image_channel_order Order =
      sycl::detail::convertChannelOrder(URFormat.channelOrder);

  return static_cast<unsigned int>(sycl::detail::getImageNumberChannels(Order));
}

__SYCL_EXPORT unsigned int
get_image_num_channels(const image_mem_handle memHandle,
                       const sycl::queue &syclQueue) {
  return get_image_num_channels(memHandle, syclQueue.get_device(),
                                syclQueue.get_context());
}

__SYCL_EXPORT std::vector<image_memory_handle_type>
get_image_memory_support(const image_descriptor &imageDescriptor,
                         const sycl::device &syclDevice,
                         const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(imageDescriptor, urDesc, urFormat);

  ur_bool_t supportsPointerAllocation{0};
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::
                    urBindlessImagesGetImageMemoryHandleTypeSupportExp>(
      urCtx, urDevice, &urDesc, &urFormat,
      ur_exp_image_mem_type_t::UR_EXP_IMAGE_MEM_TYPE_USM_POINTER,
      &supportsPointerAllocation);

  ur_bool_t supportsOpaqueAllocation{0};
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::
                    urBindlessImagesGetImageMemoryHandleTypeSupportExp>(
      urCtx, urDevice, &urDesc, &urFormat,
      ur_exp_image_mem_type_t::UR_EXP_IMAGE_MEM_TYPE_OPAQUE_HANDLE,
      &supportsOpaqueAllocation);

  std::vector<image_memory_handle_type> supportedMemHandleTypes;

  if (supportsPointerAllocation) {
    supportedMemHandleTypes.push_back(image_memory_handle_type::usm_pointer);
  }

  if (supportsOpaqueAllocation) {
    supportedMemHandleTypes.push_back(image_memory_handle_type::opaque_handle);
  }

  return supportedMemHandleTypes;
}

__SYCL_EXPORT std::vector<image_memory_handle_type>
get_image_memory_support(const image_descriptor &imageDescriptor,
                         const sycl::queue &syclQueue) {
  return get_image_memory_support(imageDescriptor, syclQueue.get_device(),
                                  syclQueue.get_context());
}

template <>
__SYCL_EXPORT bool is_image_handle_supported<unsampled_image_handle>(
    const image_descriptor &imageDescriptor,
    image_memory_handle_type imageMemoryHandleType,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(imageDescriptor, urDesc, urFormat);

  const ur_exp_image_mem_type_t memHandleType =
      (imageMemoryHandleType == image_memory_handle_type::opaque_handle)
          ? ur_exp_image_mem_type_t::UR_EXP_IMAGE_MEM_TYPE_OPAQUE_HANDLE
          : ur_exp_image_mem_type_t::UR_EXP_IMAGE_MEM_TYPE_USM_POINTER;

  ur_bool_t supportsUnsampledHandle{0};
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::
                    urBindlessImagesGetImageUnsampledHandleSupportExp>(
      urCtx, urDevice, &urDesc, &urFormat, memHandleType,
      &supportsUnsampledHandle);

  return supportsUnsampledHandle;
}

template <>
__SYCL_EXPORT bool is_image_handle_supported<unsampled_image_handle>(
    const image_descriptor &imageDescriptor,
    image_memory_handle_type imageMemoryHandleType,
    const sycl::queue &syclQueue) {
  return is_image_handle_supported<unsampled_image_handle>(
      imageDescriptor, imageMemoryHandleType, syclQueue.get_device(),
      syclQueue.get_context());
}

template <>
__SYCL_EXPORT bool is_image_handle_supported<sampled_image_handle>(
    const image_descriptor &imageDescriptor,
    image_memory_handle_type imageMemoryHandleType,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(imageDescriptor, urDesc, urFormat);

  const ur_exp_image_mem_type_t memHandleType =
      (imageMemoryHandleType == image_memory_handle_type::opaque_handle)
          ? ur_exp_image_mem_type_t::UR_EXP_IMAGE_MEM_TYPE_OPAQUE_HANDLE
          : ur_exp_image_mem_type_t::UR_EXP_IMAGE_MEM_TYPE_USM_POINTER;

  ur_bool_t supportsSampledHandle{0};
  Adapter->call<
      sycl::errc::runtime,
      sycl::detail::UrApiKind::urBindlessImagesGetImageSampledHandleSupportExp>(
      urCtx, urDevice, &urDesc, &urFormat, memHandleType,
      &supportsSampledHandle);

  return supportsSampledHandle;
}

template <>
__SYCL_EXPORT bool is_image_handle_supported<sampled_image_handle>(
    const image_descriptor &imageDescriptor,
    image_memory_handle_type imageMemoryHandleType,
    const sycl::queue &syclQueue) {
  return is_image_handle_supported<sampled_image_handle>(
      imageDescriptor, imageMemoryHandleType, syclQueue.get_device(),
      syclQueue.get_context());
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
