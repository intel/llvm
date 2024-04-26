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

void populate_ur_structs(const image_descriptor &desc, ur_image_desc_t &urDesc,
                         ur_image_format_t &urFormat, size_t pitch = 0) {
  urDesc = {};
  urDesc.width = desc.width;
  urDesc.height = desc.height;
  urDesc.depth = desc.depth;

  if (desc.array_size > 1) {
    // Image array or cubemap
    urDesc.type = desc.type == image_type::cubemap
                      ? UR_MEM_TYPE_IMAGE_CUBEMAP_EXP
                  : desc.height > 0 ? UR_MEM_TYPE_IMAGE2D_ARRAY
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
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();
  ur_exp_image_handle_t urImageHandle = imageHandle.raw_handle;

  Plugin->call<sycl::errc::runtime>(
      urBindlessImagesUnsampledImageHandleDestroyExp, C, Device, urImageHandle);
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
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();
  ur_exp_image_handle_t piImageHandle = imageHandle.raw_handle;

  Plugin->call<sycl::errc::runtime>(
      urBindlessImagesSampledImageHandleDestroyExp, C, Device, piImageHandle);
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

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  image_mem_handle retHandle;

  // Call impl.
  Plugin->call<sycl::errc::memory_allocation>(
      urBindlessImagesImageAllocateExp, C, Device, &urFormat, &urDesc,
      reinterpret_cast<ur_exp_image_mem_handle_t *>(&retHandle.raw_handle));

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
  desc.verify();

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  // Call impl.
  image_mem_handle retHandle;
  Plugin->call<sycl::errc::memory_allocation>(
      urBindlessImagesImageAllocateExp, C, Device, &urFormat, &urDesc,
      reinterpret_cast<ur_exp_image_mem_handle_t *>(&retHandle.raw_handle));

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
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  // Call impl.
  image_mem_handle individual_image;
  Plugin->call<sycl::errc::runtime>(
      urBindlessImagesMipmapGetLevelExp, C, Device,
      reinterpret_cast<ur_exp_image_mem_handle_t>(mipMem.raw_handle), level,
      reinterpret_cast<ur_exp_image_mem_handle_t *>(
          &individual_image.raw_handle));

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
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  if (memHandle.raw_handle != nullptr) {
    if (imageType == image_type::mipmap) {
      Plugin->call<sycl::errc::memory_allocation>(
          urBindlessImagesMipmapFreeExp, C, Device,
          reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle));
    } else if (imageType == image_type::standard ||
               imageType == image_type::array ||
               imageType == image_type::cubemap) {
      Plugin->call<sycl::errc::memory_allocation>(
          urBindlessImagesImageFreeExp, C, Device,
          reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle));
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
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  Plugin->call<sycl::errc::memory_allocation>(
      urBindlessImagesMipmapFreeExp, C, Device,
      reinterpret_cast<ur_exp_image_mem_handle_t>(memoryHandle.raw_handle));
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
  desc.verify();

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  // Call impl.
  ur_exp_image_handle_t urImageHandle = nullptr;
  Plugin->call<sycl::errc::runtime>(
      urBindlessImagesUnsampledImageCreateExp, C, Device,
      reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle),
      &urFormat, &urDesc, nullptr, &urImageHandle);

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

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

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

  ur_sampler_handle_t urSampler = nullptr;
  Plugin->call<sycl::errc::runtime>(urSamplerCreate, C, &UrSamplerProps,
                                    &urSampler);

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat, pitch);

  // Call impl.
  ur_exp_image_handle_t urImageHandle = nullptr;
  Plugin->call<sycl::errc::runtime>(
      urBindlessImagesSampledImageCreateExp, C, Device,
      static_cast<ur_exp_image_mem_handle_t>(devPtr), &urFormat, &urDesc,
      urSampler, nullptr, &urImageHandle);

  return sampled_image_handle{urImageHandle};
}

__SYCL_EXPORT sampled_image_handle
create_image(void *devPtr, size_t pitch, const bindless_image_sampler &sampler,
             const image_descriptor &desc, const sycl::queue &syclQueue) {
  return create_image(devPtr, pitch, sampler, desc, syclQueue.get_device(),
                      syclQueue.get_context());
}

template <>
__SYCL_EXPORT interop_mem_handle import_external_memory<resource_fd>(
    external_mem_descriptor<resource_fd> externalMem,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  ur_exp_file_descriptor_t PosixFD{};
  PosixFD.stype = UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR;
  PosixFD.fd = externalMem.external_resource.file_descriptor;

  ur_exp_interop_mem_desc_t InteropMemDesc{};
  InteropMemDesc.stype = UR_STRUCTURE_TYPE_EXP_INTEROP_MEM_DESC;
  InteropMemDesc.pNext = &PosixFD;

  ur_exp_interop_mem_handle_t urInteropMem = nullptr;
  Plugin->call<sycl::errc::invalid>(urBindlessImagesImportOpaqueFDExp, C,
                                    Device, externalMem.size_in_bytes,
                                    &InteropMemDesc, &urInteropMem);

  return interop_mem_handle{urInteropMem};
}

template <>
__SYCL_EXPORT interop_mem_handle import_external_memory<resource_fd>(
    external_mem_descriptor<resource_fd> externalMem,
    const sycl::queue &syclQueue) {
  return import_external_memory<resource_fd>(
      externalMem, syclQueue.get_device(), syclQueue.get_context());
}

template <>
__SYCL_EXPORT_DEPRECATED(
    "import_external_memory templated by external_mem_fd is deprecated."
    "Template with resource_fd instead.")
interop_mem_handle import_external_memory<external_mem_fd>(
    external_mem_descriptor<external_mem_fd> externalMem,
    const sycl::device &syclDevice, const sycl::context &syclContext) {

  external_mem_descriptor<resource_fd> extMem;
  extMem.external_resource.file_descriptor =
      externalMem.external_resource.file_descriptor;
  extMem.size_in_bytes = externalMem.size_in_bytes;
  return import_external_memory<resource_fd>(extMem, syclDevice, syclContext);
}

template <>
__SYCL_EXPORT_DEPRECATED(
    "import_external_memory templated by external_mem_fd is deprecated."
    "Template with resource_fd instead.")
interop_mem_handle import_external_memory<external_mem_fd>(
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
  desc.verify();

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  ur_image_desc_t urDesc;
  ur_image_format_t urFormat;
  populate_ur_structs(desc, urDesc, urFormat);

  ur_exp_interop_mem_handle_t urInteropMem{memHandle.raw_handle};

  image_mem_handle retHandle;
  Plugin->call<sycl::errc::invalid>(
      urBindlessImagesMapExternalArrayExp, C, Device, &urFormat, &urDesc,
      urInteropMem,
      reinterpret_cast<ur_exp_image_mem_handle_t *>(&retHandle.raw_handle));

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
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  Plugin->call<sycl::errc::invalid>(urBindlessImagesReleaseInteropExp, C,
                                    Device, interopMem.raw_handle);
}

__SYCL_EXPORT void release_external_memory(interop_mem_handle interopMem,
                                           const sycl::queue &syclQueue) {
  release_external_memory(interopMem, syclQueue.get_device(),
                          syclQueue.get_context());
}

template <>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<resource_fd> externalSemaphoreDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();

  ur_exp_file_descriptor_t FileDescriptor = {
      UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR, nullptr,
      externalSemaphoreDesc.external_resource.file_descriptor};

  ur_exp_interop_semaphore_desc_t InteropSemDesc = {
      UR_STRUCTURE_TYPE_EXP_INTEROP_SEMAPHORE_DESC, &FileDescriptor};

  ur_exp_interop_semaphore_handle_t UrInteropSemaphore = nullptr;
  Plugin->call<sycl::errc::invalid>(
      urBindlessImagesImportExternalSemaphoreOpaqueFDExp, C, Device,
      &InteropSemDesc, &UrInteropSemaphore);

  return interop_semaphore_handle{UrInteropSemaphore};
}

template <>
__SYCL_EXPORT interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<resource_fd> externalSemaphoreDesc,
    const sycl::queue &syclQueue) {
  return import_external_semaphore(
      externalSemaphoreDesc, syclQueue.get_device(), syclQueue.get_context());
}

template <>
__SYCL_EXPORT_DEPRECATED("import_external_semaphore templated by "
                         "external_semaphore_fd is deprecated."
                         "Template with resource_fd instead.")
interop_semaphore_handle import_external_semaphore(
    external_semaphore_descriptor<external_semaphore_fd> externalSemaphoreDesc,
    const sycl::device &syclDevice, const sycl::context &syclContext) {

  external_semaphore_descriptor<resource_fd> extSem;
  extSem.external_resource.file_descriptor =
      externalSemaphoreDesc.external_resource.file_descriptor;
  return import_external_semaphore<resource_fd>(extSem, syclDevice,
                                                syclContext);
}

template <>
__SYCL_EXPORT_DEPRECATED("import_external_semaphore templated by "
                         "external_semaphore_fd is deprecated."
                         "Template with resource_fd instead.")
interop_semaphore_handle import_external_semaphore(
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
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();
  ur_context_handle_t C = CtxImpl->getUrHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getUrHandleRef();

  Plugin->call<sycl::errc::invalid>(urBindlessImagesDestroyExternalSemaphoreExp,
                                    C, Device, semaphoreHandle.raw_handle);
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
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  size_t Width = 0, Height = 0, Depth = 0;

  Plugin->call<sycl::errc::invalid>(
      urBindlessImagesImageGetInfoExp,
      reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle),
      UR_IMAGE_INFO_WIDTH, &Width, nullptr);

  Plugin->call<sycl::errc::invalid>(
      urBindlessImagesImageGetInfoExp,
      reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle),
      UR_IMAGE_INFO_HEIGHT, &Height, nullptr);

  Plugin->call<sycl::errc::invalid>(
      urBindlessImagesImageGetInfoExp,
      reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle),
      UR_IMAGE_INFO_DEPTH, &Depth, nullptr);

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
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();

  ur_image_format_t URFormat;

  Plugin->call<sycl::errc::invalid>(
      urBindlessImagesImageGetInfoExp,
      reinterpret_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle),
      UR_IMAGE_INFO_FORMAT, &URFormat, nullptr);

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

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  if (CtxImpl->is_host()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                          "Cannot allocate pitched memory on host!");
  }

  ur_context_handle_t UrContext = CtxImpl->getUrHandleRef();
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();
  ur_device_handle_t UrDevice =
      sycl::detail::getSyclObjImpl(syclDevice)->getUrHandleRef();

  Plugin->call<sycl::errc::memory_allocation>(
      urUSMPitchedAllocExp, UrContext, UrDevice, nullptr, nullptr, widthInBytes,
      height, elementSizeBytes, &RetVal, resultPitch);

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
  std::ignore = syclDevice;

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  const sycl::detail::UrPluginPtr &Plugin = CtxImpl->getUrPlugin();
  ur_image_format_t URFormat = {};

  Plugin->call<sycl::errc::runtime>(
      urBindlessImagesImageGetInfoExp,
      static_cast<ur_exp_image_mem_handle_t>(memHandle.raw_handle),
      UR_IMAGE_INFO_FORMAT, &URFormat, nullptr);

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

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
