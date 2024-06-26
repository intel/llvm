// REQUIRES: cuda
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"
#include <sycl/properties/queue_properties.hpp>

#include <random>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Helpers and utilities
namespace util {
struct handles_t {
  syclexp::interop_mem_handle input_interop_mem_handle_1,
      input_interop_mem_handle_2, output_interop_mem_handle;
  syclexp::image_mem_handle input_mem_handle_1, input_mem_handle_2,
      output_mem_handle;
  syclexp::interop_semaphore_handle sycl_wait_interop_semaphore_handle,
      sycl_done_interop_semaphore_handle;
  syclexp::unsampled_image_handle input_1, input_2, output;
};

template <typename InteropMemHandleT, typename InteropSemHandleT>
handles_t
create_test_handles(sycl::context &ctxt, sycl::device &dev,
                    InteropMemHandleT img_in_interop_handle_1,
                    InteropMemHandleT img_in_interop_handle_2,
                    InteropMemHandleT img_out_interop_handle,
                    InteropSemHandleT sycl_wait_semaphore_handle,
                    InteropSemHandleT sycl_done_semaphore_handle,
                    const size_t img_size,
                    sycl::ext::oneapi::experimental::image_descriptor &desc) {
  // Extension: map the external memory descriptors

#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      input_ext_mem_desc_1{img_in_interop_handle_1,
                           syclexp::external_mem_handle_type::win32_nt_handle,
                           img_size};
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      input_ext_mem_desc_2{img_in_interop_handle_2,
                           syclexp::external_mem_handle_type::win32_nt_handle,
                           img_size};
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      output_ext_mem_desc{img_out_interop_handle,
                          syclexp::external_mem_handle_type::win32_nt_handle,
                          img_size};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> input_ext_mem_desc_1{
      img_in_interop_handle_1, syclexp::external_mem_handle_type::opaque_fd,
      img_size};
  syclexp::external_mem_descriptor<syclexp::resource_fd> input_ext_mem_desc_2{
      img_in_interop_handle_2, syclexp::external_mem_handle_type::opaque_fd,
      img_size};
  syclexp::external_mem_descriptor<syclexp::resource_fd> output_ext_mem_desc{
      img_out_interop_handle, syclexp::external_mem_handle_type::opaque_fd,
      img_size};
#endif

  // Extension: create interop memory handles
  syclexp::interop_mem_handle input_interop_mem_handle_1 =
      syclexp::import_external_memory(input_ext_mem_desc_1, dev, ctxt);
  syclexp::interop_mem_handle input_interop_mem_handle_2 =
      syclexp::import_external_memory(input_ext_mem_desc_2, dev, ctxt);
  syclexp::interop_mem_handle output_interop_mem_handle =
      syclexp::import_external_memory(output_ext_mem_desc, dev, ctxt);

  // Extension: map image memory handles
  syclexp::image_mem_handle input_mapped_mem_handle_1 =
      syclexp::map_external_image_memory(input_interop_mem_handle_1, desc, dev,
                                         ctxt);
  syclexp::image_mem_handle input_mapped_mem_handle_2 =
      syclexp::map_external_image_memory(input_interop_mem_handle_2, desc, dev,
                                         ctxt);
  syclexp::image_mem_handle output_mapped_mem_handle =
      syclexp::map_external_image_memory(output_interop_mem_handle, desc, dev,
                                         ctxt);

  // Extension: create the image and return the handle
  syclexp::unsampled_image_handle input_1 =
      syclexp::create_image(input_mapped_mem_handle_1, desc, dev, ctxt);
  syclexp::unsampled_image_handle input_2 =
      syclexp::create_image(input_mapped_mem_handle_2, desc, dev, ctxt);
  syclexp::unsampled_image_handle output =
      syclexp::create_image(output_mapped_mem_handle, desc, dev, ctxt);

  // Extension: import semaphores
#ifdef _WIN32
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      sycl_wait_external_semaphore_desc{
          sycl_wait_semaphore_handle,
          syclexp::external_semaphore_handle_type::win32_nt_handle};
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      sycl_done_external_semaphore_desc{
          sycl_done_semaphore_handle,
          syclexp::external_semaphore_handle_type::win32_nt_handle};
#else
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      sycl_wait_external_semaphore_desc{
          sycl_wait_semaphore_handle,
          syclexp::external_semaphore_handle_type::opaque_fd};
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      sycl_done_external_semaphore_desc{
          sycl_done_semaphore_handle,
          syclexp::external_semaphore_handle_type::opaque_fd};
#endif

  syclexp::interop_semaphore_handle sycl_wait_interop_semaphore_handle =
      syclexp::import_external_semaphore(sycl_wait_external_semaphore_desc, dev,
                                         ctxt);
  syclexp::interop_semaphore_handle sycl_done_interop_semaphore_handle =
      syclexp::import_external_semaphore(sycl_done_external_semaphore_desc, dev,
                                         ctxt);

  return {input_interop_mem_handle_1,
          input_interop_mem_handle_2,
          output_interop_mem_handle,
          input_mapped_mem_handle_1,
          input_mapped_mem_handle_2,
          output_mapped_mem_handle,
          sycl_wait_interop_semaphore_handle,
          sycl_done_interop_semaphore_handle,
          input_1,
          input_2,
          output};
}

void cleanup_test(sycl::context &ctxt, sycl::device &dev, handles_t handles) {
  syclexp::destroy_external_semaphore(
      handles.sycl_wait_interop_semaphore_handle, dev, ctxt);
  syclexp::destroy_external_semaphore(
      handles.sycl_done_interop_semaphore_handle, dev, ctxt);
  syclexp::destroy_image_handle(handles.input_1, dev, ctxt);
  syclexp::destroy_image_handle(handles.input_2, dev, ctxt);
  syclexp::destroy_image_handle(handles.output, dev, ctxt);
  syclexp::free_image_mem(handles.input_mem_handle_1,
                          syclexp::image_type::standard, dev, ctxt);
  syclexp::free_image_mem(handles.input_mem_handle_1,
                          syclexp::image_type::standard, dev, ctxt);
  syclexp::free_image_mem(handles.output_mem_handle,
                          syclexp::image_type::standard, dev, ctxt);
  syclexp::release_external_memory(handles.input_interop_mem_handle_1, dev,
                                   ctxt);
  syclexp::release_external_memory(handles.input_interop_mem_handle_2, dev,
                                   ctxt);
  syclexp::release_external_memory(handles.output_interop_mem_handle, dev,
                                   ctxt);
}

template <typename InteropMemHandleT, typename InteropSemHandleT, int NDims,
          typename DType, sycl::image_channel_type CType, int NChannels,
          typename KernelName>
void run_ndim_test(sycl::range<NDims> global_size,
                   sycl::range<NDims> local_size,
                   InteropMemHandleT img_in_interop_handle_1,
                   InteropMemHandleT img_in_interop_handle_2,
                   InteropMemHandleT img_out_interop_handle,
                   InteropSemHandleT sycl_wait_semaphore_handle,
                   InteropSemHandleT sycl_done_semaphore_handle) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::device dev;
  sycl::queue q{dev, {sycl::property::queue::in_order{}}};
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  syclexp::image_descriptor desc(global_size, NChannels, CType);

  const size_t img_size = global_size.size() * sizeof(DType) * NChannels;

  auto handles = create_test_handles(
      ctxt, dev, img_in_interop_handle_1, img_in_interop_handle_2,
      img_out_interop_handle, sycl_wait_semaphore_handle,
      sycl_done_semaphore_handle, img_size, desc);

  // Extension: wait for imported semaphore
  q.ext_oneapi_wait_external_semaphore(
      handles.sycl_wait_interop_semaphore_handle);

  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(
          sycl::nd_range<NDims>{global_size, local_size},
          [=](sycl::nd_item<NDims> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);

            if constexpr (NDims == 2) {
              if constexpr (NChannels > 1) {
                VecType px1 = syclexp::fetch_image<VecType>(
                    handles.input_1, sycl::int2(dim0, dim1));
                VecType px2 = syclexp::fetch_image<VecType>(
                    handles.input_2, sycl::int2(dim0, dim1));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<VecType>(
                    handles.output, sycl::int2(dim0, dim1), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_image<DType>(handles.input_1,
                                                        sycl::int2(dim0, dim1));
                DType px2 = syclexp::fetch_image<DType>(handles.input_2,
                                                        sycl::int2(dim0, dim1));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<DType>(handles.output,
                                            sycl::int2(dim0, dim1), DType(sum));
              }
            } else {
              size_t dim2 = it.get_global_id(2);

              if constexpr (NChannels > 1) {
                VecType px1 = syclexp::fetch_image<VecType>(
                    handles.input_1, sycl::int3(dim0, dim1, dim2));
                VecType px2 = syclexp::fetch_image<VecType>(
                    handles.input_2, sycl::int3(dim0, dim1, dim2));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<VecType>(
                    handles.output, sycl::int3(dim0, dim1, dim2), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_image<DType>(
                    handles.input_1, sycl::int3(dim0, dim1, dim2));
                DType px2 = syclexp::fetch_image<DType>(
                    handles.input_2, sycl::int3(dim0, dim1, dim2));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image<DType>(
                    handles.output, sycl::int3(dim0, dim1, dim2), DType(sum));
              }
            }
          });
    });

    // Extension: signal imported semaphore
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_signal_external_semaphore(
          handles.sycl_done_interop_semaphore_handle);
    });

    // Wait for kernel completion before destroying external objects
    q.wait_and_throw();

    // Cleanup
    cleanup_test(ctxt, dev, handles);
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }
}
} // namespace util

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> local_size,
              unsigned int seed = 0) {
  uint32_t width = static_cast<uint32_t>(dims[0]);
  uint32_t height = 1;
  uint32_t depth = 1;

  size_t num_elems = dims[0];
  VkImageType imgType = VK_IMAGE_TYPE_1D;

  if (NDims > 1) {
    num_elems *= dims[1];
    height = static_cast<uint32_t>(dims[1]);
    imgType = VK_IMAGE_TYPE_2D;
  }
  if (NDims > 2) {
    num_elems *= dims[2];
    depth = static_cast<uint32_t>(dims[2]);
    imgType = VK_IMAGE_TYPE_3D;
  }

  VkFormat format = vkutil::to_vulkan_format(COrder, CType);
  const size_t imageSizeBytes = num_elems * NChannels * sizeof(DType);

  vkutil::vulkan_image_test_resources_t inVkImgRes1(
      imgType, format, {width, height, depth}, imageSizeBytes);
  vkutil::vulkan_image_test_resources_t inVkImgRes2(
      imgType, format, {width, height, depth}, imageSizeBytes);
  vkutil::vulkan_image_test_resources_t outVkImgRes(
      imgType, format, {width, height, depth}, imageSizeBytes);

  printString("Populating staging buffer\n");
  // Populate staging memory
  using VecType = sycl::vec<DType, NChannels>;
  auto init =
      bindless_helpers::init_vector<DType, NChannels>(static_cast<DType>(0));

  std::vector<VecType> input_vector_0(num_elems, init);
  std::srand(seed);
  bindless_helpers::fill_rand(input_vector_0);

  VecType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inVkImgRes1.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < num_elems; ++i) {
    inputStagingData[i] = input_vector_0[i];
  }
  vkUnmapMemory(vk_device, inVkImgRes1.stagingMemory);

  std::vector<VecType> input_vector_1(num_elems, init);
  std::srand(seed);
  bindless_helpers::fill_rand(input_vector_1);

  VK_CHECK_CALL(vkMapMemory(vk_device, inVkImgRes2.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < num_elems; ++i) {
    inputStagingData[i] = input_vector_1[i];
  }
  vkUnmapMemory(vk_device, inVkImgRes2.stagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput1 =
        vkutil::createImageMemoryBarrier(inVkImgRes1.vkImage, 1 /*mipLevels*/);
    VkImageMemoryBarrier barrierInput2 =
        vkutil::createImageMemoryBarrier(inVkImgRes2.vkImage, 1 /*mipLevels*/);

    VkImageMemoryBarrier barrierOutput =
        vkutil::createImageMemoryBarrier(outVkImgRes.vkImage, 1 /*mipLevels*/);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput1);

    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput2);

    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierOutput);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  // Create semaphore to later import in SYCL
  printString("Creating semaphores\n");
  VkSemaphore syclWaitSemaphore;
  {
    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#ifdef _WIN32
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    VK_CHECK_CALL(
        vkCreateSemaphore(vk_device, &sci, nullptr, &syclWaitSemaphore));
  }

  VkSemaphore syclDoneSemaphore;
  {
    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#ifdef _WIN32
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    VK_CHECK_CALL(
        vkCreateSemaphore(vk_device, &sci, nullptr, &syclDoneSemaphore));
  }

  printString("Copying staging memory to images\n");
  // Copy staging to main image memory
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, depth};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inVkImgRes1.stagingBuffer,
                           inVkImgRes1.vkImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inVkImgRes2.stagingBuffer,
                           inVkImgRes2.vkImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];

    submission.signalSemaphoreCount = 1;
    submission.pSignalSemaphores = &syclWaitSemaphore;
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
  }

  printString("Getting memory interop handles\n");

  // Pass memory to SYCL for modification
  auto global_size = dims;
#ifdef _WIN32
  auto input_mem_handle_1 =
      vkutil::getMemoryWin32Handle(inVkImgRes1.imageMemory);
  auto input_mem_handle_2 =
      vkutil::getMemoryWin32Handle(inVkImgRes2.imageMemory);
  auto output_mem_handle =
      vkutil::getMemoryWin32Handle(outVkImgRes.imageMemory);
#else
  auto input_mem_handle_1 = vkutil::getMemoryOpaqueFD(inVkImgRes1.imageMemory);
  auto input_mem_handle_2 = vkutil::getMemoryOpaqueFD(inVkImgRes2.imageMemory);
  auto output_mem_handle = vkutil::getMemoryOpaqueFD(outVkImgRes.imageMemory);
#endif

  printString("Getting semaphore interop handles\n");

  // Pass semaphores to SYCL for synchronization
#ifdef _WIN32
  auto sycl_wait_semaphore_handle =
      vkutil::getSemaphoreWin32Handle(syclWaitSemaphore);
  auto sycl_done_semaphore_handle =
      vkutil::getSemaphoreWin32Handle(syclDoneSemaphore);
#else
  auto sycl_wait_semaphore_handle =
      vkutil::getSemaphoreOpaqueFD(syclWaitSemaphore);
  auto sycl_done_semaphore_handle =
      vkutil::getSemaphoreOpaqueFD(syclDoneSemaphore);
#endif

  printString("Calling into SYCL with interop memory and semaphore handles\n");

  util::run_ndim_test<decltype(input_mem_handle_1),
                      decltype(sycl_wait_semaphore_handle), NDims, DType, CType,
                      NChannels, KernelName>(
      global_size, local_size, input_mem_handle_1, input_mem_handle_2,
      output_mem_handle, sycl_wait_semaphore_handle,
      sycl_done_semaphore_handle);

  printString("Copying image memory to staging memory\n");
  // Copy main image memory to staging
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, depth};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[1], &cbbi));
    vkCmdCopyImageToBuffer(vk_transferCmdBuffers[1], outVkImgRes.vkImage,
                           VK_IMAGE_LAYOUT_GENERAL, outVkImgRes.stagingBuffer,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[1]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[1];

    submission.waitSemaphoreCount = 1;
    submission.pWaitSemaphores = &syclDoneSemaphore;
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  printString("Validating\n");
  // Validate that SYCL made changes to the memory
  bool validated = true;
  VecType *outputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, outVkImgRes.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&outputStagingData));
  for (int i = 0; i < num_elems; ++i) {
    VecType expected = input_vector_0[i] + input_vector_1[i];
    for (int j = 0; j < NChannels; ++j) {
      // Use helper function to determine if data is accepted
      // For integers, exact results are expected
      // For floats, accepted error variance is passed
      if (!util::is_equal(outputStagingData[i][j], expected[j])) {
        std::cerr << "Result mismatch! actual[" << i << "][" << j
                  << "] == " << outputStagingData[i][j]
                  << " : expected == " << expected[j] << "\n";
        validated = false;
      }
    }
    if (!validated)
      break;
  }
  vkUnmapMemory(vk_device, outVkImgRes.stagingMemory);

  if (validated) {
    printString("  Results are correct!\n");
  }

  // Cleanup
  vkDestroySemaphore(vk_device, syclWaitSemaphore, nullptr);
  vkDestroySemaphore(vk_device, syclDoneSemaphore, nullptr);

  return validated;
}

bool run_all() {
  unsigned int seed = 0;

  printString("Running 3D uint4\n");
  bool valid = run_test<3, uint32_t, 4, sycl::image_channel_type::signed_int32,
                        sycl::image_channel_order::rgba, class uint4_3d>(
      {272, 144, 4}, {16, 16, 4}, seed);

  printString("Running 3D uint2\n");
  valid &= run_test<3, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rg, class uint2_3d>(
      {272, 144, 4}, {16, 16, 4}, seed);

  printString("Running 3D uint\n");
  valid &= run_test<3, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::r, class uint1_3d>(
      {272, 144, 4}, {16, 16, 4}, seed);

  printString("Running 3D float4\n");
  valid &= run_test<3, float, 4, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rgba, class float4_3d>(
      {16, 16, 16}, {16, 16, 4}, seed);

  printString("Running 3D float2\n");
  valid &= run_test<3, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float2_3d>(
      {128, 128, 16}, {16, 16, 4}, seed);
  printString("Running 3D float\n");
  valid &= run_test<3, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class float1_3d>(
      {1024, 1024, 16}, {16, 16, 4}, seed);

  printString("Running 2D uint4\n");
  valid &= run_test<2, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rgba, class uint4_2d>(
      {1024, 1024}, {2, 2}, seed);

  printString("Running 2D uint2\n");
  valid &= run_test<2, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rg, class uint2_2d>(
      {128, 128}, {2, 2}, seed);

  printString("Running 2D uint\n");
  valid &= run_test<2, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::r, class uint1_2d>({512, 512},
                                                                  {2, 2}, seed);

  printString("Running 2D float4\n");
  valid &= run_test<2, float, 4, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rgba, class float4_2d>(
      {128, 64}, {2, 2}, seed);

  printString("Running 2D float2\n");
  valid &= run_test<2, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float2_2d>(
      {1024, 512}, {2, 2}, seed);

  printString("Running 2D float\n");
  valid &= run_test<2, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class float1_2d>(
      {32, 32}, {2, 2}, seed);

  return valid;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  const char *devices[] = {"Intel", "NVIDIA"};
  if (std::none_of(std::begin(devices), std::end(devices),
                   [](const char *device) {
                     return vkutil::setupDevice(device) == VK_SUCCESS;
                   })) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  auto run_ok = run_all();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  if (run_ok) {
    std::cout << "All tests passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
