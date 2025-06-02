// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes -DTEST_L0_SUPPORTED_VK_FORMAT %}
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"
#include <sycl/properties/queue_properties.hpp>

#include <random>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

uint64_t waitSemaphoreValue = 0;
uint64_t doneSemaphoreValue = 0;

// Helpers and utilities
namespace util {
struct handles_t {
  syclexp::external_mem inputExternalMem, outputExternalMem;
  syclexp::image_mem_handle inputMemHandle, outputMemHandle;
  syclexp::external_semaphore syclWaitExternalSemaphore,
      syclDoneExternalSemaphore;
  syclexp::unsampled_image_handle input, output;
};

template <typename InteropMemHandleT, typename InteropSemHandleT>
handles_t
create_sycl_handles(sycl::context &ctxt, sycl::device &dev,
                    InteropMemHandleT imgInInteropHandle,
                    InteropMemHandleT imgOutInteropHandle,
                    [[maybe_unused]] InteropSemHandleT syclWaitSemaphoreHandle,
                    [[maybe_unused]] InteropSemHandleT syclDoneSemaphoreHandle,
                    const size_t imgSize,
                    sycl::ext::oneapi::experimental::image_descriptor &desc) {
  // Extension: map the external memory descriptors

#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      inputExtMemDesc{imgInInteropHandle,
                      syclexp::external_mem_handle_type::win32_nt_handle,
                      imgSize};
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      outputExtMemDesc{imgOutInteropHandle,
                       syclexp::external_mem_handle_type::win32_nt_handle,
                       imgSize};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> inputExtMemDesc{
      imgInInteropHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSize};
  syclexp::external_mem_descriptor<syclexp::resource_fd> outputExtMemDesc{
      imgOutInteropHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSize};
#endif

  // Extension: create interop memory handles
  syclexp::external_mem inputExternalMem =
      syclexp::import_external_memory(inputExtMemDesc, dev, ctxt);
  syclexp::external_mem outputExternalMem =
      syclexp::import_external_memory(outputExtMemDesc, dev, ctxt);

  // Extension: map image memory handles
  syclexp::image_mem_handle inputMappedMemHandle =
      syclexp::map_external_image_memory(inputExternalMem, desc, dev, ctxt);
  syclexp::image_mem_handle outputMappedMemHandle =
      syclexp::map_external_image_memory(outputExternalMem, desc, dev, ctxt);

  // Extension: create the image and return the handle
  syclexp::unsampled_image_handle input =
      syclexp::create_image(inputMappedMemHandle, desc, dev, ctxt);
  syclexp::unsampled_image_handle output =
      syclexp::create_image(outputMappedMemHandle, desc, dev, ctxt);

  // Extension: import semaphores
#ifdef _WIN32
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      syclWaitExternalSemaphoreDesc{
          syclWaitSemaphoreHandle,
          syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      syclDoneExternalSemaphoreDesc{
          syclDoneSemaphoreHandle,
          syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
#else
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      syclWaitExternalSemaphoreDesc{
          syclWaitSemaphoreHandle,
          syclexp::external_semaphore_handle_type::timeline_fd};
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      syclDoneExternalSemaphoreDesc{
          syclDoneSemaphoreHandle,
          syclexp::external_semaphore_handle_type::timeline_fd};
#endif

  syclexp::external_semaphore syclWaitExternalSemaphore =
      syclexp::import_external_semaphore(syclWaitExternalSemaphoreDesc, dev,
                                         ctxt);
  syclexp::external_semaphore syclDoneExternalSemaphore =
      syclexp::import_external_semaphore(syclDoneExternalSemaphoreDesc, dev,
                                         ctxt);

  return {inputExternalMem,
          outputExternalMem,
          inputMappedMemHandle,
          outputMappedMemHandle,
          syclWaitExternalSemaphore,
          syclDoneExternalSemaphore,
          input,
          output};
}

void cleanup_sycl(sycl::context &ctxt, sycl::device &dev, handles_t handles) {
  syclexp::release_external_semaphore(handles.syclWaitExternalSemaphore, dev,
                                      ctxt);
  syclexp::release_external_semaphore(handles.syclDoneExternalSemaphore, dev,
                                      ctxt);
  syclexp::destroy_image_handle(handles.input, dev, ctxt);
  syclexp::destroy_image_handle(handles.output, dev, ctxt);
  syclexp::unmap_external_image_memory(
      handles.inputMemHandle, syclexp::image_type::standard, dev, ctxt);
  syclexp::unmap_external_image_memory(
      handles.outputMemHandle, syclexp::image_type::standard, dev, ctxt);
  syclexp::release_external_memory(handles.inputExternalMem, dev, ctxt);
  syclexp::release_external_memory(handles.outputExternalMem, dev, ctxt);
}

void run_sycl(sycl::queue q, sycl::range<3> globalSize,
              sycl::range<3> localSize, handles_t handles) {
  using VecType = sycl::vec<float, 4>;

  // Extension: wait for imported semaphore
  q.ext_oneapi_wait_external_semaphore(handles.syclWaitExternalSemaphore,
                                       waitSemaphoreValue);

  try {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class mulKernel>(
          sycl::nd_range<3>{globalSize, localSize}, [=](sycl::nd_item<3> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);
            size_t dim2 = it.get_global_id(2);

            VecType px1 = syclexp::fetch_image<VecType>(
                handles.input, sycl::int3(dim0, dim1, dim2));

            syclexp::write_image<VecType>(handles.output,
                                          sycl::int3(dim0, dim1, dim2),
                                          VecType(px1 * 2.f));
          });
    });

    // Extension: signal imported semaphore
    doneSemaphoreValue += 1;
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_signal_external_semaphore(
          handles.syclDoneExternalSemaphore, doneSemaphoreValue);
    });

  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }
}
} // namespace util

bool run_test() {

  using DType = float;

  unsigned int seed = 0;
  int numChannels = 4;
  sycl::range<3> dims = {272, 144, 4};
  sycl::range<3> localSize{16, 16, 4};
  sycl::image_channel_type channelType = sycl::image_channel_type::fp32;
  sycl::image_channel_order channelOrder = sycl::image_channel_order::rgba;

  uint32_t width = static_cast<uint32_t>(dims[0]);
  uint32_t height = static_cast<uint32_t>(dims[1]);
  uint32_t depth = static_cast<uint32_t>(dims[2]);

  size_t numElems = dims[0] * dims[1] * dims[2];
  VkImageType imgType = VK_IMAGE_TYPE_3D;

  VkFormat format = vkutil::to_vulkan_format(channelOrder, channelType);
  const size_t imageSizeBytes = numElems * numChannels * sizeof(DType);

  vkutil::vulkan_image_test_resources_t inVkImgRes(
      imgType, format, {width, height, depth}, imageSizeBytes);
  vkutil::vulkan_image_test_resources_t outVkImgRes(
      imgType, format, {width, height, depth}, imageSizeBytes);

  printString("Populating staging buffer\n");
  // Populate staging memory
  std::vector<DType> inputVector(numElems * numChannels, static_cast<DType>(0));
  std::srand(seed);
  bindless_helpers::fill_rand(inputVector);

  DType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inVkImgRes.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < (numElems * numChannels); ++i) {
    inputStagingData[i] = inputVector[i];
  }
  vkUnmapMemory(vk_device, inVkImgRes.stagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput =
        vkutil::createImageMemoryBarrier(inVkImgRes.vkImage, 1 /*mipLevels*/);

    VkImageMemoryBarrier barrierOutput =
        vkutil::createImageMemoryBarrier(outVkImgRes.vkImage, 1 /*mipLevels*/);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput);

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
    // Always wait for this queue to finish as there are no semaphores yet to
    // sync
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  // Create semaphore to later import in SYCL
  printString("Creating semaphores\n");
  VkSemaphore syclWaitSemaphore;
  {

    VkSemaphoreTypeCreateInfo stci = {};
    stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = waitSemaphoreValue;

    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    esci.pNext = &stci;
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

    VkSemaphoreTypeCreateInfo stci = {};
    stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = doneSemaphoreValue;

    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    esci.pNext = &stci;
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
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inVkImgRes.stagingBuffer,
                           inVkImgRes.vkImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    waitSemaphoreValue += 1;
    // Create const pointer to waitSemaphoreValue to use in timelineInfo.
    const uint64_t *constSemaphoreValue = &waitSemaphoreValue;

    VkTimelineSemaphoreSubmitInfo timelineInfo;
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.pNext = NULL;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = constSemaphoreValue;

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];

    submission.signalSemaphoreCount = 1;
    submission.pSignalSemaphores = &syclWaitSemaphore;
    submission.pNext = &timelineInfo;
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    // Do not wait when using semaphores as they can handle the kernel execution
    // order.
  }

  printString("Getting memory interop handles\n");

  // Pass memory to SYCL for modification
  auto globalSize = dims;
#ifdef _WIN32
  auto inputMemHandle = vkutil::getMemoryWin32Handle(inVkImgRes.imageMemory);
  auto outputMemHandle = vkutil::getMemoryWin32Handle(outVkImgRes.imageMemory);
#else
  auto inputMemHandle = vkutil::getMemoryOpaqueFD(inVkImgRes.imageMemory);
  auto outputMemHandle = vkutil::getMemoryOpaqueFD(outVkImgRes.imageMemory);
#endif

  printString("Getting semaphore interop handles\n");

  // Pass semaphores to SYCL for synchronization
#ifdef _WIN32
  auto syclWaitSemaphoreHandle =
      vkutil::getSemaphoreWin32Handle(syclWaitSemaphore);
  auto syclDoneSemaphoreHandle =
      vkutil::getSemaphoreWin32Handle(syclDoneSemaphore);
#else
  auto syclWaitSemaphoreHandle =
      vkutil::getSemaphoreOpaqueFD(syclWaitSemaphore);
  auto syclDoneSemaphoreHandle =
      vkutil::getSemaphoreOpaqueFD(syclDoneSemaphore);
#endif

  printString("Calling into SYCL with interop memory and semaphore handles\n");

  sycl::device dev;
  sycl::queue q{dev, {sycl::property::queue::in_order{}}};
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  syclexp::image_descriptor desc(globalSize, numChannels, channelType);

  const size_t imgSize = globalSize.size() * sizeof(DType) * numChannels;

  auto handles = util::create_sycl_handles(
      ctxt, dev, inputMemHandle, outputMemHandle, syclWaitSemaphoreHandle,
      syclDoneSemaphoreHandle, imgSize, desc);

  util::run_sycl(q, globalSize, localSize, handles);

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

    // Create const pointer to doneSemaphoreValue to use in timelineInfo.
    const uint64_t *constSemaphoreValue = &doneSemaphoreValue;

    VkTimelineSemaphoreSubmitInfo timelineInfo;
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.pNext = NULL;
    timelineInfo.waitSemaphoreValueCount = 1;
    timelineInfo.pWaitSemaphoreValues = constSemaphoreValue;

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[1];

    submission.waitSemaphoreCount = 1;
    submission.pWaitSemaphores = &syclDoneSemaphore;
    submission.pNext = &timelineInfo;

    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    // Wait here to ensure vulkan jobs are done before verifying data and
    // starting next test.
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  printString("Validating\n");
  // Validate that SYCL made changes to the memory
  bool validated = true;
  DType *outputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, outVkImgRes.stagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&outputStagingData));

  for (int i = 0; i < (numElems * numChannels); ++i) {
    DType expected = inputVector[i] * 2.f;
    // Use helper function to determine if data is accepted
    // For integers, exact results are expected
    // For floats, accepted error variance is passed
    if (!util::is_equal(outputStagingData[i], expected)) {
      std::cerr << "Result mismatch! actual[" << i
                << "] == " << outputStagingData[i]
                << " : expected == " << expected << "\n";
      validated = false;
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
  cleanup_sycl(ctxt, dev, handles);

  return validated;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  sycl::device dev;

  if (vkutil::setupDevice(dev) != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  auto runOk = run_test();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  if (runOk) {
    std::cout << "All tests passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
