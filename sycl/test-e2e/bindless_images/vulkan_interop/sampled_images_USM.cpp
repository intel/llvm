// REQUIRES: cuda
// REQUIRES: vulkan
// REQUIRES: build-and-run-mode

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} %t.out

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct handles_t {
  syclexp::sampled_image_handle imgInput;
  void *imgMem;
  syclexp::external_mem inputInteropMemHandle;
};

template <typename InteropHandleT>
handles_t create_test_handles(sycl::context &ctxt, sycl::device &dev,
                              const syclexp::bindless_image_sampler &samp,
                              InteropHandleT interopHandle,
                              syclexp::image_descriptor desc,
                              const size_t imgSize,
                              const size_t imgPitchBytes) {
  // Extension: external memory descriptor
#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      inputExtMemDesc{interopHandle,
                      syclexp::external_mem_handle_type::win32_nt_handle,
                      imgSize};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> inputExtMemDesc{
      interopHandle, syclexp::external_mem_handle_type::opaque_fd, imgSize};
#endif

  // Extension: interop mem handle imported from file descriptor
  syclexp::external_mem inputInteropMemHandle =
      syclexp::import_external_memory(inputExtMemDesc, dev, ctxt);

  // Extension: interop mem handle imported from file descriptor
  void *mappedImgMem = syclexp::map_external_linear_memory(
      inputInteropMemHandle, 0, imgSize, dev, ctxt);

  // Extension: create the image and return the handle
  syclexp::sampled_image_handle imgInput =
      syclexp::create_image(mappedImgMem, imgPitchBytes, samp, desc, dev, ctxt);

  return {imgInput, mappedImgMem, inputInteropMemHandle};
}

template <typename InteropHandleT, int NDims = 2, typename DType, int NChannels,
          sycl::image_channel_type CType, typename KernelName>
bool run_sycl(InteropHandleT inputInteropMemHandle, size_t imgPitchBytes,
              sycl::range<NDims> globalSize, sycl::range<NDims> localSize) {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  syclexp::image_descriptor desc(globalSize, NChannels, CType);

  syclexp::bindless_image_sampler samp(
      sycl::addressing_mode::repeat,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::linear);

  const auto numElems = globalSize.size();

  auto width = globalSize[0];
  auto height = globalSize[1];

  const size_t img_size =
      (imgPitchBytes / 8) * height * sizeof(DType) * NChannels;

  using VecType = sycl::vec<DType, NChannels>;

  auto handles = create_test_handles(ctxt, dev, samp, inputInteropMemHandle,
                                     desc, img_size, imgPitchBytes);

  sycl::range<NDims> outBufferRange;
  outBufferRange = sycl::range<NDims>{height, width};

  std::vector<VecType> out(width * height);
  try {
    sycl::buffer<VecType, NDims> buf((VecType *)out.data(), outBufferRange);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.template get_access<sycl::access_mode::write>(
          cgh, outBufferRange);
      cgh.parallel_for<KernelName>(
          sycl::nd_range<NDims>{globalSize, localSize},
          [=](sycl::nd_item<NDims> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;
            float fdim1 = float(dim1 + 0.5f) / (float)height;

            // Extension: sample image data from handle (Vulkan imported)
            VecType pixel = syclexp::sample_image<VecType>(
                handles.imgInput, sycl::float2(fdim0, fdim1));

            pixel *= static_cast<DType>(10.1f);
            outAcc[sycl::id{dim1, dim0}] = pixel;
          });
    });
    q.wait_and_throw();

    syclexp::destroy_image_handle(handles.imgInput, dev, ctxt);
    sycl::free(handles.imgMem, ctxt);
    syclexp::release_external_memory(handles.inputInteropMemHandle, dev, ctxt);
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }

  printString("Validating\n");
  bool validated = true;

  for (int i = 0; i < numElems; ++i) {
    bool mismatch = false;

    VecType expected = bindless_helpers::init_vector<DType, NChannels>(i) *
                       static_cast<DType>(10.1f);
    if (!bindless_helpers::equal_vec<DType, NChannels>(out[i], expected)) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected
                << ", Actual: " << out[i] << "\n";
#else
      break;
#endif
    }
  }

  if (validated) {
    printString("Results are correct!\n");
  }

  return validated;
}

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> localSize,
              unsigned int seed = 0) {
  uint32_t width = static_cast<uint32_t>(dims[0]);
  uint32_t height = 1;
  uint32_t depth = 1;

  size_t numElems = dims[0];
  VkImageType imgType = VK_IMAGE_TYPE_1D;

  if constexpr (NDims > 1) {
    numElems *= dims[1];
    height = static_cast<uint32_t>(dims[1]);
    imgType = VK_IMAGE_TYPE_2D;
  }

  using VecType = sycl::vec<DType, NChannels>;

  VkFormat format = vkutil::to_vulkan_format(COrder, CType);

  printString("Creating input image\n");
  // Create input image memory
  auto inputImage = vkutil::createImage(imgType, format, {width, height, depth},
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                        1 /*mipLevels*/, true /*linearTiling*/);

  const size_t imageRowPitchBytes =
      static_cast<size_t>(vkutil::getImageRowPitch(inputImage));
  const size_t imageRowPitchElements = imageRowPitchBytes / 8;
  const size_t imageSizeBytes =
      imageRowPitchElements * height * depth * NChannels * sizeof(DType);
  const size_t stagingBufferSizeBytes = imageSizeBytes;

  VkMemoryRequirements memRequirements = {};
  memRequirements.size = imageSizeBytes;
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
  auto inputMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, inputImageMemoryTypeIndex, inputImage);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, inputImage, inputMemory,
                                  0 /*memoryOffset*/));

  printString("Creating staging buffers\n");
  // Create input staging memory
  auto inputStagingBuffer = vkutil::createBuffer(
      stagingBufferSizeBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      inputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto inputStagingMemory = vkutil::allocateDeviceMemory(
      stagingBufferSizeBytes, inputStagingMemoryTypeIndex, nullptr /*image*/,
      false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, inputStagingBuffer,
                                   inputStagingMemory, 0 /*memoryOffset*/));

  printString("Populating staging buffer\n");
  // Populate staging memory
  VecType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      int index = (j * imageRowPitchElements) + i;
      int element = (j * width) + i;
      inputStagingData[index] =
          bindless_helpers::init_vector<DType, NChannels>(element);
    }
  }
  vkUnmapMemory(vk_device, inputStagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput =
        vkutil::createImageMemoryBarrier(inputImage, 1 /*mipLevels*/);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  printString("Copying staging memory to images\n");
  // Copy staging to main image memory
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, depth};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.bufferRowLength = imageRowPitchElements;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inputStagingBuffer,
                           inputImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  printString("Getting memory file descriptors\n");
  // Pass memory to SYCL for modification

#ifdef _WIN32
  auto input_mem_handle = vkutil::getMemoryWin32Handle(inputMemory);
#else
  auto input_mem_handle = vkutil::getMemoryOpaqueFD(inputMemory);
#endif

  printString("Calling into SYCL with interop memory handle\n");

  bool validated = run_sycl<decltype(input_mem_handle), NDims, DType, NChannels,
                            CType, KernelName>(
      input_mem_handle, imageRowPitchBytes, dims, localSize);

  // Cleanup
  vkDestroyBuffer(vk_device, inputStagingBuffer, nullptr);
  vkDestroyImage(vk_device, inputImage, nullptr);
  vkFreeMemory(vk_device, inputStagingMemory, nullptr);
  vkFreeMemory(vk_device, inputMemory, nullptr);

  return validated;
}

bool run_tests() {
  bool valid = run_test<2, float, 4, sycl::image_channel_type::fp32,
                        sycl::image_channel_order::rgba, class float_2d_4c>(
      {16, 16}, {16, 16}, 0);

  valid &= run_test<2, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float_2d_2c>(
      {1024, 1024}, {32, 32}, 0);

  return valid;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  sycl::device dev;

  if (vkutil::setupDevice(dev.get_info<sycl::info::device::name>()) !=
      VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Compute pipeline setup failed!\n";
    return EXIT_FAILURE;
  }

  bool result_ok = run_tests();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  if (result_ok) {
    std::cout << "All tests passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
