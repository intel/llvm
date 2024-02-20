// REQUIRES: linux
// REQUIRES: cuda
// REQUIRES: vulkan

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %link-vulkan %s -o %t.out
// RUN: %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include <sycl/sycl.hpp>

#include "vulkan_common.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename DType, int NChannels>
std::ostream &operator<<(std::ostream &os,
                         const sycl::vec<DType, NChannels> &vec) {
  std::string str{""};
  for (int i = 0; i < NChannels; ++i) {
    str += std::to_string(vec[i]) + ",";
  }
  str.pop_back();
  os << str;
  return os;
}

template <typename DType, int NChannels>
bool equal_vec(sycl::vec<DType, NChannels> v1, sycl::vec<DType, NChannels> v2) {
  for (int i = 0; i < NChannels; ++i) {
    if (v1[i] != v2[i]) {
      return false;
    }
  }

  return true;
}

template <typename DType, int NChannel>
constexpr sycl::vec<DType, NChannel> initVector(DType val) {
  if constexpr (NChannel == 1) {
    return sycl::vec<DType, NChannel>{val};
  } else if constexpr (NChannel == 2) {
    return sycl::vec<DType, NChannel>{val, val};
  } else if constexpr (NChannel == 4) {
    return sycl::vec<DType, NChannel>{val, val, val, val};
  } else {
    std::cerr << "unsupported number of channels " << NChannel << "\n";
    exit(-1);
  }
}

struct handles_t {
  sycl::ext::oneapi::experimental::sampled_image_handle imgInput;
  sycl::ext::oneapi::experimental::interop_mem_handle inputInteropMemHandle;
};

handles_t create_test_handles(
    sycl::context &ctxt, sycl::device &dev,
    const sycl::ext::oneapi::experimental::bindless_image_sampler &samp,
    int input_image_fd, sycl::ext::oneapi::experimental::image_descriptor desc,
    const size_t imgSize) {
  namespace syclexp = sycl::ext::oneapi::experimental;
  // Extension: external memory descriptor
  syclexp::external_mem_descriptor<syclexp::external_mem_fd> inputExtMemDesc{
      input_image_fd, imgSize};

  // Extension: interop mem handle imported from file descriptor
  syclexp::interop_mem_handle inputInteropMemHandle =
      syclexp::import_external_memory(inputExtMemDesc, dev, ctxt);

  // Extension: interop mem handle imported from file descriptor
  syclexp::image_mem_handle inputMappedMemHandle =
      syclexp::map_external_image_memory(inputInteropMemHandle, desc, dev,
                                         ctxt);

  // Extension: create the image and return the handle
  syclexp::sampled_image_handle imgInput =
      syclexp::create_image(inputMappedMemHandle, samp, desc, dev, ctxt);

  return {imgInput, inputInteropMemHandle};
}

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName>
bool run_sycl(sycl::range<NDims> globalSize, sycl::range<NDims> localSize,
              int input_image_fd) {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  namespace syclexp = sycl::ext::oneapi::experimental;

  // Image descriptor - mapped to Vulkan image layout
  syclexp::image_descriptor desc(globalSize, COrder, CType,
                                 syclexp::image_type::interop,
                                 1 /*num_levels*/);

  syclexp::bindless_image_sampler samp(
      sycl::addressing_mode::repeat,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::linear);

  const auto numElems = globalSize.size();

  const size_t img_size = numElems * sizeof(DType) * NChannels;

  auto width = globalSize[0];
  auto height = globalSize[1];
  auto depth = 1UL;

  sycl::range<NDims> outBufferRange;
  if constexpr (NDims == 3) {
    depth = globalSize[2];
    outBufferRange = sycl::range<NDims>{depth, height, width};
  } else {
    outBufferRange = sycl::range<NDims>{height, width};
  }

  using VecType = sycl::vec<DType, NChannels>;

  auto handles =
      create_test_handles(ctxt, dev, samp, input_image_fd, desc, img_size);

  std::vector<VecType> out(numElems);
  try {

    sycl::buffer<VecType, NDims> buf((VecType *)out.data(), outBufferRange);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.template get_access<sycl::access_mode::write>(
          cgh, outBufferRange);
      cgh.parallel_for<KernelName>(
          sycl::nd_range<NDims>{globalSize, localSize},
          [=](sycl::nd_item<NDims> it) {
            if constexpr (NDims == 3) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);

              // Normalize coordinates -- +0.5 to look towards centre of pixel
              float fdim0 = float(dim0 + 0.5f) / (float)width;
              float fdim1 = float(dim1 + 0.5f) / (float)height;
              float fdim2 = float(dim2 + 0.5f) / (float)depth;

              // Extension: read image data from handle (Vulkan imported)
              VecType pixel;
              pixel = syclexp::read_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  handles.imgInput, sycl::float3(fdim0, fdim1, fdim2));

              pixel *= static_cast<DType>(10.1f);
              outAcc[sycl::id{dim2, dim1, dim0}] = pixel;
            } else {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              // Normalize coordinates -- +0.5 to look towards centre of pixel
              float fdim0 = float(dim0 + 0.5f) / (float)width;
              float fdim1 = float(dim1 + 0.5f) / (float)height;

              // Extension: read image data from handle (Vulkan imported)
              VecType pixel = syclexp::read_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  handles.imgInput, sycl::float2(fdim0, fdim1));

              pixel *= static_cast<DType>(10.1f);
              outAcc[sycl::id{dim1, dim0}] = pixel;
            }
          });
    });
    q.wait_and_throw();

    syclexp::destroy_image_handle(handles.imgInput, dev, ctxt);
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
  for (int i = 0; i < globalSize.size(); i++) {
    bool mismatch = false;
    VecType expected =
        initVector<DType, NChannels>(i) * static_cast<DType>(10.1f);
    if (!equal_vec<DType, NChannels>(out[i], expected)) {
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
  if constexpr (NDims > 2) {
    numElems *= dims[2];
    depth = static_cast<uint32_t>(dims[2]);
    imgType = VK_IMAGE_TYPE_3D;
  }

  using VecType = sycl::vec<DType, NChannels>;

  VkFormat format = vkutil::to_vulkan_format(COrder, CType);
  const size_t imageSizeBytes = numElems * NChannels * sizeof(DType);

  printString("Creating input image\n");
  // Create input image memory
  auto inputImage = vkutil::createImage(imgType, format, {width, height, depth},
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                            VK_IMAGE_USAGE_STORAGE_BIT);
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  auto inputMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, inputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, inputImage, inputMemory,
                                  0 /*memoryOffset*/));

  printString("Creating staging buffers\n");
  // Create input staging memory
  auto inputStagingBuffer = vkutil::createBuffer(
      imageSizeBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      inputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto inputStagingMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, inputStagingMemoryTypeIndex, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, inputStagingBuffer,
                                   inputStagingMemory, 0 /*memoryOffset*/));

  printString("Populating staging buffer\n");
  // Populate staging memory
  VecType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < numElems; ++i) {
    inputStagingData[i] = initVector<DType, NChannels>(i);
  }
  vkUnmapMemory(vk_device, inputStagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput =
        vkutil::createImageMemoryBarrier(inputImage);

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

  printString("Getting memory file descriptors and calling into SYCL\n");
  // Pass memory to SYCL for modification
  auto input_fd = vkutil::getMemoryOpaqueFD(inputMemory);
  bool result = run_sycl<NDims, DType, NChannels, CType, COrder, KernelName>(
      dims, localSize, input_fd);

  // Cleanup
  vkDestroyBuffer(vk_device, inputStagingBuffer, nullptr);
  vkDestroyImage(vk_device, inputImage, nullptr);
  vkFreeMemory(vk_device, inputStagingMemory, nullptr);
  vkFreeMemory(vk_device, inputMemory, nullptr);

  return result;
}

bool run_tests() {
  bool valid = run_test<2, float, 4, sycl::image_channel_type::fp32,
                        sycl::image_channel_order::rgba, class float_2d>(
      {16, 16}, {2, 2}, 0);

  valid &= run_test<2, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float_2d_large>(
      {1024, 1024}, {4, 2}, 0);

  valid &= run_test<3, char, 2, sycl::image_channel_type::signed_int8,
                    sycl::image_channel_order::rg, class float_3d>(
      {256, 16, 2}, {2, 2, 2}, 0);

  valid &= run_test<2, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::r, class uint32_2d>({64, 32},
                                                                   {4, 2}, 0);

  valid &= run_test<3, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rgba, class uint_3d_large>(
      {1024, 256, 16}, {2, 2, 4}, 0);

  valid &= run_test<2, int32_t, 1, sycl::image_channel_type::signed_int32,
                    sycl::image_channel_order::r, class int32_2d>({64, 32},
                                                                  {4, 2}, 0);

  valid &= run_test<3, int32_t, 2, sycl::image_channel_type::signed_int32,
                    sycl::image_channel_order::rg, class int32_3d>(
      {64, 32, 64}, {4, 2, 4}, 0);

  valid &= run_test<3, int16_t, 1, sycl::image_channel_type::signed_int16,
                    sycl::image_channel_order::r, class int16_3d>({64, 32, 64},
                                                                  {4, 2, 4}, 0);

  return valid;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupDevice("NVIDIA") != VK_SUCCESS) {
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
