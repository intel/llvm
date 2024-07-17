// REQUIRES: cuda
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

// Define NOMINMAX to enable compilation on Windows
#define NOMINMAX

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct handles_t {
  syclexp::sampled_image_handle imgInput;
  syclexp::image_mem_handle imgMem;
  syclexp::interop_mem_handle inputInteropMemHandle;
};

template <typename InteropMemHandleT>
handles_t create_handles(sycl::context &ctxt, sycl::device &dev,
                         const syclexp::bindless_image_sampler &samp,
                         InteropMemHandleT inputImgInteropHandle,
                         syclexp::image_descriptor desc, size_t imgSize) {

  // Extension: external memory descriptor
#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      inputExtMemDesc{inputImgInteropHandle,
                      syclexp::external_mem_handle_type::win32_nt_handle,
                      imgSize};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> inputExtMemDesc{
      inputImgInteropHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSize};
#endif

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

  return {imgInput, inputMappedMemHandle, inputInteropMemHandle};
}

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, typename InteropMemHandleT,
          typename KernelName>
bool run_sycl(sycl::range<NDims> globalSize, sycl::range<NDims> localSize,
              InteropMemHandleT inputImgInteropHandle, size_t mipLevels,
              size_t reqSize) {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  syclexp::image_descriptor desc(globalSize, NChannels, CType,
                                 syclexp::image_type::mipmap, mipLevels);

  syclexp::bindless_image_sampler samp(
      sycl::addressing_mode::repeat,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::nearest, sycl::filtering_mode::linear, 0.0f,
      (float)mipLevels, 8.0f);

  const auto mip0Elems = globalSize.size();

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
      create_handles(ctxt, dev, samp, inputImgInteropHandle, desc, reqSize);

  std::vector<VecType> out(mip0Elems);
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
              VecType pixel1 = syclexp::sample_mipmap<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  handles.imgInput, sycl::float3(fdim0, fdim1, fdim2), 0.0f);

              VecType pixel2 = syclexp::sample_mipmap<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  handles.imgInput, sycl::float3(fdim0, fdim1, fdim2), 1.0f);

              outAcc[sycl::id{dim2, dim1, dim0}] = pixel1 + pixel2;
            } else {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              // Normalize coordinates -- +0.5 to look towards centre of pixel
              float fdim0 = float(dim0 + 0.5f) / (float)width;
              float fdim1 = float(dim1 + 0.5f) / (float)height;

              // Extension: read image data from handle (Vulkan imported)
              VecType pixel1 = syclexp::sample_mipmap<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  handles.imgInput, sycl::float2(fdim0, fdim1), 0.0f);

              VecType pixel2 = syclexp::sample_mipmap<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  handles.imgInput, sycl::float2(fdim0, fdim1), 1.0f);

              outAcc[sycl::id{dim1, dim0}] = pixel1 + pixel2;
            }
          });
    });
    q.wait_and_throw();

    syclexp::destroy_image_handle(handles.imgInput, dev, ctxt);
    syclexp::free_image_mem(handles.imgMem, syclexp::image_type::mipmap, dev,
                            ctxt);
    syclexp::release_external_memory(handles.inputInteropMemHandle, dev, ctxt);
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }

  printString("Validating\n");
  // Expected is sum of first two levels in the mipmap
  // Each subsequent level repeats in each dimension
  bool validated = true;
  if constexpr (NDims == 3) {
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < height; ++j) {
        for (int k = 0; k < depth; ++k) {
          bool mismatch = false;
          float norm_coord_x = ((i + 0.5f) / (float)width);
          int x = norm_coord_x * (width >> 1);
          float norm_coord_y = ((j + 0.5f) / (float)height);
          int y = norm_coord_y * (height >> 1);
          float norm_coord_z = ((k + 0.5f) / (float)depth);
          int z = norm_coord_z * (depth >> 1);

          VecType expected = bindless_helpers::init_vector<DType, NChannels>(
                                 i + width * (j + height * k)) +
                             bindless_helpers::init_vector<DType, NChannels>(
                                 x + (width / 2) * (y + (height / 2) * z));

          if (!bindless_helpers::equal_vec<DType, NChannels>(
                  out[i + width * (j + height * k)], expected)) {
            mismatch = true;
            validated = false;
          }
          if (mismatch) {
#ifdef VERBOSE_PRINT
            std::cout << "Result mismatch! Expected: " << expected
                      << ", Actual: " << out[i + width * (j + height * k)]
                      << "\n";
#else
            break;
#endif
          }
        }
      }
    }
  } else {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        bool mismatch = false;
        float norm_coord_x = ((i + 0.5f) / (float)width);
        int x = norm_coord_x * (width >> 1);
        float norm_coord_y = ((j + 0.5f) / (float)height);
        int y = norm_coord_y * (height >> 1);

        VecType expected =
            bindless_helpers::init_vector<DType, NChannels>(j + (width * i)) +
            bindless_helpers::init_vector<DType, NChannels>(y +
                                                            (width / 2 * x));

        if (!bindless_helpers::equal_vec<DType, NChannels>(out[j + (width * i)],
                                                           expected)) {
          mismatch = true;
          validated = false;
        }
        if (mismatch) {
#ifdef VERBOSE_PRINT
          std::cout << "Result mismatch! Expected: " << expected
                    << ", Actual: " << out[j + (width * i)] << "\n";
#else
          break;
#endif
        }
      }
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
              size_t mipLevels, unsigned int seed = 0) {

  uint32_t width = static_cast<uint32_t>(dims[0]);
  uint32_t height = 1;
  uint32_t depth = 1;

  size_t mip0Elems = dims[0];
  VkImageType imgType = VK_IMAGE_TYPE_1D;

  if constexpr (NDims > 1) {
    mip0Elems *= dims[1];
    height = static_cast<uint32_t>(dims[1]);
    imgType = VK_IMAGE_TYPE_2D;
  }
  if constexpr (NDims > 2) {
    mip0Elems *= dims[2];
    depth = static_cast<uint32_t>(dims[2]);
    imgType = VK_IMAGE_TYPE_3D;
  }

  using VecType = sycl::vec<DType, NChannels>;
  VkFormat format = vkutil::to_vulkan_format(COrder, CType);

  printString("Creating input image\n");
  // Create input image memory
  auto inputImage = vkutil::createImage(imgType, format, {width, height, depth},
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                            VK_IMAGE_USAGE_STORAGE_BIT,
                                        mipLevels);
  VkMemoryRequirements memRequirements;
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
  auto inputMemory = vkutil::allocateDeviceMemory(memRequirements.size,
                                                  inputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, inputImage, inputMemory,
                                  0 /*memoryOffset*/));

  printString("Creating staging buffers\n");
  // Create input staging memory
  auto inputStagingBuffer = vkutil::createBuffer(
      memRequirements.size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      inputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto inputStagingMemory = vkutil::allocateDeviceMemory(
      memRequirements.size, inputStagingMemoryTypeIndex, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, inputStagingBuffer,
                                   inputStagingMemory, 0 /*memoryOffset*/));

  printString("Populating staging buffer\n");
  // Populate staging memory
  VecType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            memRequirements.size, 0 /*flags*/,
                            (void **)&inputStagingData));

  // Set input data as each mip level -- 0 -> mip size e.g. (0,1,...,63,0,1,...)
  size_t offset = 0;
  size_t mipElems = mip0Elems;
  for (int i = 0; i < mipLevels; ++i) {
    mipElems = (std::max(width >> i, (uint32_t)1) *
                std::max(height >> i, (uint32_t)1) *
                std::max(depth >> i, (uint32_t)1));
    for (int j = 0; j < mipElems; ++j) {
      inputStagingData[j + offset] =
          bindless_helpers::init_vector<DType, NChannels>(j);
    }
    offset += mipElems;
  }
  vkUnmapMemory(vk_device, inputStagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput =
        vkutil::createImageMemoryBarrier(inputImage, mipLevels);

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
    VkDeviceSize currentOffset{0};

    // Copy each mip level individually
    for (int i = 0; i < mipLevels; ++i) {
      VkCommandBufferBeginInfo cbbi = {};
      cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      VkBufferImageCopy copyRegion = {};
      copyRegion.imageExtent = {std::max(width >> i, (uint32_t)1),
                                std::max(height >> i, (uint32_t)1),
                                std::max(depth >> i, (uint32_t)1)};
      copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      copyRegion.imageSubresource.layerCount = 1;
      copyRegion.imageSubresource.mipLevel = i;
      copyRegion.bufferOffset = currentOffset;

      currentOffset += std::max(width >> i, (uint32_t)1) *
                       std::max(height >> i, (uint32_t)1) *
                       std::max(depth >> i, (uint32_t)1) * NChannels *
                       sizeof(DType);

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
  }

  printString("Getting memory file descriptors and calling into SYCL\n");
  // Pass memory to SYCL for modification
#ifdef _WIN32
  auto inputMemHandle = vkutil::getMemoryWin32Handle(inputMemory);
#else
  auto inputMemHandle = vkutil::getMemoryOpaqueFD(inputMemory);
#endif
  bool result = run_sycl<NDims, DType, NChannels, CType,
                         decltype(inputMemHandle), KernelName>(
      dims, localSize, inputMemHandle, mipLevels, memRequirements.size);

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
      {16, 16}, {2, 2}, 2, 0);

  valid &= run_test<2, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float_2d_large>(
      {8, 8}, {4, 2}, 2, 0);

  valid &= run_test<3, char, 2, sycl::image_channel_type::signed_int8,
                    sycl::image_channel_order::rg, class float_3d>(
      {8, 8, 8}, {2, 2, 2}, 2, 0);

  valid &= run_test<2, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::r, class uint32_2d>(
      {32, 32}, {4, 2}, 2, 0);

  valid &= run_test<3, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                    sycl::image_channel_order::rgba, class uint_3d_large>(
      {8, 8, 8}, {2, 2, 4}, 2, 0);

  valid &= run_test<2, int32_t, 1, sycl::image_channel_type::signed_int32,
                    sycl::image_channel_order::r, class int32_2d>({64, 64},
                                                                  {4, 2}, 2, 0);

  valid &= run_test<3, int32_t, 2, sycl::image_channel_type::signed_int32,
                    sycl::image_channel_order::rg, class int32_3d>(
      {8, 8, 8}, {4, 2, 4}, 2, 0);

  valid &= run_test<3, int16_t, 1, sycl::image_channel_type::signed_int16,
                    sycl::image_channel_order::r, class int16_3d>(
      {32, 32, 32}, {4, 2, 4}, 2, 0);

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
