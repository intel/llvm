// REQUIRES: cuda || (windows && level_zero && gpu-intel-dg2)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if any-device-is-level_zero %{ -Wno-ignored-attributes -DTEST_L0_SUPPORTED_VK_FORMAT %}
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct handles_t {
  syclexp::sampled_image_handle imgInput;
  syclexp::image_mem_handle imgMem;
  syclexp::external_mem inputExternalMem;
};

template <typename DType, sycl::image_channel_type CType> struct OutputType {
  using type = DType;
};

template <> struct OutputType<uint8_t, sycl::image_channel_type::unorm_int8> {
  using type = float;
};

template <typename InteropHandleT>
handles_t create_test_handles(sycl::context &ctxt, sycl::device &dev,
                              const syclexp::bindless_image_sampler &samp,
                              InteropHandleT interopHandle,
                              syclexp::image_descriptor desc,
                              const size_t imgSize) {
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

  // Extension: external memory imported from file descriptor
  syclexp::external_mem inputExternalMem =
      syclexp::import_external_memory(inputExtMemDesc, dev, ctxt);

  // Extension: mapped memory handle from external memory
  syclexp::image_mem_handle inputMappedMemHandle =
      syclexp::map_external_image_memory(inputExternalMem, desc, dev, ctxt);

  // Extension: create the image and return the handle
  syclexp::sampled_image_handle imgInput =
      syclexp::create_image(inputMappedMemHandle, samp, desc, dev, ctxt);

  return {imgInput, inputMappedMemHandle, inputExternalMem};
}

template <typename InteropHandleT, int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, typename KernelName>
bool run_sycl(InteropHandleT inputInteropMemHandle,
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

  const size_t img_size = numElems * sizeof(DType) * NChannels;

  auto width = globalSize[0];
  auto height = 1UL;
  auto depth = 1UL;

  sycl::range<NDims> outBufferRange;
  if constexpr (NDims == 3) {
    height = globalSize[1];
    depth = globalSize[2];
    outBufferRange = sycl::range<NDims>{depth, height, width};
  } else if constexpr (NDims == 2) {
    height = globalSize[1];
    outBufferRange = sycl::range<NDims>{height, width};
  } else {
    outBufferRange = sycl::range<NDims>{width};
  }

  using OutType = typename OutputType<DType, CType>::type;
  using VecType = sycl::vec<OutType, NChannels>;

  auto handles = create_test_handles(ctxt, dev, samp, inputInteropMemHandle,
                                     desc, img_size);

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

              // Extension: sample image data from handle (Vulkan imported)
              VecType pixel;
              pixel = syclexp::sample_image<
                  std::conditional_t<NChannels == 1, OutType, VecType>>(
                  handles.imgInput, sycl::float3(fdim0, fdim1, fdim2));

              pixel /= static_cast<OutType>(2.f);
              outAcc[sycl::id{dim2, dim1, dim0}] = pixel;
            } else if constexpr (NDims == 2) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              // Normalize coordinates -- +0.5 to look towards centre of pixel
              float fdim0 = float(dim0 + 0.5f) / (float)width;
              float fdim1 = float(dim1 + 0.5f) / (float)height;

              // Extension: sample image data from handle (Vulkan imported)
              VecType pixel = syclexp::sample_image<
                  std::conditional_t<NChannels == 1, OutType, VecType>>(
                  handles.imgInput, sycl::float2(fdim0, fdim1));

              pixel /= static_cast<OutType>(2.f);
              outAcc[sycl::id{dim1, dim0}] = pixel;
            } else {
              size_t dim0 = it.get_global_id(0);

              // Normalize coordinates -- +0.5 to look towards centre of pixel
              float fdim0 = float(dim0 + 0.5f) / (float)width;

              // Extension: sample image data from handle (Vulkan imported)
              VecType pixel = syclexp::sample_image<
                  std::conditional_t<NChannels == 1, OutType, VecType>>(
                  handles.imgInput, fdim0);

              pixel /= static_cast<OutType>(2.f);
              outAcc[dim0] = pixel;
            }
          });
    });
    q.wait_and_throw();

    syclexp::destroy_image_handle(handles.imgInput, dev, ctxt);
    syclexp::free_image_mem(handles.imgMem, syclexp::image_type::standard, dev,
                            ctxt);
    syclexp::release_external_memory(handles.inputExternalMem, dev, ctxt);
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
        bindless_helpers::init_vector<OutType, NChannels>(static_cast<OutType>(
            CType == sycl::image_channel_type::unorm_int8 ? 0.5f : (i / 2.f)));
    if (!bindless_helpers::equal_vec<OutType, NChannels>(out[i], expected)) {
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
#ifdef VERBOSE_PRINT
    std::cout << "\tTest passed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(CType) << "\n";
#endif
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
                                            VK_IMAGE_USAGE_STORAGE_BIT,
                                        1 /*mipLevels*/);
  VkMemoryRequirements memRequirements;
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
  auto inputMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, inputImageMemoryTypeIndex, inputImage);
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
  auto inputStagingMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, inputStagingMemoryTypeIndex,
                                   nullptr /*image*/, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, inputStagingBuffer,
                                   inputStagingMemory, 0 /*memoryOffset*/));

  printString("Populating staging buffer\n");
  // Populate staging memory
  VecType *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < numElems; ++i) {
    inputStagingData[i] =
        bindless_helpers::init_vector<DType, NChannels>(static_cast<DType>(
            CType == sycl::image_channel_type::unorm_int8 ? 255 : i));
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

  bool validated =
      run_sycl<decltype(input_mem_handle), NDims, DType, NChannels, CType,
               KernelName>(input_mem_handle, dims, localSize);

  // Cleanup
  vkDestroyBuffer(vk_device, inputStagingBuffer, nullptr);
  vkDestroyImage(vk_device, inputImage, nullptr);
  vkFreeMemory(vk_device, inputStagingMemory, nullptr);
  vkFreeMemory(vk_device, inputMemory, nullptr);

  return validated;
}

bool run_tests() {
  bool valid = true;
#ifdef TEST_L0_SUPPORTED_VK_FORMAT
  valid &=
      run_test<1, float, 1, sycl::image_channel_type::fp32,
               sycl::image_channel_order::r, class fp32_1d_c1>({1024}, {4}, 0);
  valid &=
      run_test<1, sycl::half, 2, sycl::image_channel_type::fp16,
               sycl::image_channel_order::rg, class fp16_1d_c2>({1024}, {4}, 0);
  valid &= run_test<1, sycl::half, 4, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rgba, class fp16_1d_c4>({1024},
                                                                       {4}, 0);
  valid &= run_test<1, uint8_t, 4, sycl::image_channel_type::unorm_int8,
                    sycl::image_channel_order::rgba, class unorm_int8_1d_c4>(
      {1024}, {4}, 0);

  valid &= run_test<2, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class fp32_2d_c1>({32, 32},
                                                                    {2, 2}, 0);
  valid &= run_test<2, sycl::half, 2, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rg, class fp16_2d_c2>({32, 32},
                                                                     {2, 2}, 0);
  valid &= run_test<2, sycl::half, 4, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rgba, class fp16_2d_c4>(
      {32, 32}, {2, 2}, 0);
  valid &= run_test<2, uint8_t, 4, sycl::image_channel_type::unorm_int8,
                    sycl::image_channel_order::rgba, class unorm_int8_2d_c4>(
      {32, 32}, {2, 2}, 0);

  valid &= run_test<3, float, 1, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::r, class fp32_3d_c1>(
      {64, 16, 2}, {2, 2, 2}, 0);
  valid &= run_test<3, sycl::half, 2, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rg, class fp16_3d_c2>(
      {64, 16, 2}, {2, 2, 2}, 0);
  valid &= run_test<3, sycl::half, 4, sycl::image_channel_type::fp16,
                    sycl::image_channel_order::rgba, class fp16_3d_c4>(
      {64, 16, 2}, {2, 2, 2}, 0);
  valid &= run_test<3, uint8_t, 4, sycl::image_channel_type::unorm_int8,
                    sycl::image_channel_order::rgba, class unorm_int8_3d_c4>(
      {64, 16, 2}, {2, 2, 2}, 0);
#else
  valid &= run_test<2, float, 4, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rgba, class float_2d>({16, 16},
                                                                     {2, 2}, 0);

  valid &= run_test<2, float, 2, sycl::image_channel_type::fp32,
                    sycl::image_channel_order::rg, class float_2d_large>(
      {1024, 1024}, {4, 2}, 0);

  valid &= run_test<3, char, 2, sycl::image_channel_type::signed_int8,
                    sycl::image_channel_order::rg, class int8_3d>({256, 16, 2},
                                                                  {2, 2, 2}, 0);

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
#endif
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
