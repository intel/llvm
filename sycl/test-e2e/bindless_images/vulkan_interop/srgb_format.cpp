// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_images_srgb
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

// Sampling an imported VK_FORMAT_R8G8B8A8_SRGB image must apply the hardware
// IEC 61966-2-1 decode, exactly as it does for images allocated through SYCL.
// The imported path builds its own ze_image_desc_t pNext chain, separate from
// image_mem allocation.

// Uncomment to print additional test information
// #define VERBOSE_PRINT
#include <sstream>

#include "../helpers/common.hpp"
#include "sycl_vulkan_setup.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

template <typename InteropMemHandleT>
std::vector<sycl::float4>
runSycl(const sycl::device &syclDevice, sycl::range<2> globalSize,
        sycl::range<2> localSize, InteropMemHandleT extMemInHandle,
        size_t imgSizeBytes) {
  sycl::queue syclQueue{syclDevice};

#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemInDesc{
      extMemInHandle, syclexp::external_mem_handle_type::win32_nt_handle,
      imgSizeBytes};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> extMemInDesc{
      extMemInHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSizeBytes};
#endif

  syclexp::external_mem externalMemIn =
      syclexp::import_external_memory(extMemInDesc, syclQueue);

  syclexp::image_descriptor imgDesc(
      globalSize, 4, sycl::image_channel_type::ext_oneapi_unorm_int8_srgb);

  syclexp::image_mem_handle imgMemIn =
      syclexp::map_external_image_memory(externalMemIn, imgDesc, syclQueue);

  syclexp::bindless_image_sampler sampler(
      sycl::addressing_mode::clamp_to_edge,
      sycl::coordinate_normalization_mode::unnormalized,
      sycl::filtering_mode::nearest);
  syclexp::sampled_image_handle imgIn =
      syclexp::create_image(imgMemIn, sampler, imgDesc, syclQueue);

  std::vector<sycl::float4> out(globalSize.size());
  try {
    {
      sycl::buffer<sycl::float4, 2> outBuf(out.data(), globalSize);
      syclQueue.submit([&](sycl::handler &cgh) {
        auto acc = outBuf.get_access<sycl::access_mode::write>(cgh);
        cgh.parallel_for<class TestSrgbInteropSample>(
            sycl::nd_range<2>{globalSize, localSize}, [=](sycl::nd_item<2> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              acc[sycl::id<2>(dim0, dim1)] =
                  syclexp::sample_image<sycl::float4>(imgIn,
                                                      sycl::float2(dim0, dim1));
            });
      });
      syclQueue.wait_and_throw();
    }

    syclexp::destroy_image_handle(imgIn, syclQueue);
    syclexp::unmap_external_image_memory(
        imgMemIn, syclexp::image_type::standard, syclQueue);
    syclexp::release_external_memory(externalMemIn, syclQueue);
  } catch (sycl::exception e) {
    log_error(std::string("\tKernel submission failed! ") + e.what());
    exit(-1);
  } catch (...) {
    log_error("\tKernel submission failed!");
    exit(-1);
  }

  return out;
}

bool runTest(VulkanContext &vkCtx, const sycl::device &syclDevice,
             sycl::range<2> dims, sycl::range<2> localSize) {
  const uint32_t imgWidth = static_cast<uint32_t>(dims[0]);
  const uint32_t imgHeight = static_cast<uint32_t>(dims[1]);

  const size_t imgSizeElems = imgWidth * imgHeight;
  const size_t imgSizeBytes = imgSizeElems * 4;
  const VkExtent3D imgExtent = {imgWidth, imgHeight, 1};

  // A mid-tone byte, far enough from the encoded value that a missing decode
  // cannot be mistaken for rounding error: 186/255 == 0.729 encoded, 0.491
  // linear.
  const uint8_t rawByte = 186;
  const float encoded = static_cast<float>(rawByte) / 255.0f;
  const float expectedLinear = bindless_helpers::srgb_to_linear(encoded);

  // Vulkan sRGB formats do not support VK_IMAGE_USAGE_STORAGE_BIT, so the
  // image is sampled rather than fetched.
  const VkImageUsageFlags imgUsage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (!isExportableImageFormatSupported(vkCtx, VK_FORMAT_R8G8B8A8_SRGB,
                                        VK_IMAGE_TYPE_2D,
                                        VK_IMAGE_TILING_OPTIMAL, imgUsage)) {
    log_error("Exportable VK_FORMAT_R8G8B8A8_SRGB image is not supported by "
              "this Vulkan driver");
    return false;
  }

  ImageResources inputImage = createExportableImage(
      vkCtx, imgExtent, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TYPE_2D,
      VK_IMAGE_TILING_OPTIMAL, imgUsage);
  const size_t importSizeBytes =
      std::max<size_t>(imgSizeBytes, inputImage.allocationSize);

  log_debug("Uploading image data");
  uploadImage(vkCtx, inputImage, 4, VK_NULL_HANDLE,
              [=](size_t, int) -> uint8_t { return rawByte; });

  log_debug("Getting memory interop handles");
#ifdef _WIN32
  auto imgMemIn = getMemHandle(vkCtx, inputImage.memory);
#else
  auto imgMemIn = getMemFd(vkCtx, inputImage.memory);
#endif

  log_debug("Calling into SYCL with interop memory handles");
  std::vector<sycl::float4> outputVec =
      runSycl(syclDevice, dims, localSize, imgMemIn, importSizeBytes);

  cleanupImageResources(vkCtx, inputImage);

  bool validated = true;
  for (size_t i = 0; i < imgSizeElems; ++i) {
    const sycl::float4 &got = outputVec[i];
    if (!checkValue(got.x(), expectedLinear) ||
        !checkValue(got.y(), expectedLinear) ||
        !checkValue(got.z(), expectedLinear)) {
      std::ostringstream oss;
      oss << "sRGB decode mismatch at " << i << "! actual == (" << got.x()
          << ", " << got.y() << ", " << got.z()
          << ") : expected == " << expectedLinear
          << " (undecoded value would be " << encoded << ")";
      log_error(oss.str());
      validated = false;
    }
    // Alpha must be passed through without decoding.
    if (!checkValue(got.w(), encoded)) {
      std::ostringstream oss;
      oss << "Alpha must not be decoded at " << i << "! actual == " << got.w()
          << " : expected == " << encoded;
      log_error(oss.str());
      validated = false;
    }
    if (!validated)
      break;
  }

  if (validated) {
    log_info("Results are correct!");
  }

  return validated;
}

int main() {
  try {
    sycl::device syclDevice;
    VulkanContext vkCtx = createSyclVulkanContext(syclDevice);
    VulkanContextGuard guard{vkCtx};

    auto testPassed = runTest(vkCtx, syclDevice, {32, 32}, {16, 16});

    if (testPassed) {
      log_info("Test passed!");
      return EXIT_SUCCESS;
    }

    log_error("Test failed");
    return EXIT_FAILURE;
  } catch (const std::exception &e) {
    log_error(std::string("Vulkan interop test failed: ") + e.what());
    return EXIT_FAILURE;
  }
}
