// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

void printString(std::string name) {
#ifdef VERBOSE_PRINT
  std::cout << name;
#endif
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  size_t height = 13;
  size_t width = 7;
  size_t depth = 11;

  bool validated = true;

  try {
    // Submit dummy kernel to let the runtime decide the backend (CUDA)
    // Without this, the default Level Zero backend is active
    q.submit([&](sycl::handler &cgh) { cgh.single_task([]() {}); });

    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, 1, sycl::image_channel_type::signed_int32);

    // Extension: returns the device pointer to the allocated memory
    // Input images memory
    sycl::ext::oneapi::experimental::image_mem imgMem(desc, dev, ctxt);

    // Extension: query for bindless image support -- device aspects
    bool bindlessSupport = dev.has(sycl::aspect::ext_oneapi_bindless_images);
    bool bindlessSharedUsmSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_shared_usm);
    bool usm1dSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_1d_usm);
    bool usm2dSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_2d_usm);

#ifdef VERBOSE_PRINT
    std::cout << "bindless_images_support: " << bindlessSupport
              << "\nbindless_images_shared_usm_support: "
              << bindlessSharedUsmSupport
              << "\nbindless_images_1d_usm_support: " << usm1dSupport
              << "\nbindless_images_2d_usm_support: " << usm2dSupport << "\n";
#endif

    // Extension: query for sampled image fetch capabilities
    bool sampledFetch1DUSMSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_1d_usm);
    bool sampledFetch2DUSMSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d_usm);
    bool sampledFetch3DUSMSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_3d_usm);
    bool sampledFetch1DSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_1d);
    bool sampledFetch2DSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d);
    bool sampledFetch3DSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_3d);

#ifdef VERBOSE_PRINT
    std::cout << "sampledFetch1DUSMSupport: " << sampledFetch1DUSMSupport
              << "\nsampledFetch2DUSMSupport: " << sampledFetch2DUSMSupport
              << "\nsampledFetch3DUSMSupport: " << sampledFetch3DUSMSupport
              << "\nsampledFetch1DSupport: " << sampledFetch1DSupport
              << "\nsampledFetch2DSupport: " << sampledFetch2DSupport
              << "\nsampledFetch3DSupport: " << sampledFetch3DSupport << "\n";
#endif

    // Extension: get pitch alignment information from device -- device info
    // Make sure our pitch alignment queries work properly
    // These can be different depending on the device so we cannot test that the
    // values are correct
    // But we should at least see that the query itself works
    auto pitchAlign = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::image_row_pitch_align>();
    auto maxPitch = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                     max_image_linear_row_pitch>();
    auto maxWidth = dev.get_info<sycl::ext::oneapi::experimental::info::device::
                                     max_image_linear_width>();
    auto maxheight = dev.get_info<sycl::ext::oneapi::experimental::info::
                                      device::max_image_linear_height>();

#ifdef VERBOSE_PRINT
    std::cout << "image_row_pitch_align: " << pitchAlign
              << "\nmax_image_linear_row_pitch: " << maxPitch
              << "\nmax_image_linear_width: " << maxWidth
              << "\nmax_image_linear_height: " << maxheight << "\n";
#endif

    // Extension: query for bindless image mipmaps support -- aspects & info
    bool mipmapSupport = dev.has(sycl::aspect::ext_oneapi_mipmap);
    bool mipmapAnisotropySupport =
        dev.has(sycl::aspect::ext_oneapi_mipmap_anisotropy);
    float mipmapMaxAnisotropy = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::mipmap_max_anisotropy>();
    bool mipmapLevelReferenceSupport =
        dev.has(sycl::aspect::ext_oneapi_mipmap_level_reference);

#ifdef VERBOSE_PRINT
    std::cout << "mipmapSupport: " << mipmapSupport
              << "\nmipmapAnisotropySupport: " << mipmapAnisotropySupport
              << "\nmipmapMaxAnisotropy: " << mipmapMaxAnisotropy
              << "\nmipmapLevelReferenceSupport: "
              << mipmapLevelReferenceSupport << "\n";
#endif

    // Extension: query for bindless image cubemaps support -- aspects.
    bool cubemapSupport = dev.has(sycl::aspect::ext_oneapi_cubemap);
    bool cubemapSeamlessFilterSupport =
        dev.has(sycl::aspect::ext_oneapi_cubemap_seamless_filtering);

#ifdef VERBOSE_PRINT
    std::cout << "cubemapSupport: " << cubemapSupport
              << "\ncubemapSeamlessFilterSupport: "
              << cubemapSeamlessFilterSupport << "\n";
#endif

    // Extension: query for bindless image interop support -- device aspects
    bool interopMemoryImportSupport =
        dev.has(sycl::aspect::ext_oneapi_interop_memory_import);
    bool interopMemoryExportSupport =
        dev.has(sycl::aspect::ext_oneapi_interop_memory_export);
    bool interopSemaphoreImportSupport =
        dev.has(sycl::aspect::ext_oneapi_interop_semaphore_import);
    bool interopSemaphoreExportSupport =
        dev.has(sycl::aspect::ext_oneapi_interop_semaphore_export);

#ifdef VERBOSE_PRINT
    std::cout << "interopMemoryImportSupport: " << interopMemoryImportSupport
              << "\ninteropMemoryExportSupport: " << interopMemoryExportSupport
              << "\ninteropSemaphoreImportSupport: "
              << interopSemaphoreImportSupport
              << "\ninteropSemaphoreExportSupport: "
              << interopSemaphoreExportSupport << "\n";
#endif

    auto rangeMem = imgMem.get_range();
    auto range = sycl::ext::oneapi::experimental::get_image_range(
        imgMem.get_handle(), dev, ctxt);
    if (rangeMem != range) {
      printString("handle and mem object disagree on image dimensions!\n");
      validated = false;
    }
    if (range[0] == width) {
      printString("width is correct!\n");
    } else {
      printString("width is NOT correct!\n");
      validated = false;
    }
    if (range[1] == height) {
      printString("height is correct!\n");
    } else {
      printString("height is NOT correct!\n");
      validated = false;
    }
    if (range[2] == depth) {
      printString("depth is correct!\n");
    } else {
      printString("depth is NOT correct!\n");
      validated = false;
    }

    auto type = imgMem.get_type();
    if (type == sycl::ext::oneapi::experimental::image_type::standard) {
      printString("image type is correct!\n");
    } else {
      printString("image type is NOT correct!\n");
      validated = false;
    }

    auto ctypeMem = imgMem.get_channel_type();
    auto ctype = sycl::ext::oneapi::experimental::get_image_channel_type(
        imgMem.get_handle(), dev, ctxt);
    if (ctypeMem != ctype) {
      printString("handle and mem object disagree on image channel type!\n");
      validated = false;
    }
    if (ctype == sycl::image_channel_type::signed_int32) {
      printString("channel type is correct!\n");
    } else {
      printString("channel type is NOT correct!\n");
      validated = false;
    }

    auto numchannelsMem = imgMem.get_num_channels();
    auto numchannels = sycl::ext::oneapi::experimental::get_image_num_channels(
        imgMem.get_handle(), dev, ctxt);
    if (numchannelsMem != numchannels) {
      printString("handle and mem object disagree on number of channels!\n");
      validated = false;
    }
    if (numchannels == 1) {
      printString("num channels is correct!\n");
    } else {
      printString("num channels is NOT correct!\n");
      validated = false;
    }

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  if (validated) {
    std::cout << "Test Passed!\n";
    return 0;
  }

  std::cout << "Test Failed!" << std::endl;
  return 3;
}
