// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

int main() {

  sycl::device dev;

  bool validated = true;

  try {
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
    bool sampledFetch1DSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_1d);
    bool sampledFetch2DSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d);
    bool sampledFetch3DSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_3d);

#ifdef VERBOSE_PRINT
    std::cout << "sampledFetch1DUSMSupport: " << sampledFetch1DUSMSupport
              << "\nsampledFetch2DUSMSupport: " << sampledFetch2DUSMSupport
              << "\nsampledFetch1DSupport: " << sampledFetch1DSupport
              << "\nsampledFetch2DSupport: " << sampledFetch2DSupport
              << "\nsampledFetch3DSupport: " << sampledFetch3DSupport << "\n";
#endif
    // Extension: query for bindless image mipmaps support -- aspects & info
    bool mipmapSupport = dev.has(sycl::aspect::ext_oneapi_mipmap);
    bool mipmapAnisotropySupport =
        dev.has(sycl::aspect::ext_oneapi_mipmap_anisotropy);
    float mipmapMaxAnisotropy;
    if (mipmapAnisotropySupport) {
      mipmapMaxAnisotropy = dev.get_info<sycl::ext::oneapi::experimental::info::
                                             device::mipmap_max_anisotropy>();
    }
    bool mipmapLevelReferenceSupport =
        dev.has(sycl::aspect::ext_oneapi_mipmap_level_reference);

#ifdef VERBOSE_PRINT
    std::cout << "mipmapSupport: " << mipmapSupport
              << "\nmipmapAnisotropySupport: " << mipmapAnisotropySupport
              << "\nmipmapLevelReferenceSupport: "
              << mipmapLevelReferenceSupport << "\n";
    if (mipmapAnisotropySupport) {
      std::cout << "mipmapMaxAnisotropy: " << mipmapMaxAnisotropy << "\n";
    }
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
    bool externalMemoryImportSupport =
        dev.has(sycl::aspect::ext_oneapi_external_memory_import);
    bool externalSemaphoreImportSupport =
        dev.has(sycl::aspect::ext_oneapi_external_semaphore_import);

#ifdef VERBOSE_PRINT
    std::cout << "externalMemoryImportSupport: " << externalMemoryImportSupport
              << "\nexternalSemaphoreImportSupport: "
              << externalSemaphoreImportSupport << "\n";
#endif

    // Extension: query for bindless image array support - device aspect
    bool imageArraySupport = dev.has(sycl::aspect::ext_oneapi_image_array);

#ifdef VERBOSE_PRINT
    std::cout << "imageArraySupport: " << imageArraySupport << "\n";
#endif

    // Extension: query for bindless image unique addressing support - device
    // aspect
    bool uniqueAddrSupport =
        dev.has(sycl::aspect::ext_oneapi_unique_addressing_per_dim);

#ifdef VERBOSE_PRINT
    std::cout << "uniqueAddrSupport: " << uniqueAddrSupport << "\n";
#endif

    // Extension: query for usm sample support - device aspect
    bool usm1DSampleSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_sample_1d_usm);
    bool usm2DSampleSupport =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_sample_2d_usm);

#ifdef VERBOSE_PRINT
    std::cout << "usm1DSampleSupport: " << usm1DSampleSupport << "\n";
    std::cout << "usm2DSampleSupport: " << usm2DSampleSupport << "\n";
#endif

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
