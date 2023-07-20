// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

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
        {width, height, depth}, sycl::image_channel_order::r,
        sycl::image_channel_type::signed_int32);

    // Extension: returns the device pointer to the allocated memory
    // Input images memory
    sycl::ext::oneapi::experimental::image_mem img_mem_0(desc, dev, ctxt);

    // Extension: query for bindless image support -- device aspects
    bool bindless_support = dev.has(sycl::aspect::ext_oneapi_bindless_images);
    bool bindless_shared_usm_support =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_shared_usm);
    bool usm_1d_support =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_1d_usm);
    bool usm_2d_support =
        dev.has(sycl::aspect::ext_oneapi_bindless_images_2d_usm);

#ifdef VERBOSE_PRINT
    std::cout << "bindless_images_support: " << bindless_support
              << "\nbindless_images_shared_usm_support: "
              << bindless_shared_usm_support
              << "\nbindless_images_1d_usm_support: " << usm_1d_support
              << "\nbindless_images_2d_usm_support: " << usm_2d_support << "\n";
#endif

    // Extension: get pitch alignment information from device -- device info
    // Make sure our pitch alignment queries work properly
    // These can be different depending on the device so we cannot test that the
    // values are correct
    // But we should at least see that the query itself works
    auto pitch_align = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::image_pitch_align>();
    auto max_pitch = dev.get_info<sycl::ext::oneapi::experimental::info::
                                      device::max_image_linear_pitch>();
    auto max_width = dev.get_info<sycl::ext::oneapi::experimental::info::
                                      device::max_image_linear_width>();
    auto max_height = dev.get_info<sycl::ext::oneapi::experimental::info::
                                       device::max_image_linear_height>();

#ifdef VERBOSE_PRINT
    std::cout << "image_pitch_align: " << pitch_align
              << "\nmax_image_linear_pitch: " << max_pitch
              << "\nmax_image_linear_width: " << max_width
              << "\nmax_image_linear_height: " << max_height << "\n";
#endif

    // Extension: query for bindless image mipmaps support -- aspects & info
    bool mipmap_support = dev.has(sycl::aspect::ext_oneapi_mipmap);
    bool mipmap_anisotropy_support =
        dev.has(sycl::aspect::ext_oneapi_mipmap_anisotropy);
    float mipmap_max_anisotropy = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::mipmap_max_anisotropy>();
    bool mipmap_level_reference_support =
        dev.has(sycl::aspect::ext_oneapi_mipmap_level_reference);

#ifdef VERBOSE_PRINT
    std::cout << "mipmap_support: " << mipmap_support
              << "\nmipmap_anisotropy_support: " << mipmap_anisotropy_support
              << "\nmipmap_max_anisotropy: " << mipmap_max_anisotropy
              << "\nmipmap_level_reference_support: "
              << mipmap_level_reference_support << "\n";
#endif

    // Extension: query for bindless image interop support -- device aspects
    bool interop_memory_import_support =
        dev.has(sycl::aspect::ext_oneapi_interop_memory_import);
    bool interop_memory_export_support =
        dev.has(sycl::aspect::ext_oneapi_interop_memory_export);
    bool interop_semaphore_import_support =
        dev.has(sycl::aspect::ext_oneapi_interop_semaphore_import);
    bool interop_semaphore_export_support =
        dev.has(sycl::aspect::ext_oneapi_interop_semaphore_export);

#ifdef VERBOSE_PRINT
    std::cout << "interop_memory_import_support: "
              << interop_memory_import_support
              << "\ninterop_memory_export_support: "
              << interop_memory_export_support
              << "\ninterop_semaphore_import_support: "
              << interop_semaphore_import_support
              << "\ninterop_semaphore_export_support: "
              << interop_semaphore_export_support << "\n";
#endif

    auto rangeMem = img_mem_0.get_range();
    auto range = sycl::ext::oneapi::experimental::get_image_range(
        img_mem_0.get_handle(), dev, ctxt);
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

    auto type = img_mem_0.get_type();
    if (type == sycl::ext::oneapi::experimental::image_type::standard) {
      printString("image type is correct!\n");
    } else {
      printString("image type is NOT correct!\n");
      validated = false;
    }

    auto ctypeMem = img_mem_0.get_channel_type();
    auto ctype = sycl::ext::oneapi::experimental::get_image_channel_type(
        img_mem_0.get_handle(), dev, ctxt);
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

    auto corder = img_mem_0.get_channel_order();
    if (corder == sycl::image_channel_order::r) {
      printString("channel order is correct!\n");
    } else {
      printString("channel order is NOT correct!\n");
      validated = false;
    }

    auto numchannelsMem = img_mem_0.get_num_channels();
    auto numchannels = sycl::ext::oneapi::experimental::get_image_num_channels(
        img_mem_0.get_handle(), dev, ctxt);
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
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
  }

  if (validated) {
    std::cout << "Test Passed!\n";
    return 0;
  }

  std::cout << "Test Failed!\n";
  return 1;
}
