// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

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
    sycl::ext::oneapi::experimental::image_mem img_mem_0(ctxt, desc);

    // Extension: query for bindless image support
    bool bindless_support =
        dev.get_info<sycl::info::device::ext_oneapi_bindless_images_support>();
    bool usm_1d_support = dev.get_info<
        sycl::info::device::ext_oneapi_bindless_images_1d_usm_support>();
    bool usm_2d_support = dev.get_info<
        sycl::info::device::ext_oneapi_bindless_images_2d_usm_support>();
    bool usm_3d_support = dev.get_info<
        sycl::info::device::ext_oneapi_bindless_images_3d_usm_support>();

    std::cout << "ext_oneapi_bindless_images_support: " << bindless_support
              << "\next_oneapi_bindless_images_1d_usm_support: "
              << usm_1d_support
              << "\next_oneapi_bindless_images_2d_usm_support: "
              << usm_2d_support
              << "\next_oneapi_bindless_images_3d_usm_support: "
              << usm_3d_support << "\n";

    // Extension: get pitch alignment information from device
    // Make sure our pitch alignment queries work properly
    // These can be different depending on the device so we cannot test that the
    // values are correct
    // But we should at least see that the query itself works
    auto pitch_align =
        dev.get_info<sycl::info::device::ext_oneapi_image_pitch_align>();
    auto max_pitch =
        dev.get_info<sycl::info::device::ext_oneapi_max_image_linear_pitch>();
    auto max_width =
        dev.get_info<sycl::info::device::ext_oneapi_max_image_linear_width>();
    auto max_height =
        dev.get_info<sycl::info::device::ext_oneapi_max_image_linear_height>();

    std::cout << "ext_oneapi_image_pitch_align: " << pitch_align
              << "\next_oneapi_max_image_linear_pitch: " << max_pitch
              << "\next_oneapi_max_image_linear_width: " << max_width
              << "\next_oneapi_max_image_linear_height: " << max_height << "\n";

    // Extension: query for bindless image mipmaps support
    bool mipmap_support =
        dev.get_info<sycl::info::device::ext_oneapi_mipmap_support>();
    bool mipmap_anisotropy_support = dev.get_info<
        sycl::info::device::ext_oneapi_mipmap_anisotropy_support>();
    float mipmap_max_anisotropy =
        dev.get_info<sycl::info::device::ext_oneapi_mipmap_max_anisotropy>();
    bool mipmap_level_reference_support = dev.get_info<
        sycl::info::device::ext_oneapi_mipmap_level_reference_support>();

    std::cout << "ext_oneapi_mipmap_support: " << mipmap_support
              << "\next_oneapi_mipmap_anisotropy_support: "
              << mipmap_anisotropy_support
              << "\next_oneapi_mipmap_max_anisotropy: " << mipmap_max_anisotropy
              << "\next_oneapi_mipmap_level_reference_support: "
              << mipmap_level_reference_support << "\n";

    // Extension: query for bindless image interop support
    bool interop_memory_import_support = dev.get_info<
        sycl::info::device::ext_oneapi_interop_memory_import_support>();
    bool interop_memory_export_support = dev.get_info<
        sycl::info::device::ext_oneapi_interop_memory_export_support>();
    bool interop_semaphore_import_support = dev.get_info<
        sycl::info::device::ext_oneapi_interop_semaphore_import_support>();
    bool interop_semaphore_export_support = dev.get_info<
        sycl::info::device::ext_oneapi_interop_semaphore_export_support>();

    std::cout << "ext_oneapi_interop_memory_import_support: "
              << interop_memory_import_support
              << "\next_oneapi_interop_memory_export_support: "
              << interop_memory_export_support
              << "\next_oneapi_interop_semaphore_import_support: "
              << interop_semaphore_import_support
              << "\next_oneapi_interop_semaphore_export_support: "
              << interop_semaphore_export_support << "\n";

    auto rangeMem = img_mem_0.get_range();
    auto range = sycl::ext::oneapi::experimental::get_image_range(
        ctxt, img_mem_0.get_handle());
    if (rangeMem != range) {
      std::cout << "handle and mem object disagree on image dimensions!\n";
      validated = false;
    }
    if (range[0] == width) {
      std::cout << "width is correct!\n";
    } else {
      std::cout << "width is NOT correct!\n";
      validated = false;
    }
    if (range[1] == height) {
      std::cout << "height is correct!\n";
    } else {
      std::cout << "height is NOT correct!\n";
      validated = false;
    }
    if (range[2] == depth) {
      std::cout << "depth is correct!\n";
    } else {
      std::cout << "depth is NOT correct!\n";
      validated = false;
    }

    auto type = img_mem_0.get_type();
    if (type == sycl::ext::oneapi::experimental::image_type::standard) {
      std::cout << "image type is correct!\n";
    } else {
      std::cout << "image type is NOT correct!\n";
      validated = false;
    }

    auto ctypeMem = img_mem_0.get_channel_type();
    auto ctype = sycl::ext::oneapi::experimental::get_image_channel_type(
        ctxt, img_mem_0.get_handle());
    if (ctypeMem != ctype) {
      std::cout << "handle and mem object disagree on image channel type!\n";
      validated = false;
    }
    if (ctype == sycl::image_channel_type::signed_int32) {
      std::cout << "channel type is correct!\n";
    } else {
      std::cout << "channel type is NOT correct!\n";
      validated = false;
    }

    auto corder = img_mem_0.get_channel_order();
    if (corder == sycl::image_channel_order::r) {
      std::cout << "channel order is correct!\n";
    } else {
      std::cout << "channel order is NOT correct!\n";
      validated = false;
    }

    auto numchannelsMem = img_mem_0.get_num_channels();
    auto numchannels = sycl::ext::oneapi::experimental::get_image_num_channels(
        ctxt, img_mem_0.get_handle());
    if (numchannelsMem != numchannels) {
      std::cout << "handle and mem object disagree on number of channels!\n";
      validated = false;
    }
    if (numchannels == 1) {
      std::cout << "num channels is correct!\n";
    } else {
      std::cout << "num channels is NOT correct!\n";
      validated = false;
    }

    auto flagsMem = img_mem_0.get_flags();
    auto flags = sycl::ext::oneapi::experimental::get_image_flags(
        ctxt, img_mem_0.get_handle());
    if (flagsMem != flags) {
      std::cout << "handle and mem object disagree on image flags!\n";
      validated = false;
    }

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
  }

  return validated ? 0 : 1;
}
