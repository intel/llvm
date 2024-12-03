// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

int main() {
  // Set up device, queue, and context
  sycl::device dev;
  sycl::queue q(dev);
  sycl::context ctxt = q.get_context();

  // declare image data
  constexpr size_t width = 16;
  unsigned int num_levels = 2;
  std::vector<float> dataIn1(width);
  std::vector<float> dataIn2(width / 2);
  std::vector<float> dataOut(width);
  std::vector<float> dataExpected(width);
  int j = 0;
  for (int i = 0; i < width; i++) {
    dataExpected[i] = static_cast<float>(i + (j + 10));
    if (i % 2)
      j++;
    dataIn1[i] = static_cast<float>(i);
    if (i < (width / 2))
      dataIn2[i] = static_cast<float>(i + 10);
  }

  // Image descriptor -- number of levels
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width}, 1, sycl::image_channel_type::fp32,
      sycl::ext::oneapi::experimental::image_type::mipmap, num_levels);

  // Allocate the mipmap
  sycl::ext::oneapi::experimental::image_mem mip_mem(desc, q);

  // Retrieve level 0
  sycl::ext::oneapi::experimental::image_mem_handle img_mem1 =
      mip_mem.get_mip_level_mem_handle(0);

  // Copy over data to level 0
  q.ext_oneapi_copy(dataIn1.data(), img_mem1, desc);

  // Copy over data to level 1
  q.ext_oneapi_copy(dataIn2.data(), mip_mem.get_mip_level_mem_handle(1),
                    desc.get_mip_level_desc(1));
  q.wait_and_throw();

  // Extended sampler object to take in mipmap attributes
  sycl::ext::oneapi::experimental::bindless_image_sampler samp(
      sycl::addressing_mode::mirrored_repeat,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::nearest, sycl::filtering_mode::nearest, 0.0f,
      static_cast<float>(num_levels), 8.0f);

  // Create a sampled image handle to represent the mipmap
  sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
      sycl::ext::oneapi::experimental::create_image(mip_mem, samp, desc, q);
  q.wait_and_throw();

  {
    sycl::buffer<float, 1> buf((float *)dataOut.data(), width);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<class image_addition>(width, [=](sycl::id<1> id) {
        float sum = 0;
        float x =
            (static_cast<float>(id[0]) + 0.5f) / static_cast<float>(width);
        // Read mipmap level 0 with anisotropic filtering
        // and level 1 with level filtering
        float px1 = sycl::ext::oneapi::experimental::sample_mipmap<float>(
            mipHandle, x, 0.0f, 0.0f);
        float px2 = sycl::ext::oneapi::experimental::sample_mipmap<float>(
            mipHandle, x, 1.0f);

        sum = px1 + px2;
        outAcc[id] = sum;
      });
    });
  }

  q.wait_and_throw();

  // Cleanup
  sycl::ext::oneapi::experimental::destroy_image_handle(mipHandle, q);

  // Validate that `dataOut` is correct
  for (size_t i = 0; i < width; i++) {
    if (dataOut[i] != dataExpected[i]) {
      return 1;
    }
  }
  return 0;
}
