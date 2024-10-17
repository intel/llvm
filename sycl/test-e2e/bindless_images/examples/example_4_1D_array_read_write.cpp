// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

using VecType = sycl::vec<float, 4>;

int main() {
  // Set up device, queue, and context
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t width = 5;
  constexpr size_t array_size = 2;
  constexpr size_t N = width;
  std::vector<VecType> out(N * array_size);
  std::vector<float> expected(N * array_size);
  std::vector<float> outBuf(N);
  std::vector<VecType> dataIn1(N * array_size);
  std::vector<VecType> dataIn2(N * array_size);

  for (int i = 0; i < N * array_size; i++) {
    // Populate input data (to-be image arrays)
    dataIn1[i] = VecType(i);
    dataIn2[i] = VecType(2 * i);
  }

  // Populate expected output
  for (int i = 0; i < width; i++) {
    for (int l = 0; l < array_size; l++) {
      expected[l * N + i] = dataIn1[l * N + i][0] + dataIn2[l * N + i][0];
    }
  }

  // Extension: image descriptor -- number of layers
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width}, 4, sycl::image_channel_type::fp32,
      sycl::ext::oneapi::experimental::image_type::array, 1, array_size);

  // Extension: allocate image array memory on device
  sycl::ext::oneapi::experimental::image_mem arrayMem1(desc, dev, ctxt);
  sycl::ext::oneapi::experimental::image_mem arrayMem2(desc, dev, ctxt);
  sycl::ext::oneapi::experimental::image_mem outMem(desc, dev, ctxt);

  // Extension: copy over data to device
  q.ext_oneapi_copy(dataIn1.data(), arrayMem1.get_handle(), desc);
  q.ext_oneapi_copy(dataIn2.data(), arrayMem2.get_handle(), desc);
  q.wait_and_throw();

  // Extension: create a unsampled image handles to represent the image arrays
  sycl::ext::oneapi::experimental::unsampled_image_handle arrayHandle1 =
      sycl::ext::oneapi::experimental::create_image(arrayMem1, desc, dev, ctxt);
  sycl::ext::oneapi::experimental::unsampled_image_handle arrayHandle2 =
      sycl::ext::oneapi::experimental::create_image(arrayMem2, desc, dev, ctxt);
  sycl::ext::oneapi::experimental::unsampled_image_handle outHandle =
      sycl::ext::oneapi::experimental::create_image(outMem, desc, dev, ctxt);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class kernel>(N, [=](sycl::id<1> id) {
      float sum1 = 0;
      float sum2 = 0;

      // Extension: read image layers 0 and 1
      VecType px1 = sycl::ext::oneapi::experimental::fetch_image_array<VecType>(
          arrayHandle1, int(id[0]), 0);
      VecType px2 = sycl::ext::oneapi::experimental::fetch_image_array<VecType>(
          arrayHandle1, int(id[0]), 1);

      // Extension: read image layers 0 and 1
      VecType px3 = sycl::ext::oneapi::experimental::fetch_image_array<VecType>(
          arrayHandle2, int(id[0]), 0);
      VecType px4 = sycl::ext::oneapi::experimental::fetch_image_array<VecType>(
          arrayHandle2, int(id[0]), 1);

      sum1 = px1[0] + px3[0];
      sum2 = px2[0] + px4[0];

      // Extension: write to image layers with handle
      sycl::ext::oneapi::experimental::write_image_array<VecType>(
          outHandle, int(id[0]), 0, VecType(sum1));
      sycl::ext::oneapi::experimental::write_image_array<VecType>(
          outHandle, int(id[0]), 1, VecType(sum2));
    });
  });

  q.wait_and_throw();

  // Extension: copy data from device to host
  q.ext_oneapi_copy(outMem.get_handle(), out.data(), desc);
  q.wait_and_throw();

  // Extension: cleanup
  sycl::ext::oneapi::experimental::destroy_image_handle(arrayHandle1, dev,
                                                        ctxt);
  sycl::ext::oneapi::experimental::destroy_image_handle(arrayHandle2, dev,
                                                        ctxt);
  sycl::ext::oneapi::experimental::destroy_image_handle(outHandle, dev, ctxt);

  // Collect and validate output
  for (size_t i = 0; i < N * array_size; i++) {
    if (out[i][0] != expected[i]) {
      return 1;
    }
  }
  return 0;
}
