// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip, cuda
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/sycl.hpp>

void run_test(sycl::coordinate_normalization_mode norm_mode,
              sycl::addressing_mode addr_mode,
              sycl::filtering_mode filter_mode) {
  std::cout << "Combination: " << (int)norm_mode << " - " << (int)addr_mode
            << " - " << (int)filter_mode << "\n";
  sycl::queue sycl_queue;

  const int height = 4;
  const int width = 4;
  auto image_range = sycl::range<2>(height, width);
  const int channels = 4;

  sycl::float2 coord{0.0, 0.0};
  float result{0.0};
  sycl::buffer<float> result_buff(&result, sycl::range<1>{1});

  std::vector<float> in_data(height * width * channels, 0.5f);
  sycl::image<2> image_in(in_data.data(), sycl::image_channel_order::rgba,
                          sycl::image_channel_type::fp32, image_range);

  sycl_queue.submit([&](sycl::handler &cgh) {
    sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                   sycl::access::target::image>
        in_acc(image_in, cgh);
    sycl::accessor<float, 1, sycl::access::mode::read_write> result_acc(
        result_buff, cgh);

    sycl::sampler smpl(norm_mode, addr_mode, filter_mode);

    cgh.single_task<class sampler_read>(
        [=]() { result_acc[0] = in_acc.read(coord, smpl)[3]; });
  });
  sycl_queue.wait();
  std::cout << "OKAY!\n";
}

int main() {
  for (auto n : {sycl::coordinate_normalization_mode::normalized,
                 sycl::coordinate_normalization_mode::unnormalized}) {
    for (auto a :
         {sycl::addressing_mode::none, sycl::addressing_mode::clamp_to_edge,
          sycl::addressing_mode::clamp, sycl::addressing_mode::repeat,
          sycl::addressing_mode::mirrored_repeat}) {
      for (auto f :
           {sycl::filtering_mode::linear, sycl::filtering_mode::nearest}) {
        run_test(n, a, f);
      }
    }
  }
  std::cout << "All tests executed, dropping.\n";
  return 0;
}
