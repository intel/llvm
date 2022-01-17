// RUN: %clangxx -S -fsycl -fsycl-device-only %s -o

#include <CL/sycl.hpp>

int main() {
  const sycl::range<2> range{16, 16};
  const std::size_t size{range.size()};
  const sycl::float4 default_value{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<sycl::float4> src_data(size, default_value);
  sycl::image<2> src_image{src_data.data(), sycl::image_channel_order::rgba,
                           sycl::image_channel_type::fp32, range};
  sycl::image<2> dst_image{sycl::image_channel_order::rgba,
                           sycl::image_channel_type::fp32, range};
  sycl::queue queue;

  queue.submit([&](sycl::handler &handler) {
    const auto src_accessor =
        src_image.get_access<sycl::float4, sycl::access::mode::read>(handler);
    const auto dst_accessor =
        dst_image.get_access<sycl::float4, sycl::access::mode::write>(handler);
    handler.parallel_for(range, [=](const sycl::item<2> item) {
      const sycl::int2 coords{item[1], item[0]};
      auto pixel = src_accessor.read(coords);
      dst_accessor.write(coords, pixel);
    });
  });
  return 0;
}
