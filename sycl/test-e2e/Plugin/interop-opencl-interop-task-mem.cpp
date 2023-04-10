// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %opencl_lib
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test for OpenCL interop_task.

#include <CL/opencl.h>
#include <iostream>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t SIZE = 16;

int main() {
  queue queue{};

  try {
    buffer<uint8_t, 1> buffer(SIZE);
    image<2> image(image_channel_order::rgba, image_channel_type::fp32,
                   {SIZE, SIZE});

    queue
        .submit([&](handler &cgh) {
          auto buffer_acc = buffer.get_access<access::mode::write>(cgh);
          auto image_acc = image.get_access<float4, access::mode::write>(cgh);
          cgh.interop_task([=](const interop_handler &ih) {
            cl_mem buffer_mem = ih.get_mem<backend::opencl>(buffer_acc);
            size_t size = 0;
            clGetMemObjectInfo(buffer_mem, CL_MEM_SIZE, sizeof(size),
                               (void *)&size, nullptr);
            assert(size == SIZE);

            cl_mem mem = ih.get_mem<backend::opencl>(image_acc);
            size_t width = 0;
            clGetImageInfo(mem, CL_IMAGE_WIDTH, sizeof(width), (void *)&width,
                           nullptr);
            assert(width == SIZE);
          });
        })
        .wait();
  } catch (exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  } catch (const char *msg) {
    std::cout << "Exception caught: " << msg << std::endl;
    return 1;
  }

  return 0;
}
