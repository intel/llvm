// REQUIRES: level_zero, opencl, level_zero_dev_kit
// RUN: %{build} %level_zero_options -lOpenCL -o %t.ze.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" %t.ze.out

#include <cstdlib>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

using namespace sycl;

const char *kernel_src = "kernel void test(global int *i){ i[0] = 34; }";

void handle_ze(ze_result_t status) {
  if (status != ZE_RESULT_SUCCESS)
    throw std::runtime_error("Level 0 error occurred");
}

void handle_cl(cl_int status) {
  if (status != CL_SUCCESS)
    throw std::runtime_error("OpenCL error occurred");
}

cl_device_id get_cl_device() {
  constexpr auto max_platforms = 16;
  cl_platform_id plat_ids[max_platforms];
  cl_device_id dev;
  cl_uint nplat, ndev;

  handle_cl(clGetPlatformIDs(max_platforms, plat_ids, &nplat));

  for (int pidx = 0; pidx < nplat && pidx < max_platforms; pidx++)
    if (clGetDeviceIDs(plat_ids[pidx], CL_DEVICE_TYPE_GPU, 1, &dev, &ndev) ==
        CL_SUCCESS)
      if (ndev > 0)
        return dev;

  throw std::runtime_error("No OpenCL GPU devices found.");
}

cl_program make_cl_program(const queue &Q) {
  cl_int status;

  auto device_cl = get_cl_device();
  auto context_cl =
      clCreateContext(nullptr, 1, &device_cl, nullptr, nullptr, &status);
  handle_cl(status);
  auto program_cl =
      clCreateProgramWithSource(context_cl, 1, &kernel_src, nullptr, &status);
  handle_cl(status);
  handle_cl(clBuildProgram(program_cl, 0, nullptr, "", nullptr, nullptr));

  return program_cl;
}

ze_module_handle_t convert_to_module(const queue &Q, cl_program program) {
  size_t binary_size = 0;
  handle_cl(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t),
                             &binary_size, NULL));

  auto binary = (unsigned char *)malloc(binary_size);
  if (binary == nullptr)
    handle_cl(CL_OUT_OF_HOST_MEMORY);

  handle_cl(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binary),
                             &binary, NULL));

  ze_module_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  desc.format = ZE_MODULE_FORMAT_NATIVE;
  desc.inputSize = binary_size;
  desc.pInputModule = binary;
  desc.pBuildFlags = "";
  desc.pConstants = nullptr;

  ze_module_handle_t module;
  auto device_ze = get_native<backend::ext_oneapi_level_zero>(Q.get_device());
  auto context_ze = get_native<backend::ext_oneapi_level_zero>(Q.get_context());

  handle_ze(zeModuleCreate(context_ze, device_ze, &desc, &module, nullptr));

  return module;
}

ze_kernel_handle_t create_kernel(ze_module_handle_t module,
                                 const char *kernel_name) {
  ze_kernel_handle_t kernel;
  ze_kernel_desc_t ze_kernel_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                  kernel_name};
  handle_ze(zeKernelCreate(module, &ze_kernel_desc, &kernel));
  return kernel;
}

int main() {
  device D{gpu_selector_v};

  try {
    auto tiles = D.create_sub_devices<
        info::partition_property::partition_by_affinity_domain>(
        info::partition_affinity_domain::numa);
    D = tiles[0];
  } catch (...) {
    fprintf(stderr, "Note: could not create subdevices\n");
  }

  queue Q{D};

  auto program = make_cl_program(Q);
  auto module = convert_to_module(Q, program);
  auto kernel = create_kernel(module, "test");

  auto P =
      make_kernel_bundle<backend::ext_oneapi_level_zero,
                         bundle_state::executable>({module}, Q.get_context());
  auto K =
      make_kernel<backend::ext_oneapi_level_zero>({P, kernel}, Q.get_context());

  buffer<int, 1> buf{1};
  {
    auto acc = buf.get_host_access();
    acc[0] = 1;
  }

  Q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::write>(cgh);
    cgh.set_args(acc);
    cgh.parallel_for(range<1>(1), K);
  });

  Q.wait_and_throw();
  {
    auto acc = buf.get_host_access();
    std::cout << acc[0] << std::endl;
  }
}
