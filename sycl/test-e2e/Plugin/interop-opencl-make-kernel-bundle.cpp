// REQUIRES: opencl, opencl_icd

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

#include <CL/cl.h>

#include <cassert>
#include <iostream>
#include <string>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <vector>

char const *source = R"OCL(kernel void do_nothing() {})OCL";

using namespace sycl;

int select_opencl(const sycl::device &dev) {
  if (dev.get_platform().get_backend() == backend::opencl) {
    return 1;
  }
  return -1;
}

int main() {
  auto q = queue{select_opencl};

  std::cerr << "Get native context." << std::endl;
  auto native_context = get_native<backend::opencl, context>(q.get_context());
  std::cerr << "Create native program." << std::endl;
  cl_program p =
      clCreateProgramWithSource(native_context, 1, &source, nullptr, nullptr);
  std::cerr << "Build native program." << std::endl;
  clBuildProgram(p, 0, nullptr, nullptr, nullptr, nullptr);

  std::cerr << "Make kernel bundle." << std::endl;
  auto bundle = make_kernel_bundle<backend::opencl, bundle_state::executable>(
      p, q.get_context());

  std::cerr << "Get native program." << std::endl;
  std::vector<cl_program> device_image =
      get_native<backend::opencl, bundle_state::executable>(bundle);
  assert(device_image.size() == 1);
  std::cerr << "Create native kernel." << std::endl;
  cl_kernel k = clCreateKernel(device_image.front(), "do_nothing", nullptr);

  std::cerr << "Make kernel." << std::endl;
  make_kernel<backend::opencl>(k, q.get_context());

  clReleaseProgram(p);
  clReleaseKernel(k);

  return 0;
}
