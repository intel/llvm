// REQUIRES: gpu && linux
// RUN: %{build} -o %t.out
// RUN: env SYCL_HOST_UNIFIED_MEMORY=1 %{run} %t.out

#include <stdio.h>
#include <stdlib.h>
#include <sycl/detail/core.hpp>
#include <unistd.h>

using namespace sycl;

static buffer<char, 1> *inBufP = nullptr;

int main(int argc, char *argv[]) {
  queue Q;
  auto BE =
      (bool)(Q.get_device()
                 .template get_info<sycl::info::device::opencl_c_version>()
                 .empty())
          ? "L0"
          : "OpenCL";
  device dev = Q.get_device();
  size_t max_compute_units = dev.get_info<info::device::max_compute_units>();
  printf("Device: %s max_compute_units %zu, Backend: %s\n",
         dev.get_info<info::device::name>().c_str(), max_compute_units, BE);

  size_t size = 32;
  void *p;
  posix_memalign(&p, 8, size);
  property::buffer::use_host_ptr prop_use_host_ptr;
  inBufP = new buffer<char, 1>((char *)p, size, prop_use_host_ptr);

  int iters = 3;
  for (int r = 0; r < iters; r++) {
    printf("Start iter %d\n", r);
    { host_accessor InhostAcc{*inBufP, write_only}; }
    Q.submit([&](handler &cgh) {
      accessor inAcc{*inBufP, cgh, read_write};
      cgh.single_task<class X5>([=]() { (void)inAcc; });
    });
    Q.wait();
    printf("End iter\n");
  }
  delete inBufP;
  free(p);

  return 0;
}
