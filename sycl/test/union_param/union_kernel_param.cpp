// This test checks kernel execution with union kernel parameters.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

using namespace cl::sycl;

typedef float realw;
constexpr unsigned int NELEMS = 128;

typedef union dpct_type_54e08f {
#ifdef USE_OPENCL
  cl_mem ocl;
#endif
#ifdef USE_CUDA
  realw *cuda;
#endif
} gpu_realw_mem;

// creates real array on GPU
void gpuMalloc_realw(gpu_realw_mem *buffer, size_t size) {
  // allocates array on GPU
#ifdef USE_OPENCL
  cl_int errcode;
  buffer->ocl = clCreateBuffer(mocl.context, CL_MEM_READ_WRITE,
                               size * sizeof(realw), NULL, clck_(&errcode));
#endif
#ifdef USE_CUDA
  buffer->cuda = sycl::malloc_device<realw>(size, dpct::get_default_queue());
#endif
}

void call_some_dummy_kernel(realw *data, sycl::nd_item<3> it) {
  auto id = it.get_local_id(2);
  data[id] = id * 2.0f;
}

int main() {
  default_selector device_selector;
  queue q(device_selector);

  // Some device memory allocated outside function of interest
  gpu_realw_mem accelFromOut;
  gpuMalloc_realw(&accelFromOut, NELEMS * sizeof(realw));

  // Inside function where the device code is invoked
  gpu_realw_mem accel;

  accel = accelFromOut;

  auto grid = sycl::range<3>(1, 1, 1);
  auto threads = sycl::range<3>(NELEMS, 1, 1);

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        sdata_acc_ct1(sycl::range<1>(128 /*(BLOCKSIZE_TRANSFER)*/), cgh);

    auto dpct_global_range = grid * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          call_some_dummy_kernel(accel_cuda, item_ct1);
        });
  });

  realw *hostData = new realw[NELEMS];
  dpct::dpct_memcpy(hostData, accel.cuda, NELEMS * sizeof(realw),
                    dpct::device_to_host);

  auto isError = false;

  for (auto i = 0; i < NELEMS; i++) {
    // std::cout<< "Result = " <<  hostData[i]<<"  Expected = "<< 2.0f * i <<
    // "\n";
    if (hostData[i] != (2.0f * i)) {
      isError = true;
    }
  }
  if (isError)
    std::cout << " Error !!!"
              << "\n";
  else
    std::cout << " Results match !!!"
              << "\n";
}
