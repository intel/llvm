// REQUIRES: cuda
//
// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out

// These tests use the add_native_events API to ensure that the SYCL RT can
// handle the events submitted to add_native_events within its runtime DAG.
//
// If manual_interop_sync is used then the user deals with async dependencies
// manually in the HT lambda through the get_native_events interface.

#include "host-task-native-events-cuda.hpp"
#include <cuda.h>
#include <sycl/sycl.hpp>

using T = unsigned; // We don't need to test lots of types, we just want a race
                    // condition
constexpr size_t bufSize = 1e6;
constexpr T pattern = 42;

// Tries to check for a race condition if the backend events are not added to
// the SYCL dag.
template <typename WaitOnType> void test1() {
  printf("Running test 2\n");
  sycl::queue q;
  std::vector<T> out(bufSize, 0);

  T *ptrHost = sycl::malloc_host<T>(bufSize, q); // malloc_host is necessary to
                                                 // make the memcpy as async as
                                                 // possible

  auto syclEvent = q.submit([&](sycl::handler &cgh) {
    cgh.host_task([&](sycl::interop_handle ih) {
      auto [stream, ev] = cudaSetCtxAndGetStreamAndEvent(ih);
      CUdeviceptr cuPtr;
      CUDA_CHECK(cuMemAlloc_v2(&cuPtr, bufSize * sizeof(T)));
      CUDA_CHECK(cuMemsetD32Async(cuPtr, pattern, bufSize, stream));
      CUDA_CHECK(
          cuMemcpyDtoHAsync(ptrHost, cuPtr, bufSize * sizeof(T), stream));

      CUDA_CHECK(cuEventRecord(ev, stream));

      ih.add_native_events<sycl::backend::ext_oneapi_cuda>({ev});
    });
  });
  waitHelper<WaitOnType>(syclEvent, q);
  checkResults(ptrHost, bufSize, pattern);
}

// Using host task event as a cgh.depends_on with USM
template <typename WaitOnType> void test2() {
  printf("Running test 3\n");
  using T = unsigned;

  sycl::queue q;
  std::vector<T> out(bufSize, 0);

  T *ptrHostA = sycl::malloc_host<T>(bufSize, q);
  T *ptrHostB = sycl::malloc_host<T>(bufSize, q);

  auto hostTaskEvent = q.submit([&](sycl::handler &cgh) {
    cgh.host_task([&](sycl::interop_handle ih) {
      auto [stream, ev] = cudaSetCtxAndGetStreamAndEvent(ih);
      CUdeviceptr cuPtr;
      CUDA_CHECK(cuMemAlloc_v2(&cuPtr, bufSize * sizeof(T)));

      CUDA_CHECK(cuMemsetD32Async(cuPtr, pattern, bufSize, stream));
      CUDA_CHECK(
          cuMemcpyDtoHAsync(ptrHostA, cuPtr, bufSize * sizeof(T), stream));

      CUDA_CHECK(cuEventRecord(ev, stream));

      ih.add_native_events<sycl::backend::ext_oneapi_cuda>({ev});
    });
  });

  auto syclEvent = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(hostTaskEvent);
    cgh.memcpy(ptrHostB, ptrHostA, bufSize * sizeof(T));
  });

  waitHelper<WaitOnType>(syclEvent, q);
  checkResults(ptrHostB, bufSize, pattern);
  printf("Tests passed\n");
}

// Using host task event with implicit DAG from buffer accessor model
template <typename WaitOnType> void test3() {
  printf("Running test 4\n");
  using T = unsigned;

  sycl::queue q;

  T *ptrHostIn = sycl::malloc_host<T>(bufSize, q);
  T *ptrHostOut = sycl::malloc_host<T>(bufSize, q);

  // Dummy buffer to create dependencies between commands. Use a host malloc
  // for host ptr to make sure the buffer has pinned memory
  sycl::buffer<T, 1> buf{
      ptrHostIn, bufSize, {sycl::property::buffer::use_host_ptr{}}};

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{buf, sycl::write_only};

    cgh.host_task([&](sycl::interop_handle ih) {
      // FIXME: this call fails
      auto accPtr = ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc);
      auto [stream, ev] = cudaSetCtxAndGetStreamAndEvent(ih);

      CUDA_CHECK(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(accPtr),
                                  pattern, bufSize, stream));
      CUDA_CHECK(cuEventRecord(ev, stream));

      ih.add_native_events<sycl::backend::ext_oneapi_cuda>({ev});
    });
  });

  {
    sycl::host_accessor hostAcc{buf};
    for (auto i = 0; i < bufSize; ++i)
      ptrHostOut[i] = hostAcc[i];
  }

  q.wait();
  checkResults(ptrHostOut, bufSize, pattern);
  printf("Tests passed\n");
}

int main() {
  test1<sycl::queue>();
  test1<sycl::event>();
  test2<sycl::queue>();
  test2<sycl::event>();
  // test3<sycl::queue>(); Fails with `SyclObject.impl && "every constructor
  //                       should create an impl"' failed.
  // test3<sycl::event>();
}
