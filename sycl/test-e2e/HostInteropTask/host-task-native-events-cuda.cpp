// REQUIRES: cuda
//
// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out

#include <cuda.h>
#include <sycl/sycl.hpp>

#define CUDA_CHECK(expr)                                                       \
  if (auto var = expr; var != CUDA_SUCCESS) {                                  \
    printf(#expr " failed, returned val %d\n", var);                           \
    throw;                                                                     \
  }

void cudaSetCtx(sycl::interop_handle &ih) {
  auto dev = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
  CUcontext ctx;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CUDA_CHECK(cuCtxSetCurrent(ctx));
}

// Check that the SYCL event that we submit with add_native_events can be
// retrieved later through get_native(syclEvent)
template <typename WaitOnType> void test1() {
  printf("Running test 1\n");
  sycl::queue q;

  std::atomic<CUevent>
      atomicEvent; // To share the event from the host task with the main thread

  auto syclEvent = q.submit([&](sycl::handler &cgh) {
    cgh.host_task([&](sycl::interop_handle ih) {
      cudaSetCtx(ih);
      CUevent Ev;
      cuEventCreate(&Ev, 0);
      cuEventRecord(Ev, 0);
      atomicEvent.store(Ev);
      ih.add_native_events<sycl::backend::ext_oneapi_cuda>({Ev});
    });
  });

  if constexpr (std::is_same_v<WaitOnType, sycl::queue>) {
    q.wait();
  } else if constexpr (std::is_same_v<WaitOnType, sycl::event>) {
    syclEvent.wait();
  }

  auto nativeEvents =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(syclEvent);
  // Check that the vec of native events contains the event we stored in the
  // atomic var
  assert(std::find(nativeEvents.begin(), nativeEvents.end(),
                   atomicEvent.load()) != nativeEvents.end());
}

// Tries to check for a race condition if the backend events are not added to
// the SYCL dag.
template <typename WaitOnType> void test2() {
  printf("Running test 2\n");
  using T = unsigned; // We don't need to test lots of types, we just want a
                      // race condition
  sycl::queue q;
  size_t bufSize = 1e6;
  T pattern = 42;
  std::vector<T> out(bufSize, 0);

  T *ptrHost = sycl::malloc_host<T>(bufSize, q);

  auto syclEvent = q.submit([&](sycl::handler &cgh) {
    cgh.host_task([&](sycl::interop_handle ih) {
      cudaSetCtx(ih);
      CUstream stream;
      CUDA_CHECK(cuStreamCreate(&stream, 0));
      CUevent ev;
      CUDA_CHECK(cuEventCreate(&ev, 0));
      CUdeviceptr cuPtr;
      CUDA_CHECK(cuMemAlloc_v2(&cuPtr, bufSize * sizeof(T)));
      CUDA_CHECK(cuMemsetD32Async(cuPtr, pattern, bufSize, stream));
      CUDA_CHECK(
          cuMemcpyDtoHAsync(ptrHost, cuPtr, bufSize * sizeof(T), stream));

      CUDA_CHECK(cuEventRecord(ev, stream));

      ih.add_native_events<sycl::backend::ext_oneapi_cuda>({ev});
    });
  });
  if constexpr (std::is_same_v<WaitOnType, sycl::queue>) {
    q.wait();
  } else if constexpr (std::is_same_v<WaitOnType, sycl::event>) {
    syclEvent.wait();
  }
  for (auto i = 0; i < bufSize; ++i) {
    if (ptrHost[i] != pattern) {
      printf("Wrong result at index: %d, have %d vs %d\n", i, out[i], pattern);
      throw;
    };
  }
}

int main() {
  test1<sycl::queue>();
  test1<sycl::event>();
  test2<sycl::queue>();
  test2<sycl::event>();
}
