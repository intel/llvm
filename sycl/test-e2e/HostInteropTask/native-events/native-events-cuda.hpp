#include <cuda.h>
#include <sycl/sycl.hpp>

#define CUDA_CHECK(expr)                                                       \
  if (auto var = expr; var != CUDA_SUCCESS) {                                  \
    printf(#expr " failed, returned val %d\n", var);                           \
    throw;                                                                     \
  }

// Set up the context in host task and get a created event and stream
std::pair<CUstream, CUevent>
cudaSetCtxAndGetStreamAndEvent(sycl::interop_handle &ih) {
  auto dev = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
  CUcontext ctx;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CUDA_CHECK(cuCtxSetCurrent(ctx));
  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, 0));
  CUevent ev;
  CUDA_CHECK(cuEventCreate(&ev, 0));
  return {stream, ev};
}

// To test that waiting on events is equivalent to wait on queues
template <typename WaitOnType>
void waitHelper(sycl::event &event, sycl::queue &q) {
  if constexpr (std::is_same_v<WaitOnType, sycl::queue>) {
    q.wait();
  } else if constexpr (std::is_same_v<WaitOnType, sycl::event>) {
    event.wait();
  }
}

template <typename T> void checkResults(T *ptr, size_t bufSize, T pattern) {
  for (auto i = 0; i < bufSize; ++i) {
    if (ptr[i] != pattern) {
      printf("Wrong result at index: %d, have %d vs %d\n", i, ptr[i], pattern);
      throw;
    }
  }
}
