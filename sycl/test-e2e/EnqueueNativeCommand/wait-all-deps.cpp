// RUN-IF: target-nvidia && cuda_dev_kit, %{build} -DUSE_CUDA_IN_TEST -o %t.cuda.out %cuda_options
// RUN-IF: target-nvidia && cuda_dev_kit, %{run} %t.cuda.out

// RUN-IF: target-amd && hip_dev_kit, %{build} -DUSE_HIP_IN_TEST -o %t.hip.out %hip_options
// RUN-IF: target-amd && hip_dev_kit, %{run} %t.hip.out

// REQUIRES: target-nvidia || target-amd

// Checks that native commands wait for all declared dependencies, including
// dependencies spanning compute commands and the transfer stream.

#include <cassert>
#include <future>

#ifdef USE_CUDA_IN_TEST
#include <cuda.h>
#endif

#ifdef USE_HIP_IN_TEST
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

#include <sycl/interop_handle.hpp>
#include <sycl/sycl.hpp>

#ifdef USE_CUDA_IN_TEST
static constexpr auto NativeBackend = sycl::backend::ext_oneapi_cuda;
using NativeStream = CUstream;
#elif defined(USE_HIP_IN_TEST)
static constexpr auto NativeBackend = sycl::backend::ext_oneapi_hip;
using NativeStream = hipStream_t;
#else
#error Either USE_CUDA_IN_TEST or USE_HIP_IN_TEST must be defined
#endif

static void launchHostWait(NativeStream stream, std::shared_future<void> *fut) {
  auto callback = [](void *userData) {
    static_cast<std::shared_future<void> *>(userData)->wait();
  };

#ifdef USE_CUDA_IN_TEST
  assert(cuLaunchHostFunc(stream, callback, fut) == CUDA_SUCCESS);
#else
  assert(hipLaunchHostFunc(stream, callback, fut) == hipSuccess);
#endif
}

static bool streamIsBusy(NativeStream stream) {
#ifdef USE_CUDA_IN_TEST
  return cuStreamQuery(stream) == CUDA_ERROR_NOT_READY;
#else
  return hipStreamQuery(stream) == hipErrorNotReady;
#endif
}

static void asyncMemcpyD2H(NativeStream stream, int *host_dst,
                           const int *dev_src) {
#ifdef USE_CUDA_IN_TEST
  assert(cuMemcpyDtoHAsync(host_dst, reinterpret_cast<CUdeviceptr>(dev_src),
                           sizeof(int), stream) == CUDA_SUCCESS);
#else
  assert(hipMemcpyDtoHAsync(host_dst, reinterpret_cast<hipDeviceptr_t>(dev_src),
                            sizeof(int), stream) == hipSuccess);
#endif
}

static void asyncMemcpyH2D(NativeStream stream, int *dev_dst,
                           const int *host_src) {
#ifdef USE_CUDA_IN_TEST
  assert(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dev_dst), host_src,
                           sizeof(int), stream) == CUDA_SUCCESS);
#else
  assert(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dev_dst), host_src,
                            sizeof(int), stream) == hipSuccess);
#endif
}

static void testMultipleComputeDependencies(sycl::queue &q) {
  int *src_42 = sycl::malloc_host<int>(1, q);
  *src_42 = 42;

  int *host_out = sycl::malloc_host<int>(1, q);
  *host_out = 0;

  int *dev_val = sycl::malloc_device<int>(1, q);

  q.memset(dev_val, 0, sizeof(int)).wait();

  std::promise<void> gate;
  std::shared_future<void> wait_for_host = gate.get_future().share();

  sycl::event e1 = q.single_task([]() {});

  sycl::event e2 = q.submit([&](sycl::handler &cgh) {
    cgh.ext_codeplay_enqueue_native_command([&](sycl::interop_handle ih) {
      auto stream = ih.get_native_queue<NativeBackend>();
      launchHostWait(stream, &wait_for_host);
      asyncMemcpyH2D(stream, dev_val, src_42);
    });
  });

  e1.wait();

  bool stream_blocked = false;
  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({e1, e2});
    cgh.ext_codeplay_enqueue_native_command([&](sycl::interop_handle ih) {
      auto stream = ih.get_native_queue<NativeBackend>();
      stream_blocked = streamIsBusy(stream);
      asyncMemcpyD2H(stream, host_out, dev_val);
    });
  });

  gate.set_value();
  q.wait();

  assert(stream_blocked && "Native stream should wait for e2");
  assert(*host_out == 42 && "Host output should be 42");

  sycl::free(src_42, q);
  sycl::free(host_out, q);
  sycl::free(dev_val, q);
}

static void testTransferStreamDependency(sycl::queue &q) {
  int *src_99 = sycl::malloc_host<int>(1, q);
  *src_99 = 99;

  int *host_out = sycl::malloc_host<int>(1, q);
  *host_out = 0;

  int *dev_val = sycl::malloc_device<int>(1, q);
  q.memset(dev_val, 0, sizeof(int)).wait();

  std::promise<void> gate;
  std::shared_future<void> wait_for_host = gate.get_future().share();

  sycl::event e_gate = q.submit([&](sycl::handler &cgh) {
    cgh.ext_codeplay_enqueue_native_command([&](sycl::interop_handle ih) {
      auto stream = ih.get_native_queue<NativeBackend>();
      launchHostWait(stream, &wait_for_host);
    });
  });

  sycl::event e_transfer = q.memcpy(dev_val, src_99, sizeof(int), {e_gate});

  bool stream_blocked = false;
  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({e_transfer});
    cgh.ext_codeplay_enqueue_native_command([&](sycl::interop_handle ih) {
      auto stream = ih.get_native_queue<NativeBackend>();
      stream_blocked = streamIsBusy(stream);
      asyncMemcpyD2H(stream, host_out, dev_val);
    });
  });

  gate.set_value();
  q.wait();

  assert(stream_blocked && "Native stream should wait for e_transfer");
  assert(*host_out == 99 && "Host output should be 99");

  sycl::free(src_99, q);
  sycl::free(host_out, q);
  sycl::free(dev_val, q);
}

int main() {
  sycl::queue q;

  testMultipleComputeDependencies(q);
  testTransferStreamDependency(q);

  return 0;
}
