// REQUIRES: cuda && cuda_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %cuda_options %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/sycl.hpp>

#include <cuda.h>

#include <assert.h>

void cuda_check(CUresult error) { assert(error == CUDA_SUCCESS); }

template <typename refT, typename T> void check_type(T var) {
  static_assert(std::is_same_v<decltype(var), refT>);
}

#define CUDA_CHECK(error) cuda_check(error)

bool check_queue(sycl::queue &Q) {
  constexpr size_t vec_size = 5;
  double A_Data[vec_size] = {4.0};
  double B_Data[vec_size] = {-3.0};
  double C_Data[vec_size] = {0.0};

  sycl::buffer<double, 1> A_buff(A_Data, sycl::range<1>(vec_size));
  sycl::buffer<double, 1> B_buff(B_Data, sycl::range<1>(vec_size));
  sycl::buffer<double, 1> C_buff(C_Data, sycl::range<1>(vec_size));

  Q.submit([&](sycl::handler &cgh) {
     auto A_acc = A_buff.get_access<sycl::access::mode::read>(cgh);
     auto B_acc = B_buff.get_access<sycl::access::mode::read>(cgh);
     auto C_acc = C_buff.get_access<sycl::access::mode::write>(cgh);
     cgh.parallel_for(sycl::range<1>{vec_size}, [=](sycl::id<1> idx) {
       C_acc[idx] = A_acc[idx] + B_acc[idx];
     });
   }).wait();

  sycl::host_accessor C_acc(C_buff);
  return C_acc[0] == 1;
}

int main() {
  sycl::queue Q;

  CUcontext Q_cu_ctx;
  auto native_queue = sycl::get_native<sycl::backend::ext_oneapi_cuda>(Q);
  check_type<CUstream>(native_queue);
  CUDA_CHECK(cuStreamGetCtx(native_queue, &Q_cu_ctx));
  auto Q_sycl_ctx =
      sycl::make_context<sycl::backend::ext_oneapi_cuda>(Q_cu_ctx);

  // Create sycl queue with queue construct from Q's native types and submit
  // some work
  {
    sycl::queue new_Q(Q_sycl_ctx, sycl::default_selector());
    assert(check_queue(new_Q));
  }

  // Check Q still works
  assert(check_queue(Q));

  // Get native cuda device
  CUdevice cu_dev;
  CUDA_CHECK(cuDeviceGet(&cu_dev, 0));
  auto sycl_dev = sycl::make_device<sycl::backend::ext_oneapi_cuda>(cu_dev);
  auto native_dev = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_dev);

  check_type<sycl::device>(sycl_dev);
  check_type<CUdevice>(native_dev);
  assert(native_dev == cu_dev);

  // Create sycl queue with new device and submit some work
  {
    sycl::queue new_Q(sycl_dev);
    assert(check_queue(new_Q));
  }

  // Create new context
  CUcontext curr_ctx, cu_ctx;
  CUDA_CHECK(cuCtxGetCurrent(&curr_ctx));
  CUDA_CHECK(cuCtxCreate(&cu_ctx, CU_CTX_MAP_HOST, cu_dev));
  CUDA_CHECK(cuCtxSetCurrent(curr_ctx));

  auto sycl_ctx = sycl::make_context<sycl::backend::ext_oneapi_cuda>(cu_ctx);
  auto native_ctx = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_ctx);

  check_type<sycl::context>(sycl_ctx);
  check_type<std::vector<CUcontext>>(native_ctx);

  // Create sycl queue with new queue and submit some work
  {
    sycl::queue new_Q(sycl_ctx, sycl::default_selector());
    assert(check_queue(new_Q));
  }

  // Create new event
  CUevent cu_event;

  CUDA_CHECK(cuCtxSetCurrent(cu_ctx));
  CUDA_CHECK(cuEventCreate(&cu_event, CU_EVENT_DEFAULT));

  auto sycl_event =
      sycl::make_event<sycl::backend::ext_oneapi_cuda>(cu_event, sycl_ctx);
  auto native_event =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_event);

  check_type<sycl::event>(sycl_event);
  check_type<CUevent>(native_event);

  // Check sycl queue with sycl_ctx still works
  {
    sycl::queue new_Q(sycl_ctx, sycl::default_selector());
    assert(check_queue(new_Q));
  }

  // Check has_native_event
  {
    auto e = Q.submit([&](sycl::handler &cgh) { cgh.single_task([] {}); });
    assert(sycl::ext::oneapi::cuda::has_native_event(e));
  }

  {
    auto e = Q.submit([&](sycl::handler &cgh) { cgh.host_task([] {}); });
    assert(!sycl::ext::oneapi::cuda::has_native_event(e));
  }

  // Create new queue
  CUstream cu_queue;
  CUDA_CHECK(cuCtxSetCurrent(cu_ctx));
  CUDA_CHECK(cuStreamCreate(&cu_queue, CU_STREAM_DEFAULT));

  auto sycl_queue =
      sycl::make_queue<sycl::backend::ext_oneapi_cuda>(cu_queue, sycl_ctx);
  native_queue = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);

  check_type<sycl::queue>(sycl_queue);
  check_type<CUstream>(native_queue);

  // Submit some work to new queue
  assert(check_queue(sycl_queue));

  // Create new queue with Q's native type and submit some work
  {
    CUstream Q_native_stream =
        sycl::get_native<sycl::backend::ext_oneapi_cuda>(Q);
    sycl::queue new_Q = sycl::make_queue<sycl::backend::ext_oneapi_cuda>(
        Q_native_stream, Q_sycl_ctx);
    assert(check_queue(new_Q));
  }

  // Check Q still works
  assert(check_queue(Q));
}
