#include "detail/event_impl.hpp"
#include "detail/queue_impl.hpp"
#include "sycl/platform.hpp"
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

inline ur_result_t after_urKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  constexpr char MockKernel[] = "TestKernel";
  if (*params.ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(MockKernel));
      std::memcpy(*params.ppPropValue, MockKernel, sizeof(MockKernel));
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(MockKernel);
  }
  return UR_RESULT_SUCCESS;
}

static thread_local size_t counter_urEnqueueKernelLaunch = 0;
inline ur_result_t redefined_urEnqueueKernelLaunch(void *pParams) {
  ++counter_urEnqueueKernelLaunch;
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static thread_local size_t counter_urUSMEnqueueMemcpy = 0;
inline ur_result_t redefined_urUSMEnqueueMemcpy(void *pParams) {
  ++counter_urUSMEnqueueMemcpy;
  auto params = *static_cast<ur_enqueue_usm_memcpy_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static thread_local size_t counter_urUSMEnqueueFill = 0;
inline ur_result_t redefined_urUSMEnqueueFill(void *pParams) {
  ++counter_urUSMEnqueueFill;
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static thread_local size_t counter_urUSMEnqueuePrefetch = 0;
inline ur_result_t redefined_urUSMEnqueuePrefetch(void *pParams) {
  ++counter_urUSMEnqueuePrefetch;
  auto params = *static_cast<ur_enqueue_usm_prefetch_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static thread_local size_t counter_urUSMEnqueueMemAdvise = 0;
inline ur_result_t redefined_urUSMEnqueueMemAdvise(void *pParams) {
  ++counter_urUSMEnqueueMemAdvise;
  auto params = *static_cast<ur_enqueue_usm_advise_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static thread_local size_t counter_urEnqueueEventsWaitWithBarrier = 0;
static thread_local std::chrono::time_point<std::chrono::steady_clock>
    timestamp_urEnqueueEventsWaitWithBarrier;
inline ur_result_t after_urEnqueueEventsWaitWithBarrier(void *pParams) {
  ++counter_urEnqueueEventsWaitWithBarrier;
  timestamp_urEnqueueEventsWaitWithBarrier = std::chrono::steady_clock::now();
  return UR_RESULT_SUCCESS;
}
