#include "detail/event_impl.hpp"
#include "detail/queue_impl.hpp"
#include "sycl/platform.hpp"
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

namespace FreeFunctionEventsHelpers {

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

template <bool NativeHostTasksSupported>
inline ur_result_t redefined_urDeviceGetInfo(void *pParams) {
  auto &Params = *reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP) {
    if (*Params.ppPropValue)
      *reinterpret_cast<ur_bool_t *>(*Params.ppPropValue) =
          NativeHostTasksSupported;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_bool_t);
  }
  return UR_RESULT_SUCCESS;
}

static size_t counter_urEnqueueKernelLaunchWithArgsExp = 0;
inline ur_result_t redefined_urEnqueueKernelLaunchWithArgsExp(void *pParams) {
  ++counter_urEnqueueKernelLaunchWithArgsExp;
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static size_t counter_urEnqueueKernelLaunchWithEvent = 0;
inline ur_result_t redefined_urEnqueueKernelLaunchWithEvent(void *pParams) {
  ++counter_urEnqueueKernelLaunchWithEvent;
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  EXPECT_NE(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static size_t counter_urUSMEnqueueMemcpy = 0;
inline ur_result_t redefined_urUSMEnqueueMemcpy(void *pParams) {
  ++counter_urUSMEnqueueMemcpy;
  auto params = *static_cast<ur_enqueue_usm_memcpy_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static size_t counter_urUSMEnqueueFill = 0;
inline ur_result_t redefined_urUSMEnqueueFill(void *pParams) {
  ++counter_urUSMEnqueueFill;
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static size_t counter_urUSMEnqueuePrefetch = 0;
inline ur_result_t redefined_urUSMEnqueuePrefetch(void *pParams) {
  ++counter_urUSMEnqueuePrefetch;
  auto params = *static_cast<ur_enqueue_usm_prefetch_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static size_t counter_urUSMEnqueueMemAdvise = 0;
inline ur_result_t redefined_urUSMEnqueueMemAdvise(void *pParams) {
  ++counter_urUSMEnqueueMemAdvise;
  auto params = *static_cast<ur_enqueue_usm_advise_params_t *>(pParams);
  EXPECT_EQ(*params.pphEvent, nullptr);
  return UR_RESULT_SUCCESS;
}

static size_t counter_urEnqueueEventsWaitWithBarrier = 0;
static std::chrono::time_point<std::chrono::steady_clock>
    timestamp_urEnqueueEventsWaitWithBarrier;
inline ur_result_t after_urEnqueueEventsWaitWithBarrier(void *pParams) {
  ++counter_urEnqueueEventsWaitWithBarrier;
  timestamp_urEnqueueEventsWaitWithBarrier = std::chrono::steady_clock::now();
  return UR_RESULT_SUCCESS;
}

static size_t counter_urEnqueueHostTaskExp = 0;
inline ur_result_t redefined_urEnqueueHostTaskExp(void *pParams) {
  ++counter_urEnqueueHostTaskExp;
  return UR_RESULT_SUCCESS;
}

} // namespace FreeFunctionEventsHelpers
