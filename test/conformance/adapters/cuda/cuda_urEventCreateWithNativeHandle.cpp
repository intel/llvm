#include "cuda_fixtures.h"

using urCudaEventCreateWithNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaEventCreateWithNativeHandleTest);

TEST_P(urCudaEventCreateWithNativeHandleTest, Success) {

    CUevent cuda_event;
    ASSERT_SUCCESS_CUDA(cuEventCreate(&cuda_event, CU_EVENT_DEFAULT));

    ur_native_handle_t native_event =
        reinterpret_cast<ur_native_handle_t>(cuda_event);

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEventCreateWithNativeHandle(native_event, context, nullptr, &event));

    ASSERT_SUCCESS(urEventRelease(event));
}
