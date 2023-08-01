#include "cuda_fixtures.h"

using urCudaContextGetNativeHandle = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaContextGetNativeHandle);

TEST_P(urCudaContextGetNativeHandle, Success) {
    ur_native_handle_t native_context = nullptr;
    ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_context));
    CUcontext cuda_context = reinterpret_cast<CUcontext>(native_context);

    unsigned int cudaVersion;
    ASSERT_SUCCESS_CUDA(cuCtxGetApiVersion(cuda_context, &cudaVersion));
}
