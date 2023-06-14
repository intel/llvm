#include <iostream>
#include <vector>

#include "helpers.h"
#include "ur_api.h"

void ur_check(const ur_result_t r) {
    if (r != UR_RESULT_SUCCESS) {
        urTearDown(nullptr);
        throw std::runtime_error("Unified runtime error: " + std::to_string(r));
    }
}

std::vector<ur_platform_handle_t> get_platforms() {
    uint32_t platformCount = 0;
    ur_check(urPlatformGet(1, nullptr, &platformCount));

    if (!platformCount) {
        throw std::runtime_error("No platforms available.");
    }

    std::vector<ur_platform_handle_t> platforms(platformCount);
    ur_check(urPlatformGet(platformCount, platforms.data(), nullptr));
    return platforms;
}

std::vector<ur_device_handle_t> get_gpus(ur_platform_handle_t p) {
    uint32_t deviceCount = 0;
    ur_check(urDeviceGet(p, UR_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount));

    if (!deviceCount) {
        throw std::runtime_error("No GPUs available.");
    }

    std::vector<ur_device_handle_t> devices(deviceCount);
    ur_check(urDeviceGet(p, UR_DEVICE_TYPE_GPU, deviceCount, devices.data(),
                         nullptr));
    return devices;
}

template <typename T, size_t N> struct alignas(4096) AlignedArray {
    T data[N];
};

int main() {
    ur_check(urInit(UR_DEVICE_INIT_FLAG_GPU));

    auto platforms = get_platforms();
    auto gpus = get_gpus(platforms.front());
    auto spv = generate_plus_one_spv();

    constexpr int a_size = 32;
    AlignedArray<int, a_size> a, b;
    for (auto i = 0; i < a_size; ++i) {
        a.data[i] = a_size - i;
        b.data[i] = i;
    }

    auto current_device = gpus.front();

    ur_context_handle_t hContext;
    ur_check(urContextCreate(1, &current_device, nullptr, &hContext));

    ur_program_handle_t hProgram;
    ur_check(urProgramCreateWithIL(hContext, spv.data(), spv.size(), nullptr,
                                   &hProgram));
    ur_check(urProgramBuild(hContext, hProgram, nullptr));

    ur_mem_handle_t dA, dB;
    ur_check(urMemBufferCreate(hContext, UR_MEM_FLAG_READ_WRITE,
                               a_size * sizeof(int), nullptr, &dA));
    ur_check(urMemBufferCreate(hContext, UR_MEM_FLAG_READ_WRITE,
                               a_size * sizeof(int), nullptr, &dB));

    ur_kernel_handle_t hKernel;
    ur_check(urKernelCreate(hProgram, "plus1", &hKernel));
    ur_check(urKernelSetArgMemObj(hKernel, 0, dA));
    ur_check(urKernelSetArgMemObj(hKernel, 1, dB));

    ur_queue_handle_t queue;
    ur_check(urQueueCreate(hContext, current_device, nullptr, &queue));

    ur_check(urEnqueueMemBufferWrite(queue, dA, true, 0, a_size * sizeof(int),
                                     a.data, 0, nullptr, nullptr));
    ur_check(urEnqueueMemBufferWrite(queue, dB, true, 0, a_size * sizeof(int),
                                     b.data, 0, nullptr, nullptr));

    const size_t gWorkOffset[] = {0, 0, 0};
    const size_t gWorkSize[] = {128, 1, 1};
    const size_t lWorkSize[] = {1, 1, 1};
    ur_event_handle_t event;
    ur_check(urEnqueueKernelLaunch(queue, hKernel, 3, gWorkOffset, gWorkSize,
                                   lWorkSize, 0, nullptr, &event));

    ur_check(urEnqueueMemBufferRead(queue, dB, true, 0, a_size * sizeof(int),
                                    b.data, 1, &event, nullptr));

    ur_check(urQueueFinish(queue));

    for (int i = 0; i < a_size; ++i) {
        std::cout << b.data[i] << " ";
    }
    std::cout << std::endl;

    return urTearDown(nullptr) == UR_RESULT_SUCCESS ? 0 : 1;
}