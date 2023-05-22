// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED

#include "ur_api.h"
#include <uur/checks.h>
#include <uur/environment.h>
#include <uur/utils.h>

#define UUR_RETURN_ON_FATAL_FAILURE(...)                                       \
    __VA_ARGS__;                                                               \
    if (this->HasFatalFailure() || this->IsSkipped()) {                        \
        return;                                                                \
    }                                                                          \
    (void)0

namespace uur {

struct urPlatformTest : ::testing::Test {
    void SetUp() override {
        platform = uur::PlatformEnvironment::instance->platform;
    }

    ur_platform_handle_t platform = nullptr;
};

inline std::pair<bool, std::vector<ur_device_handle_t>>
GetDevices(ur_platform_handle_t platform) {
    uint32_t count = 0;
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count)) {
        return {false, {}};
    }
    if (count == 0) {
        return {false, {}};
    }
    std::vector<ur_device_handle_t> devices(count);
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                    nullptr)) {
        return {false, {}};
    }
    return {true, devices};
}

inline bool
hasDevicePartitionSupport(ur_device_handle_t device,
                          const ur_device_partition_property_t property) {
    std::vector<ur_device_partition_property_t> properties;
    uur::GetDevicePartitionProperties(device, properties);
    return std::find(properties.begin(), properties.end(), property) !=
           properties.end();
}

struct urAllDevicesTest : urPlatformTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        auto devicesPair = GetDevices(platform);
        if (!devicesPair.first) {
            FAIL() << "Failed to get devices";
        }
        devices = std::move(devicesPair.second);
    }

    void TearDown() override {
        for (auto &device : devices) {
            EXPECT_SUCCESS(urDeviceRelease(device));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::TearDown());
    }

    std::vector<ur_device_handle_t> devices;
};

struct urDeviceTest : urPlatformTest,
                      ::testing::WithParamInterface<ur_device_handle_t> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        device = GetParam();
    }

    void TearDown() override {
        EXPECT_SUCCESS(urDeviceRelease(device));
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::TearDown());
    }

    ur_device_handle_t device;
};
} // namespace uur

#define UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(FIXTURE)                           \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),       \
        [](const ::testing::TestParamInfo<ur_device_handle_t> &info) {         \
            return uur::GetPlatformAndDeviceName(info.param);                  \
        })

#define UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(FIXTURE)                           \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        ::testing::ValuesIn(uur::KernelsEnvironment::instance->devices),       \
        [](const ::testing::TestParamInfo<ur_device_handle_t> &info) {         \
            return uur::GetPlatformAndDeviceName(info.param);                  \
        })

namespace uur {

template <class T>
struct urDeviceTestWithParam
    : urPlatformTest,
      ::testing::WithParamInterface<std::tuple<ur_device_handle_t, T>> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        device = std::get<0>(this->GetParam());
    }
    // TODO - I don't like the confusion with GetParam();
    const T &getParam() const { return std::get<1>(this->GetParam()); }
    ur_device_handle_t device;
};

struct urContextTest : urDeviceTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::SetUp());
        ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context));
        ASSERT_NE(context, nullptr);
    }

    void TearDown() override {
        EXPECT_SUCCESS(urContextRelease(context));
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::TearDown());
    }

    ur_context_handle_t context = nullptr;
};

struct urMemBufferTest : urContextTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 4096,
                                         nullptr, &buffer));
        ASSERT_NE(nullptr, buffer);
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_mem_handle_t buffer = nullptr;
};

} // namespace uur

#define UUR_TEST_SUITE_P(FIXTURE, VALUES, PRINTER)                             \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        testing::Combine(                                                      \
            ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),   \
            VALUES),                                                           \
        PRINTER)

namespace uur {

template <class T> struct urContextTestWithParam : urDeviceTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urContextCreate(1, &this->device, nullptr, &context));
    }

    void TearDown() override {
        EXPECT_SUCCESS(urContextRelease(context));
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTestWithParam<T>::TearDown());
    }
    ur_context_handle_t context;
};

template <class T> struct urMemBufferTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                         4096, nullptr, &buffer));
        ASSERT_NE(nullptr, buffer);
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }
    ur_mem_handle_t buffer = nullptr;
};

struct urQueueTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
        ASSERT_NE(queue, nullptr);
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_queue_handle_t queue = nullptr;
};

struct urHostPipeTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        uur::KernelsEnvironment::instance->LoadSource("foo", 0, il_binary);
        ASSERT_SUCCESS(urProgramCreateWithIL(
            context, il_binary->data(), il_binary->size(), nullptr, &program));

        size_t size = 0;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED, 0, nullptr,
            &size));
        ASSERT_NE(size, 0);
        ASSERT_EQ(sizeof(ur_bool_t), size);
        void *info_data = alloca(size);
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED, size,
            info_data, nullptr));
        ASSERT_NE(info_data, nullptr);

        bool supported;
        GetDeviceHostPipeRWSupported(device, supported);
        if (!supported) {
            GTEST_SKIP() << "Host pipe read/write is not supported.";
        }
    }

    void TearDown() override {
        if (program) {
            EXPECT_SUCCESS(urProgramRelease(program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "foo";
    ur_program_handle_t program = nullptr;

    const char *pipe_symbol = "pipe_symbol";

    static const size_t size = 1024;
    char buffer[size];
};

template <class T> struct urQueueTestWithParam : urContextTestWithParam<T> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urQueueCreate(this->context, this->device, 0, &queue));
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    ur_queue_handle_t queue;
};

struct urProfilingQueueTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ur_queue_properties_t props = {
            /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
            /*.pNext =*/nullptr,
            /*.flags =*/UR_QUEUE_FLAG_PROFILING_ENABLE,
        };
        ASSERT_SUCCESS(
            urQueueCreate(this->context, this->device, &props, &queue));
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    };

    ur_queue_handle_t queue = nullptr;
};

template <class T>
struct urProfilingQueueTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ur_queue_properties_t props = {
            /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
            /*.pNext =*/nullptr,
            /*.flags =*/UR_QUEUE_FLAG_PROFILING_ENABLE,
        };
        ASSERT_SUCCESS(
            urQueueCreate(this->context, this->device, &props, &queue));
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    };

    ur_queue_handle_t queue = nullptr;
};

struct urMultiQueueTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue1));
        ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue2));
    }

    void TearDown() override {
        if (queue1 != nullptr) {
            EXPECT_SUCCESS(urQueueRelease(queue1));
        }
        if (queue2 != nullptr) {
            EXPECT_SUCCESS(urQueueRelease(queue2));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_queue_handle_t queue1 = nullptr;
    ur_queue_handle_t queue2 = nullptr;
};

struct urMultiDeviceContextTest : urPlatformTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        auto &devices = DevicesEnvironment::instance->devices;
        if (devices.size() <= 1) {
            GTEST_SKIP();
        }
        ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(devices.size()),
                                       devices.data(), nullptr, &context));
    }

    void TearDown() override {
        if (context) {
            ASSERT_SUCCESS(urContextRelease(context));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::TearDown());
    }

    ur_context_handle_t context = nullptr;
};

struct urMultiDeviceMemBufferTest : urMultiDeviceContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                         nullptr, &buffer));
        ASSERT_NE(nullptr, buffer);
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::TearDown());
    }

    ur_mem_handle_t buffer = nullptr;
    const size_t count = 1024;
    const size_t size = count * sizeof(uint32_t);
};

struct urMultiDeviceMemBufferQueueTest : urMultiDeviceMemBufferTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemBufferTest::SetUp());
        queues.reserve(DevicesEnvironment::instance->devices.size());
        for (const auto &device : DevicesEnvironment::instance->devices) {
            ur_queue_handle_t queue = nullptr;
            ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
            queues.push_back(queue);
        }
    }

    void TearDown() override {
        for (const auto &queue : queues) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemBufferTest::TearDown());
    }

    std::vector<ur_queue_handle_t> queues;
};

struct urMemBufferQueueTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                         nullptr, &buffer));
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    const size_t count = 8;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t buffer = nullptr;
};

struct urUSMDeviceAllocTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());
        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(GetDeviceUSMDeviceSupport(device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM in not supported";
        }
        ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                        allocation_size, &ptr));
        ur_event_handle_t event = nullptr;

        uint8_t fillPattern = 0;
        ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(fillPattern),
                                        &fillPattern, allocation_size, 0,
                                        nullptr, &event));

        EXPECT_SUCCESS(urQueueFlush(queue));
        ASSERT_SUCCESS(urEventWait(1, &event));
        EXPECT_SUCCESS(urEventRelease(event));
    }

    void TearDown() override {
        ASSERT_SUCCESS(urUSMFree(context, ptr));
        uur::urQueueTest::TearDown();
    }

    size_t allocation_size = sizeof(int);
    void *ptr = nullptr;
};

struct urUSMPoolTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                     UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
        ur_usm_pool_handle_t pool = nullptr;
        ASSERT_SUCCESS(urUSMPoolCreate(this->context, &pool_desc, &pool));
    }

    void TearDown() override {
        if (pool) {
            EXPECT_SUCCESS(urUSMPoolRelease(pool));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_usm_pool_handle_t pool;
};

template <class T> struct urUSMPoolTestWithParam : urContextTestWithParam<T> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                     UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
        ur_usm_pool_handle_t pool = nullptr;
        ASSERT_SUCCESS(urUSMPoolCreate(this->context, &pool_desc, &pool));
    }

    void TearDown() override {
        if (pool) {
            EXPECT_SUCCESS(urUSMPoolRelease(pool));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    ur_usm_pool_handle_t pool;
};

template <class T>
struct urUSMDeviceAllocTestWithParam : urQueueTestWithParam<T> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTestWithParam<T>::SetUp());
        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(GetDeviceUSMDeviceSupport(this->device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM in not supported";
        }
        ASSERT_SUCCESS(urUSMDeviceAlloc(this->context, this->device, nullptr,
                                        nullptr, allocation_size, &ptr));
        ur_event_handle_t event = nullptr;

        uint8_t fillPattern = 0;
        ASSERT_SUCCESS(urEnqueueUSMFill(this->queue, ptr, sizeof(fillPattern),
                                        &fillPattern, allocation_size, 0,
                                        nullptr, &event));

        EXPECT_SUCCESS(urQueueFlush(this->queue));
        ASSERT_SUCCESS(urEventWait(1, &event));
        EXPECT_SUCCESS(urEventRelease(event));
    }

    void TearDown() override {
        ASSERT_SUCCESS(urUSMFree(this->context, ptr));
        uur::urQueueTestWithParam<T>::TearDown();
    }

    size_t allocation_size = sizeof(int);
    void *ptr = nullptr;
};

/// @brief
/// @tparam T
/// @param info
/// @return
template <class T>
std::string deviceTestWithParamPrinter(
    const ::testing::TestParamInfo<std::tuple<ur_device_handle_t, T>> &info) {
    auto device = std::get<0>(info.param);
    auto param = std::get<1>(info.param);

    std::stringstream ss;
    ss << param;
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}

struct urProgramTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        uur::KernelsEnvironment::instance->LoadSource("foo", 0, il_binary);
        ASSERT_SUCCESS(urProgramCreateWithIL(
            context, il_binary->data(), il_binary->size(), nullptr, &program));
    }

    void TearDown() override {
        if (program) {
            EXPECT_SUCCESS(urProgramRelease(program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "foo";
    ur_program_handle_t program = nullptr;
};

template <class T> struct urProgramTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        uur::KernelsEnvironment::instance->LoadSource("foo", 0, il_binary);
        ASSERT_SUCCESS(urProgramCreateWithIL(this->context, il_binary->data(),
                                             il_binary->size(), nullptr,
                                             &program));
    }

    void TearDown() override {
        if (program) {
            EXPECT_SUCCESS(urProgramRelease(program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "foo";
    ur_program_handle_t program = nullptr;
};

inline std::string getKernelName(ur_program_handle_t program) {
    size_t kernel_string_size = 0;
    if (UR_RESULT_SUCCESS != urProgramGetInfo(program,
                                              UR_PROGRAM_INFO_KERNEL_NAMES, 0,
                                              nullptr, &kernel_string_size)) {
        return "";
    }
    std::string kernel_string;
    kernel_string.resize(kernel_string_size);
    if (UR_RESULT_SUCCESS !=
        urProgramGetInfo(program, UR_PROGRAM_INFO_KERNEL_NAMES,
                         kernel_string.size(), kernel_string.data(), nullptr)) {
        return "";
    }
    std::stringstream kernel_stream(kernel_string);
    std::string kernel_name;
    bool found_kernel = false;
    // Go through the semi-colon separated list of kernel names looking for
    // one that isn't a wrapper or an offset handler.
    while (kernel_stream.good()) {
        getline(kernel_stream, kernel_name, ';');
        if (kernel_name.find("wrapper") == std::string::npos &&
            kernel_name.find("offset") == std::string::npos) {
            found_kernel = true;
            break;
        }
    }
    return found_kernel ? kernel_name : "";
}

struct urKernelTest : urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        kernel_name = getKernelName(program);
        ASSERT_FALSE(kernel_name.empty());
    }

    void TearDown() override {
        if (kernel) {
            ASSERT_SUCCESS(urKernelRelease(kernel));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::TearDown());
    }

    std::string kernel_name;
    ur_kernel_handle_t kernel = nullptr;
};

template <class T> struct urKernelTestWithParam : urProgramTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urProgramBuild(this->context, this->program, nullptr));
        kernel_name = getKernelName(this->program);
        ASSERT_FALSE(kernel_name.empty());
    }

    void TearDown() override {
        if (kernel) {
            EXPECT_SUCCESS(urKernelRelease(kernel));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTestWithParam<T>::TearDown());
    }

    std::string kernel_name;
    ur_kernel_handle_t kernel = nullptr;
};
} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED
