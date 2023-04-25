// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstring>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "ur_api.h"
#include "ur_params.hpp"

template <typename T> class ParamsTest : public testing::Test {
  protected:
    T params;
};

struct UrInitParams {
    ur_init_params_t params;
    ur_device_init_flags_t flags;
    UrInitParams(ur_device_init_flags_t _flags) : flags(_flags) {
        params.pdevice_flags = &flags;
    }

    ur_init_params_t *get_struct() { return &params; }
};

struct UrInitParamsNoFlags : UrInitParams {
    UrInitParamsNoFlags() : UrInitParams(0) {}
    const char *get_expected() { return ".device_flags = 0"; };
};

struct UrInitParamsInvalidFlags : UrInitParams {
    UrInitParamsInvalidFlags()
        : UrInitParams(UR_DEVICE_INIT_FLAG_GPU | UR_DEVICE_INIT_FLAG_MCA |
                       UR_BIT(25) | UR_BIT(30) | UR_BIT(31)) {}
    const char *get_expected() {
        return ".device_flags = UR_DEVICE_INIT_FLAG_GPU \\| "
               "UR_DEVICE_INIT_FLAG_MCA \\| unknown bit flags "
               "11000010000000000000000000000000";
    };
};

struct UrPlatformGet {
    ur_platform_get_params_t params;
    uint32_t num_entries;
    uint32_t *pNumPlatforms;
    ur_platform_handle_t *pPlatforms;
    UrPlatformGet() {
        num_entries = 0;
        pPlatforms = nullptr;
        pNumPlatforms = nullptr;
        params.pNumEntries = &num_entries;
        params.pphPlatforms = &pPlatforms;
        params.ppNumPlatforms = &pNumPlatforms;
    }

    ur_platform_get_params_t *get_struct() { return &params; }
};

struct UrPlatformGetEmptyArray : UrPlatformGet {
    UrPlatformGetEmptyArray() : UrPlatformGet() {}
    const char *get_expected() {
        return ".NumEntries = 0, .phPlatforms = \\[\\], .pNumPlatforms = "
               "nullptr";
    };
};

struct UrPlatformGetTwoPlatforms : UrPlatformGet {
    ur_platform_handle_t platforms[2] = {(ur_platform_handle_t)0xDEAFBEEFull,
                                         (ur_platform_handle_t)0xBADDCAFEull};
    uint32_t num_platforms;
    UrPlatformGetTwoPlatforms() : UrPlatformGet() {
        pPlatforms = (ur_platform_handle_t *)&platforms;
        num_entries = 2;
        num_platforms = 2;
        pNumPlatforms = &num_platforms;
    }
    const char *get_expected() {
        return ".NumEntries = 2, .phPlatforms = \\[.+, .+\\], "
               ".pNumPlatforms = .+ \\(2\\)";
    };
};

struct UrUsmHostAllocParams {
    ur_usm_host_alloc_params_t params;

    ur_context_handle_t hContext;

    const ur_usm_desc_t *pUSMDesc;
    ur_usm_pool_handle_t pool;
    size_t size;

    void *outptr;
    void **ppMem;

    UrUsmHostAllocParams() {
        hContext = nullptr;
        params.phContext = &hContext;
        pool = nullptr;
        params.ppool = &pool;
        outptr = nullptr;
        ppMem = &outptr;
        params.pppMem = &ppMem;
        pUSMDesc = nullptr;
        params.ppUSMDesc = &pUSMDesc;
        size = 0;
        params.psize = &size;
    }

    ur_usm_host_alloc_params_t *get_struct() { return &params; }
};

struct UrUsmHostAllocParamsEmpty : UrUsmHostAllocParams {
    UrUsmHostAllocParamsEmpty() : UrUsmHostAllocParams() {}
    const char *get_expected() {
        return "\\.hContext = nullptr, \\.pUSMDesc = nullptr, \\.pool = "
               "nullptr, "
               "\\.size = 0, \\.ppMem = .+ \\(nullptr\\)";
    };
};

struct UrUsmHostAllocParamsUsmDesc : UrUsmHostAllocParams {
    ur_usm_desc_t usm_desc;
    UrUsmHostAllocParamsUsmDesc() : UrUsmHostAllocParams() {
        usm_desc.align = 64;
        usm_desc.hints = UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION;
        usm_desc.pNext = nullptr;
        usm_desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
        pUSMDesc = &usm_desc;
    }
    const char *get_expected() {
        return ".*\\.pUSMDesc = .+ \\(\\(struct "
               "ur_usm_desc_t\\)\\{\\.stype = UR_STRUCTURE_TYPE_USM_DESC, "
               "\\.pNext = "
               "nullptr, \\.hints = "
               "UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION, \\.align = "
               "64\\}\\).*";
    };
};

struct UrUsmHostAllocParamsHostDesc : UrUsmHostAllocParamsUsmDesc {
    ur_usm_host_desc_t host_desc;
    UrUsmHostAllocParamsHostDesc() : UrUsmHostAllocParamsUsmDesc() {
        host_desc.flags = UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT;
        host_desc.pNext = nullptr;
        host_desc.stype = UR_STRUCTURE_TYPE_USM_HOST_DESC;
        usm_desc.pNext = &host_desc;
    }
    const char *get_expected() {
        return ".*\\.pNext = .+ \\(\\(struct "
               "ur_usm_host_desc_t\\)\\{\\.stype = "
               "UR_STRUCTURE_TYPE_USM_HOST_DESC, "
               "\\.pNext = "
               "nullptr, \\.flags = "
               "UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT\\}\\).*";
    };
};

struct UrDeviceGetInfoParams {
    ur_device_get_info_params_t params;

    ur_device_handle_t device;
    ur_device_info_t propName;
    size_t propSize;
    void *pPropValue;
    size_t propSizeRet;
    size_t *pPropSizeRet;

    UrDeviceGetInfoParams() {
        device = nullptr;
        propName = UR_DEVICE_INFO_FORCE_UINT32;
        propSize = 0;
        pPropValue = nullptr;
        propSizeRet = 0;
        pPropSizeRet = &propSizeRet;

        params.phDevice = &device;
        params.ppPropValue = &pPropValue;
        params.ppropName = &propName;
        params.ppropSize = &propSize;
        params.ppPropSizeRet = &pPropSizeRet;
    }

    ur_device_get_info_params_t *get_struct() { return &params; }
};

struct UrDeviceGetInfoParamsEmpty : UrDeviceGetInfoParams {
    UrDeviceGetInfoParamsEmpty() : UrDeviceGetInfoParams() {}
    const char *get_expected() {
        return ".hDevice = nullptr, .propName = unknown enumerator, .propSize "
               "= 0, .pPropValue = nullptr, .pPropSizeRet = .+ \\(0\\)";
    };
};

struct UrDeviceGetInfoParamsName : UrDeviceGetInfoParams {
    const char *name = "FOOBAR";
    UrDeviceGetInfoParamsName() : UrDeviceGetInfoParams() {
        propName = UR_DEVICE_INFO_NAME;
        pPropValue = (void *)name;
        propSize = strlen(name) + 1;
        propSizeRet = strlen(name) + 1;
    }
    const char *get_expected() {
        return ".hDevice = nullptr, .propName = UR_DEVICE_INFO_NAME, .propSize "
               "= 7, .pPropValue = .+ \\(FOOBAR\\), .pPropSizeRet = .+ \\(7\\)";
    };
};

struct UrDeviceGetInfoParamsQueueFlag : UrDeviceGetInfoParams {
    ur_queue_flags_t flags;
    UrDeviceGetInfoParamsQueueFlag() : UrDeviceGetInfoParams() {
        flags = UR_QUEUE_FLAG_ON_DEVICE_DEFAULT | UR_QUEUE_FLAG_PRIORITY_HIGH;
        propName = UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES;
        pPropValue = &flags;
        propSize = sizeof(flags);
        propSizeRet = sizeof(flags);
    }
    const char *get_expected() {
        return ".hDevice = nullptr, .propName = "
               "UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES, .propSize "
               "= 4, .pPropValue = .+ \\(UR_QUEUE_FLAG_ON_DEVICE_DEFAULT \\| "
               "UR_QUEUE_FLAG_PRIORITY_HIGH\\), .pPropSizeRet = .+ \\(4\\)";
    };
};

struct UrDeviceGetInfoParamsInvalidSize : UrDeviceGetInfoParams {
    ur_device_type_t t;
    UrDeviceGetInfoParamsInvalidSize() : UrDeviceGetInfoParams() {
        t = UR_DEVICE_TYPE_GPU;
        propName = UR_DEVICE_INFO_TYPE;
        pPropValue = &t;
        propSize = 1;
        propSizeRet = sizeof(t);
    }
    const char *get_expected() {
        return ".+ .pPropValue = invalid size \\(is: 1, expected: >=4\\).+";
    };
};

struct UrDeviceGetInfoParamsPartitionArray : UrDeviceGetInfoParams {
    ur_device_partition_property_t props[3] = {
        UR_DEVICE_PARTITION_BY_COUNTS, UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
        UR_DEVICE_PARTITION_BY_CSLICE};
    UrDeviceGetInfoParamsPartitionArray() : UrDeviceGetInfoParams() {
        propName = UR_DEVICE_INFO_PARTITION_PROPERTIES;
        pPropValue = &props;
        propSize = sizeof(props);
        propSizeRet = sizeof(props);
    }
    const char *get_expected() {
        return ".hDevice = nullptr, .propName = "
               "UR_DEVICE_INFO_PARTITION_PROPERTIES, .propSize "
               "= 24, .pPropValue = \\[4231, 4232, 4233\\], .pPropSizeRet = .+ "
               "\\(24\\)";
        // TODO: should resolve type values for ur_device_partition_property_t...
    };
};

struct UrContextGetInfoParams {
    ur_context_get_info_params_t params;

    ur_context_handle_t hContext;
    ur_context_info_t propName;
    size_t propSize;
    void *pPropValue;
    size_t propSizeRet;
    size_t *pPropSizeRet;

    UrContextGetInfoParams() {
        hContext = nullptr;
        propName = UR_CONTEXT_INFO_FORCE_UINT32;
        propSize = 0;
        pPropValue = nullptr;
        propSizeRet = 0;
        pPropSizeRet = &propSizeRet;

        params.phContext = &hContext;
        params.ppPropValue = &pPropValue;
        params.ppropName = &propName;
        params.ppropSize = &propSize;
        params.ppPropSizeRet = &pPropSizeRet;
    }

    ur_context_get_info_params_t *get_struct() { return &params; }
};

struct UrContextGetInfoParamsDevicesArray : UrContextGetInfoParams {
    ur_device_handle_t handles[3] = {(ur_device_handle_t)0xABADull,
                                     (ur_device_handle_t)0xCAFEull,
                                     (ur_device_handle_t)0xBABEull};
    UrContextGetInfoParamsDevicesArray() : UrContextGetInfoParams() {
        propName = UR_CONTEXT_INFO_DEVICES;
        propSize = sizeof(handles);
        propSizeRet = sizeof(handles);
        pPropValue = handles;
    }
    const char *get_expected() {
        return ".hContext = nullptr, .propName = "
               "UR_CONTEXT_INFO_DEVICES, .propSize "
               "= 24, .pPropValue = \\[.+, .+, .+\\], .pPropSizeRet = .+ "
               "\\(24\\)";
    };
};

using testing::Types;
typedef Types<
    UrInitParamsNoFlags, UrInitParamsInvalidFlags, UrUsmHostAllocParamsEmpty,
    UrPlatformGetEmptyArray, UrPlatformGetTwoPlatforms,
    UrUsmHostAllocParamsUsmDesc, UrUsmHostAllocParamsHostDesc,
    UrDeviceGetInfoParamsEmpty, UrDeviceGetInfoParamsName,
    UrDeviceGetInfoParamsQueueFlag, UrDeviceGetInfoParamsPartitionArray,
    UrContextGetInfoParamsDevicesArray, UrDeviceGetInfoParamsInvalidSize>
    Implementations;

using ::testing::MatchesRegex;
using namespace ur_params;

TYPED_TEST_SUITE(ParamsTest, Implementations, );

TYPED_TEST(ParamsTest, Serialize) {
    std::ostringstream out;
    out << this->params.get_struct();
    EXPECT_THAT(out.str(), MatchesRegex(this->params.get_expected()));
}

TEST(SerializePtr, nested_void_ptrs) {
    void *real = (void *)0xFEEDCAFEull;
    void **preal = &real;
    void ***ppreal = &preal;
    void ****pppreal = &ppreal;
    std::ostringstream out;
    serializePtr(out, pppreal);
    EXPECT_THAT(out.str(), MatchesRegex(".+ \\(.+ \\(.+ \\(.+\\)\\)\\)"));
}
