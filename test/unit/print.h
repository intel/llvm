// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include "ur_print.h"

struct UrLoaderInitParams {
    ur_loader_init_params_t params;
    ur_device_init_flags_t flags;
    ur_loader_config_handle_t config;

    UrLoaderInitParams(ur_device_init_flags_t _flags)
        : flags(_flags), config(nullptr) {
        params.pdevice_flags = &flags;
        params.phLoaderConfig = &config;
    }

    ur_loader_init_params_t *get_struct() { return &params; }
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintLoaderInitParams(&params, buffer, buff_size, out_size);
    }
};

struct UrLoaderInitParamsNoFlags : UrLoaderInitParams {
    UrLoaderInitParamsNoFlags() : UrLoaderInitParams(0) {}
    const char *get_expected() {
        return ".device_flags = 0, .hLoaderConfig = nullptr";
    };
};

struct UrLoaderInitParamsInvalidFlags : UrLoaderInitParams {
    UrLoaderInitParamsInvalidFlags()
        : UrLoaderInitParams(UR_DEVICE_INIT_FLAG_GPU | UR_DEVICE_INIT_FLAG_MCA |
                             UR_BIT(25) | UR_BIT(30) | UR_BIT(31)) {}
    const char *get_expected() {
        return ".device_flags = UR_DEVICE_INIT_FLAG_GPU \\| "
               "UR_DEVICE_INIT_FLAG_MCA \\| unknown bit flags "
               "11000010000000000000000000000000, "
               ".hLoaderConfig = nullptr";
    };
};

struct UrPlatformGet {
    ur_platform_get_params_t params;
    uint32_t num_adapters;
    ur_adapter_handle_t *phAdapters;
    uint32_t num_entries;
    uint32_t *pNumPlatforms;
    ur_platform_handle_t *pPlatforms;
    UrPlatformGet() {
        num_adapters = 0;
        phAdapters = nullptr;
        num_entries = 0;
        pPlatforms = nullptr;
        pNumPlatforms = nullptr;
        params.pNumAdapters = &num_adapters;
        params.pphAdapters = &phAdapters;
        params.pNumEntries = &num_entries;
        params.pphPlatforms = &pPlatforms;
        params.ppNumPlatforms = &pNumPlatforms;
    }

    ur_platform_get_params_t *get_struct() { return &params; }
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintPlatformGetParams(&params, buffer, buff_size, out_size);
    }
};

struct UrPlatformGetEmptyArray : UrPlatformGet {
    UrPlatformGetEmptyArray() : UrPlatformGet() {}
    const char *get_expected() {
        return ".phAdapters = \\{\\}, .NumAdapters = 0, .NumEntries = 0, "
               ".phPlatforms = \\{\\}, .pNumPlatforms = "
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
        return ".phAdapters = \\{\\}, .NumAdapters = 0, .NumEntries = 2, "
               ".phPlatforms = \\{.+, .+\\}, "
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
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintUsmHostAllocParams(&params, buffer, buff_size, out_size);
    }
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
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintDeviceGetInfoParams(&params, buffer, buff_size, out_size);
    }
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
    ur_device_partition_t props[3] = {UR_DEVICE_PARTITION_BY_COUNTS,
                                      UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                                      UR_DEVICE_PARTITION_BY_CSLICE};
    UrDeviceGetInfoParamsPartitionArray() : UrDeviceGetInfoParams() {
        propName = UR_DEVICE_INFO_SUPPORTED_PARTITIONS;
        pPropValue = &props;
        propSize = sizeof(props);
        propSizeRet = sizeof(props);
    }
    const char *get_expected() {
        return ".hDevice = nullptr, .propName = "
               "UR_DEVICE_INFO_SUPPORTED_PARTITIONS, .propSize "
               "= 12, .pPropValue = \\{UR_DEVICE_PARTITION_BY_COUNTS, "
               "UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, "
               "UR_DEVICE_PARTITION_BY_CSLICE\\}, .pPropSizeRet = .+ "
               "\\(12\\)";
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
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintContextGetInfoParams(&params, buffer, buff_size, out_size);
    }
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
               "= 24, .pPropValue = \\{.+, .+, .+\\}, .pPropSizeRet = .+ "
               "\\(24\\)";
    };
};

struct UrProgramMetadataTest {
    UrProgramMetadataTest() {
        meta.pName = "MY_META";
        meta.size = 0;
        meta.type = UR_PROGRAM_METADATA_TYPE_UINT32;
        ur_program_metadata_value_t value{};
        value.data32 = 42;
        meta.value = value;
    }

    ur_program_metadata_t &get_struct() { return meta; }
    const char *get_expected() {
        return "\\(struct ur_program_metadata_t\\)"
               "\\{"
               ".pName = .+ \\(MY_META\\), "
               ".type = UR_PROGRAM_METADATA_TYPE_UINT32, "
               ".size = 0, "
               ".value = \\(union ur_program_metadata_value_t\\)\\{"
               ".data32 = 42"
               "\\}"
               "\\}";
    }
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintProgramMetadata(meta, buffer, buff_size, out_size);
    }
    ur_program_metadata_t meta;
};

struct UrDevicePartitionPropertyTest {
    UrDevicePartitionPropertyTest() {
        prop.type = UR_DEVICE_PARTITION_EQUALLY;
        ur_device_partition_value_t value{};
        value.equally = 4;
        prop.value = value;
    }

    ur_device_partition_property_t &get_struct() { return prop; }
    const char *get_expected() {
        return "\\(struct ur_device_partition_property_t\\)"
               "\\{"
               ".type = UR_DEVICE_PARTITION_EQUALLY, "
               ".value = \\(union ur_device_partition_value_t\\)\\{"
               ".equally = 4"
               "\\}"
               "\\}";
    }
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintDevicePartitionProperty(prop, buffer, buff_size, out_size);
    }

    ur_device_partition_property_t prop;
};

struct UrSamplerAddressModesTest {
    UrSamplerAddressModesTest() {
        prop.addrModes[0] = UR_SAMPLER_ADDRESSING_MODE_CLAMP;
        prop.addrModes[1] = UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
        prop.addrModes[2] = UR_SAMPLER_ADDRESSING_MODE_REPEAT;
        prop.pNext = nullptr;
        prop.stype = UR_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES;
    }
    ur_exp_sampler_addr_modes_t &get_struct() { return prop; }
    const char *get_expected() {
        return "\\(struct ur_exp_sampler_addr_modes_t\\)"
               "\\{"
               ".stype = UR_STRUCTURE_TYPE_EXP_SAMPLER_ADDR_MODES, "
               ".pNext = nullptr, "
               ".addrModes = \\{"
               "UR_SAMPLER_ADDRESSING_MODE_CLAMP, "
               "UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT, "
               "UR_SAMPLER_ADDRESSING_MODE_REPEAT"
               "\\}"
               "\\}";
    }
    void print(char *buffer, const size_t buff_size, size_t *out_size) {
        urPrintExpSamplerAddrModes(prop, buffer, buff_size, out_size);
    }

    ur_exp_sampler_addr_modes_t prop;
};

template <typename T> class ParamsTest : public testing::Test {
  protected:
    void print(char *buffer, const size_t buff_size, size_t *out_size);
    T params;
};

typedef ::testing::Types<
    UrLoaderInitParamsNoFlags, UrLoaderInitParamsInvalidFlags,
    UrUsmHostAllocParamsEmpty, UrPlatformGetEmptyArray,
    UrPlatformGetTwoPlatforms, UrUsmHostAllocParamsUsmDesc,
    UrUsmHostAllocParamsHostDesc, UrDeviceGetInfoParamsEmpty,
    UrDeviceGetInfoParamsName, UrDeviceGetInfoParamsQueueFlag,
    UrDeviceGetInfoParamsPartitionArray, UrContextGetInfoParamsDevicesArray,
    UrDeviceGetInfoParamsInvalidSize, UrProgramMetadataTest,
    UrDevicePartitionPropertyTest, UrSamplerAddressModesTest>
    Implementations;

TYPED_TEST_SUITE(ParamsTest, Implementations, );

using ::testing::MatchesRegex;
