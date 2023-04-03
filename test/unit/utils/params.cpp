// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "ur_api.h"
#include "ur_params.hpp"

template <typename T> struct WrappedParams {
    typedef T value_type;
    virtual ~WrappedParams() {}
    virtual T *get_struct() = 0;
    virtual const char *get_expected() = 0;
};

template <typename T>
std::unique_ptr<WrappedParams<typename T::value_type>> createParams();

template <typename T> class ParamsTest : public testing::Test {
  protected:
    ParamsTest() : params(createParams<T>()) {}
    ~ParamsTest() override {}

    std::unique_ptr<WrappedParams<typename T::value_type>> params;
};

struct UrInitParams : public WrappedParams<ur_init_params_t> {
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

template <>
std::unique_ptr<WrappedParams<ur_init_params_t>>
createParams<UrInitParamsNoFlags>() {
    return std::make_unique<UrInitParamsNoFlags>();
}

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

template <>
std::unique_ptr<WrappedParams<ur_init_params_t>>
createParams<UrInitParamsInvalidFlags>() {
    return std::make_unique<UrInitParamsInvalidFlags>();
}

struct UrPlatformGet : public WrappedParams<ur_platform_get_params_t> {
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

template <>
std::unique_ptr<WrappedParams<ur_platform_get_params_t>>
createParams<UrPlatformGetEmptyArray>() {
    return std::make_unique<UrPlatformGetEmptyArray>();
}

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

template <>
std::unique_ptr<WrappedParams<ur_platform_get_params_t>>
createParams<UrPlatformGetTwoPlatforms>() {
    return std::make_unique<UrPlatformGetTwoPlatforms>();
}

struct UrUsmHostAllocParams : public WrappedParams<ur_usm_host_alloc_params_t> {
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

template <>
std::unique_ptr<WrappedParams<ur_usm_host_alloc_params_t>>
createParams<UrUsmHostAllocParamsEmpty>() {
    return std::make_unique<UrUsmHostAllocParamsEmpty>();
}

struct UrUsmHostAllocParamsUsmDesc : UrUsmHostAllocParams {
    ur_usm_desc_t usm_desc;
    UrUsmHostAllocParamsUsmDesc() : UrUsmHostAllocParams() {
        usm_desc.align = 64;
        usm_desc.flags = UR_USM_FLAG_BIAS_CACHED;
        usm_desc.hints = UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION;
        usm_desc.pNext = nullptr;
        usm_desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
        pUSMDesc = &usm_desc;
    }
    const char *get_expected() {
        return ".*\\.pUSMDesc = .+ \\(\\(struct "
               "ur_usm_desc_t\\)\\{\\.stype = UR_STRUCTURE_TYPE_USM_DESC, "
               "\\.pNext = "
               "nullptr, \\.flags = UR_USM_FLAG_BIAS_CACHED, \\.hints = "
               "UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION, \\.align = "
               "64\\}\\).*";
    };
};

template <>
std::unique_ptr<WrappedParams<ur_usm_host_alloc_params_t>>
createParams<UrUsmHostAllocParamsUsmDesc>() {
    return std::make_unique<UrUsmHostAllocParamsUsmDesc>();
}

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

template <>
std::unique_ptr<WrappedParams<ur_usm_host_alloc_params_t>>
createParams<UrUsmHostAllocParamsHostDesc>() {
    return std::make_unique<UrUsmHostAllocParamsHostDesc>();
}

using testing::Types;
typedef Types<UrInitParamsNoFlags, UrInitParamsInvalidFlags,
              UrUsmHostAllocParamsEmpty, UrPlatformGetEmptyArray,
              UrPlatformGetTwoPlatforms, UrUsmHostAllocParamsUsmDesc,
              UrUsmHostAllocParamsHostDesc>
    Implementations;

using ::testing::MatchesRegex;
using namespace ur_params;

TYPED_TEST_SUITE(ParamsTest, Implementations, );

TYPED_TEST(ParamsTest, Serialize) {
    std::ostringstream out;
    out << this->params->get_struct();
    EXPECT_THAT(out.str(), MatchesRegex(this->params->get_expected()));
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
