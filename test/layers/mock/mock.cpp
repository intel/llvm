/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file codeloc.cpp
 *
 */

#include "uur/raii.h"
#include <gtest/gtest.h>
#include <ur_api.h>

TEST(Mock, NullProperties) {
    uur::raii::LoaderConfig loader_config;
    ASSERT_EQ(urLoaderConfigCreate(loader_config.ptr()), UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderConfigSetMockCallbacks(loader_config, nullptr),
              UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST(Mock, NullCallback) {
    uur::raii::LoaderConfig loader_config;
    ASSERT_EQ(urLoaderConfigCreate(loader_config.ptr()), UR_RESULT_SUCCESS);

    ur_mock_callback_properties_t callback_properties = {
        UR_STRUCTURE_TYPE_MOCK_CALLBACK_PROPERTIES, nullptr, "urAdapterGet",
        UR_CALLBACK_OVERRIDE_MODE_REPLACE, nullptr};

    ASSERT_EQ(
        urLoaderConfigSetMockCallbacks(loader_config, &callback_properties),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

ur_result_t generic_callback(void *) { return UR_RESULT_SUCCESS; }

TEST(Mock, NullHandle) {
    ur_mock_callback_properties_t callback_properties = {
        UR_STRUCTURE_TYPE_MOCK_CALLBACK_PROPERTIES, nullptr, "urAdapterGet",
        UR_CALLBACK_OVERRIDE_MODE_REPLACE, &generic_callback};

    ASSERT_EQ(urLoaderConfigSetMockCallbacks(nullptr, &callback_properties),
              UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST(Mock, DefaultBehavior) {
    uur::raii::LoaderConfig loader_config;
    ASSERT_EQ(urLoaderConfigCreate(loader_config.ptr()), UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderConfigEnableLayer(loader_config, "UR_LAYER_MOCK"),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderInit(0, loader_config), UR_RESULT_SUCCESS);

    // Set up as far as device and check we're getting sensible, different
    // handles created.
    ur_adapter_handle_t adapter = nullptr;
    ur_platform_handle_t platform = nullptr;
    ur_device_handle_t device = nullptr;

    ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
    ASSERT_EQ(urPlatformGet(&adapter, 1, 1, &platform, nullptr),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 1, &device, nullptr),
              UR_RESULT_SUCCESS);

    ASSERT_NE(adapter, nullptr);
    ASSERT_NE(platform, nullptr);
    ASSERT_NE(device, nullptr);

    ASSERT_NE(static_cast<void *>(adapter), static_cast<void *>(platform));
    ASSERT_NE(static_cast<void *>(adapter), static_cast<void *>(device));
    ASSERT_NE(static_cast<void *>(platform), static_cast<void *>(device));

    ASSERT_EQ(urDeviceRelease(device), UR_RESULT_SUCCESS);
}

void checkPreInitAdapter(ur_adapter_handle_t adapter) {
    ur_adapter_handle_t preInitAdapter =
        reinterpret_cast<ur_adapter_handle_t>(0xF00DCAFE);
    ASSERT_EQ(adapter, preInitAdapter);
}

ur_result_t beforeUrAdapterGet(void *pParams) {
    auto params = reinterpret_cast<ur_adapter_get_params_t *>(pParams);
    checkPreInitAdapter(**params->pphAdapters);
    return UR_RESULT_SUCCESS;
}

ur_result_t replaceUrAdapterGet(void *pParams) {
    auto params = reinterpret_cast<ur_adapter_get_params_t *>(pParams);
    **params->pphAdapters = reinterpret_cast<ur_adapter_handle_t>(0xDEADBEEF);
    return UR_RESULT_SUCCESS;
}

void checkPostInitAdapter(ur_adapter_handle_t adapter) {
    ur_adapter_handle_t postInitAdapter =
        reinterpret_cast<ur_adapter_handle_t>(0xDEADBEEF);
    ASSERT_EQ(adapter, postInitAdapter);
}

ur_result_t afterUrAdapterGet(void *pParams) {
    auto params = reinterpret_cast<ur_adapter_get_params_t *>(pParams);
    checkPostInitAdapter(**params->pphAdapters);
    return UR_RESULT_SUCCESS;
}

TEST(Mock, Callbacks) {
    uur::raii::LoaderConfig loader_config;
    ASSERT_EQ(urLoaderConfigCreate(loader_config.ptr()), UR_RESULT_SUCCESS);

    // This callback is set up to check *phAdapters is still the pre-call
    // init value we set below
    ur_mock_callback_properties_t adapterGetBeforeProperties = {
        UR_STRUCTURE_TYPE_MOCK_CALLBACK_PROPERTIES, nullptr, "urAdapterGet",
        UR_CALLBACK_OVERRIDE_MODE_BEFORE, &beforeUrAdapterGet};

    // This callback is set up to return a distinct test value in phAdapters
    // rather than the default generic handle
    ur_mock_callback_properties_t adapterGetReplaceProperties = {
        UR_STRUCTURE_TYPE_MOCK_CALLBACK_PROPERTIES, &adapterGetBeforeProperties,
        "urAdapterGet", UR_CALLBACK_OVERRIDE_MODE_REPLACE,
        &replaceUrAdapterGet};

    // This callback is set up to check our replace callback did its job
    ur_mock_callback_properties_t adapterGetAfterProperties = {
        UR_STRUCTURE_TYPE_MOCK_CALLBACK_PROPERTIES,
        &adapterGetReplaceProperties, "urAdapterGet",
        UR_CALLBACK_OVERRIDE_MODE_AFTER, &afterUrAdapterGet};

    ASSERT_EQ(urLoaderConfigSetMockCallbacks(loader_config,
                                             &adapterGetAfterProperties),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderConfigEnableLayer(loader_config, "UR_LAYER_MOCK"),
              UR_RESULT_SUCCESS);
    ASSERT_EQ(urLoaderInit(0, loader_config), UR_RESULT_SUCCESS);

    ur_adapter_handle_t adapter =
        reinterpret_cast<ur_adapter_handle_t>(0xF00DCAFE);
    ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
}
