# Copyright (C) 2022-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(UR_BUILD_ADAPTER_L0)
    set(ADAPTER_LIB_TYPE SHARED)
    if(UR_STATIC_ADAPTER_L0)
        set(ADAPTER_LIB_TYPE STATIC)
    endif()

    add_ur_adapter(ur_adapter_level_zero ${ADAPTER_LIB_TYPE}
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/async_alloc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/image_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/memory_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_level_zero.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enqueue_native.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/image_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/memory_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
    )
    install_ur_library(ur_adapter_level_zero)

    target_compile_definitions(ur_adapter_level_zero PUBLIC UR_STATIC_LEVEL_ZERO)
    if(UR_STATIC_ADAPTER_L0)
        target_compile_definitions(ur_adapter_level_zero PUBLIC UR_STATIC_ADAPTER_LEVEL_ZERO)

        # 'utils' target from 'level-zero-loader' includes path which is prefixed
        # in the source directory, this breaks the installation of 'utils' target.
        set_target_properties(level_zero_utils PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
        install(TARGETS ur_umf LevelZeroLoader LevelZeroLoader-Headers ComputeRuntimeLevelZero-Headers ze_loader level_zero_utils
            EXPORT ${PROJECT_NAME}-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

    # Ensure UR flags are propagated to level zero
    # Note: UR compile options cause issues under MSVC
    if(NOT MSVC)
        foreach(TARGET IN ITEMS ze_loader ze_validation_layer ze_tracing_layer ze_null)
            if (TARGET TARGET)
                add_ur_target_compile_options(${TARGET})
                add_ur_target_link_options(${TARGET})
                target_compile_options(${TARGET} PRIVATE
                    $<$<CXX_COMPILER_ID:GNU,Clang,Intel,IntelLLVM>:-Wno-error -Wno-unused-parameter>
                    $<$<CXX_COMPILER_ID:MSVC>:/WX- /UUNICODE>
                )
            endif()
        endforeach()
    endif()

    if(NOT WIN32)
        target_sources(ur_adapter_level_zero
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/adapter_lib_init_linux.cpp
        )
    endif()

    # TODO: fix level_zero adapter conversion warnings
    target_compile_options(ur_adapter_level_zero PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/wd4805 /wd4244>
    )

    set_target_properties(ur_adapter_level_zero PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
    )

    target_link_libraries(ur_adapter_level_zero PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )

    target_include_directories(ur_adapter_level_zero PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../"
        LevelZeroLoader-Headers
    )
endif()

if(UR_BUILD_ADAPTER_L0_V2)
    add_ur_adapter(ur_adapter_level_zero_v2
        SHARED
        # sources shared with legacy adapter
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/image_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/memory_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/image_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/memory_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
        # v2-only sources
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_buffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_list_cache.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_list_manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/context.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool_cache.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_counter.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_normal.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/kernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/memory.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_api.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_immediate_in_order.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/usm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_buffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_list_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_list_manager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_counter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_normal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/kernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_create.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_immediate_in_order.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/usm.cpp
    )
    install_ur_library(ur_adapter_level_zero_v2)

    if(NOT WIN32)
        # api.cpp contains NOT_SUPPORTED functions-only
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/v2/api.cpp
            PROPERTIES APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-unused-parameter")

        target_sources(ur_adapter_level_zero_v2
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/adapter_lib_init_linux.cpp
        )
    endif()

    target_compile_definitions(ur_adapter_level_zero_v2 PUBLIC UR_ADAPTER_LEVEL_ZERO_V2)

    # TODO: fix level_zero adapter conversion warnings
    target_compile_options(ur_adapter_level_zero_v2 PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/wd4805 /wd4244>
    )

    set_target_properties(ur_adapter_level_zero_v2 PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
    )

    target_link_libraries(ur_adapter_level_zero_v2 PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
        ComputeRuntimeLevelZero-Headers
    )

    target_include_directories(ur_adapter_level_zero_v2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../.."
        "${CMAKE_CURRENT_SOURCE_DIR}/../../ur"
        LevelZeroLoader-Headers
    )
endif()
