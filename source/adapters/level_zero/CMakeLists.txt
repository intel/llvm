# Copyright (C) 2022-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_LEVEL_ZERO_LOADER_LIBRARY "" CACHE FILEPATH "Path of the Level Zero Loader library")
set(UR_LEVEL_ZERO_INCLUDE_DIR "" CACHE FILEPATH "Directory containing the Level Zero Headers")
set(UR_LEVEL_ZERO_LOADER_REPO "" CACHE STRING "Github repo to get the Level Zero loader sources from")
set(UR_LEVEL_ZERO_LOADER_TAG "" CACHE STRING " GIT tag of the Level Loader taken from github repo")

# Copy Level Zero loader/headers locally to the build to avoid leaking their path.
set(LEVEL_ZERO_COPY_DIR ${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader)
if (NOT UR_LEVEL_ZERO_LOADER_LIBRARY STREQUAL "")
    get_filename_component(LEVEL_ZERO_LIB_NAME "${UR_LEVEL_ZERO_LOADER_LIBRARY}" NAME)
    set(LEVEL_ZERO_LIBRARY ${LEVEL_ZERO_COPY_DIR}/${LEVEL_ZERO_LIB_NAME})
    message(STATUS "Level Zero Adapter: Copying Level Zero loader to local build tree")
    file(COPY ${UR_LEVEL_ZERO_LOADER_LIBRARY} DESTINATION ${LEVEL_ZERO_COPY_DIR} FOLLOW_SYMLINK_CHAIN)
endif()
if (NOT UR_LEVEL_ZERO_INCLUDE_DIR STREQUAL "")
    set(LEVEL_ZERO_INCLUDE_DIR ${LEVEL_ZERO_COPY_DIR})
    message(STATUS "Level Zero Adapter: Copying Level Zero headers to local build tree")
    file(COPY ${UR_LEVEL_ZERO_INCLUDE_DIR}/ DESTINATION ${LEVEL_ZERO_COPY_DIR})
endif()

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
    message(STATUS "Level Zero Adapter: Download Level Zero loader and headers from github.com")

    # Workaround warnings/errors for Level Zero build
    set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
    if (UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-truncation")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-extra-semi")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
    endif()

    if (UR_LEVEL_ZERO_LOADER_REPO STREQUAL "")
        set(UR_LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
    endif()
    if (UR_LEVEL_ZERO_LOADER_TAG STREQUAL "")
        set(UR_LEVEL_ZERO_LOADER_TAG v1.17.39)
    endif()

    # Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
    # Prevent L0 loader from exporting extra symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

    message(STATUS "Level Zero Adapter: Will fetch Level Zero Loader from ${UR_LEVEL_ZERO_LOADER_REPO}")
    include(FetchContent)
    FetchContent_Declare(level-zero-loader
        GIT_REPOSITORY    ${UR_LEVEL_ZERO_LOADER_REPO}
        GIT_TAG           ${UR_LEVEL_ZERO_LOADER_TAG}
    )
    if(MSVC)
        set(USE_Z7 ON)
    endif()
    FetchContent_MakeAvailable(level-zero-loader)
    FetchContent_GetProperties(level-zero-loader)

    # Restore original flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

    target_compile_options(ze_loader PRIVATE
        $<$<IN_LIST:$<CXX_COMPILER_ID>,GNU;Clang;Intel;IntelLLVM>:-Wno-error>
        $<$<CXX_COMPILER_ID:MSVC>:/WX- /UUNICODE>
    )

    set(LEVEL_ZERO_LIBRARY ze_loader)
    set(LEVEL_ZERO_INCLUDE_DIR
        ${level-zero-loader_SOURCE_DIR}/include CACHE PATH "Path to Level Zero Headers")
endif()

add_library (LevelZeroLoader INTERFACE)
# The MSVC linker does not like / at the start of a path, so to work around this
# we split it into a link library and a library path, where the path is allowed
# to have leading /.
get_filename_component(LEVEL_ZERO_LIBRARY_SRC "${LEVEL_ZERO_LIBRARY}" DIRECTORY)
get_filename_component(LEVEL_ZERO_LIB_NAME "${LEVEL_ZERO_LIBRARY}" NAME)
target_link_directories(LevelZeroLoader
    INTERFACE "${LEVEL_ZERO_LIBRARY_SRC}"
)
target_link_libraries(LevelZeroLoader
    INTERFACE "${LEVEL_ZERO_LIB_NAME}"
)

add_library (LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
    INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)

if(UR_BUILD_ADAPTER_L0)
    add_ur_adapter(ur_adapter_level_zero
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.hpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
    )

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

    if (WIN32)
    # 0x800: Search for the DLL only in the System32 folder
    target_link_options(ur_adapter_level_zero PUBLIC /DEPENDENTLOADFLAG:0x800)
    endif()

    target_link_libraries(ur_adapter_level_zero PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
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
        ${CMAKE_CURRENT_SOURCE_DIR}/program.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/helpers/kernel_helpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
        # v2-only sources
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_list_cache.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/context.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool_cache.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_counter.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_normal.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/kernel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_api.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_immediate_in_order.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/usm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/command_list_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_counter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event_provider_normal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/kernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_create.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/queue_immediate_in_order.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/v2/usm.cpp
    )

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

    if (WIN32)
    # 0x800: Search for the DLL only in the System32 folder
    target_link_options(ur_adapter_level_zero_v2 PUBLIC /DEPENDENTLOADFLAG:0x800)
    endif()

    target_link_libraries(ur_adapter_level_zero_v2 PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        LevelZeroLoader
        LevelZeroLoader-Headers
    )

    target_include_directories(ur_adapter_level_zero_v2 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../.."
        "${CMAKE_CURRENT_SOURCE_DIR}/../../ur"
        LevelZeroLoader-Headers
    )
endif()
