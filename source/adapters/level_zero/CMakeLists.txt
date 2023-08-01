# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(L0_DIR "${SYCL_ADAPTER_DIR}/sycl/plugins/unified_runtime/ur/adapters/level_zero")

set(TARGET_NAME ur_adapter_level_zero)

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
    message(STATUS "Download Level Zero loader and headers from github.com")

    set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
    set(LEVEL_ZERO_LOADER_TAG v1.11.0)

    # Disable due to a bug https://github.com/oneapi-src/level-zero/issues/104
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
    # Prevent L0 loader from exporting extra symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

    message(STATUS "Will fetch Level Zero Loader from ${LEVEL_ZERO_LOADER_REPO}")
    include(FetchContent)
    FetchContent_Declare(level-zero-loader
        GIT_REPOSITORY    ${LEVEL_ZERO_LOADER_REPO}
        GIT_TAG           ${LEVEL_ZERO_LOADER_TAG}
    )
    set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
    if(MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX-")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX-")
      # FIXME: Unified runtime build fails with /DUNICODE
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /UUNICODE")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /UUNICODE")
      # USE_Z7 forces use of /Z7 instead of /Zi which is broken with sccache
      set(USE_Z7 ON)
    endif()
    FetchContent_MakeAvailable(level-zero-loader)
    FetchContent_GetProperties(level-zero-loader)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

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

if (NOT MSVC)
    target_compile_options(${LEVEL_ZERO_LIB_NAME} PUBLIC
        -Wno-unused-but-set-variable
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-error
    )
endif()

add_library (LevelZeroLoader-Headers INTERFACE)
target_include_directories(LevelZeroLoader-Headers
    INTERFACE "${LEVEL_ZERO_INCLUDE_DIR}"
)

add_ur_adapter(${TARGET_NAME}
    SHARED
    ${L0_DIR}/ur_interface_loader.cpp
    ${L0_DIR}/adapter.hpp
    ${L0_DIR}/adapter.cpp
    ${L0_DIR}/command_buffer.hpp
    ${L0_DIR}/command_buffer.cpp
    ${L0_DIR}/common.hpp
    ${L0_DIR}/context.hpp
    ${L0_DIR}/device.hpp
    ${L0_DIR}/event.hpp
    ${L0_DIR}/usm.hpp
    ${L0_DIR}/memory.hpp
    ${L0_DIR}/kernel.hpp
    ${L0_DIR}/platform.hpp
    ${L0_DIR}/program.hpp
    ${L0_DIR}/queue.hpp
    ${L0_DIR}/sampler.hpp
    ${L0_DIR}/ur_level_zero.cpp
    ${L0_DIR}/common.cpp
    ${L0_DIR}/context.cpp
    ${L0_DIR}/device.cpp
    ${L0_DIR}/event.cpp
    ${L0_DIR}/usm.cpp
    ${L0_DIR}/usm_p2p.cpp
    ${L0_DIR}/memory.cpp
    ${L0_DIR}/kernel.cpp
    ${L0_DIR}/platform.cpp
    ${L0_DIR}/program.cpp
    ${L0_DIR}/queue.cpp
    ${L0_DIR}/sampler.cpp
    ${L0_DIR}/../../ur.cpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::unified_malloc_framework
    LevelZeroLoader
    LevelZeroLoader-Headers
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${L0_DIR}/../../../
    LevelZeroLoader-Headers
)
