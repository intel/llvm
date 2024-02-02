# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


set(TARGET_NAME ur_adapter_level_zero)

# Copy L0 loader/headers locally to the build to avoid leaking their path.
set(LEVEL_ZERO_COPY_DIR ${CMAKE_CURRENT_BINARY_DIR}/level_zero_loader)
if (DEFINED L0_LIBRARY)
  get_filename_component(LEVEL_ZERO_LIB_NAME "${L0_LIBRARY}" NAME)
  set(LEVEL_ZERO_LIBRARY ${LEVEL_ZERO_COPY_DIR}/${LEVEL_ZERO_LIB_NAME})
  message(STATUS "Copying Level Zero loader and headers to local build tree")
  file(COPY ${L0_LIBRARY} DESTINATION ${LEVEL_ZERO_COPY_DIR} FOLLOW_SYMLINK_CHAIN)
endif()
if (DEFINED L0_INCLUDE_DIR)
  set(LEVEL_ZERO_INCLUDE_DIR ${LEVEL_ZERO_COPY_DIR}/level_zero)
  file(COPY ${L0_INCLUDE_DIR}/level_zero DESTINATION ${LEVEL_ZERO_COPY_DIR})
endif()

if (NOT DEFINED LEVEL_ZERO_LIBRARY OR NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
    message(STATUS "Download Level Zero loader and headers from github.com")

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

    set(LEVEL_ZERO_LOADER_REPO "https://github.com/oneapi-src/level-zero.git")
    set(LEVEL_ZERO_LOADER_TAG v1.15.1)

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

add_ur_adapter(${TARGET_NAME}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ur_level_zero.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
)

# TODO: fix level_zero adapter conversion warnings
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4805 /wd4244>
)

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

if (WIN32)
 # 0x800: Search for the DLL only in the System32 folder
 target_link_options(ur_adapter_level_zero PUBLIC /DEPENDENTLOADFLAG:0x800)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::unified_malloc_framework
    LevelZeroLoader
    LevelZeroLoader-Headers
)

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
    LevelZeroLoader-Headers
)
