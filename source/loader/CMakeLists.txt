# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/UrLoaderVersion.rc.in
    ${CMAKE_CURRENT_BINARY_DIR}/UrLoaderVersion.rc
    @ONLY
)

add_library(loader
    SHARED
    ""
    ${CMAKE_CURRENT_BINARY_DIR}/UrLoaderVersion.rc
)

set_target_properties(loader PROPERTIES
    LIBRARY_OUTPUT_NAME ur_loader
    RUNTIME_OUTPUT_NAME ur_loader
)

add_library(${PROJECT_NAME}::loader ALIAS loader)

target_include_directories(loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
)

set_target_properties(loader PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_link_libraries(loader PRIVATE
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers
    ${CMAKE_DL_LIBS}
)

if(UR_ENABLE_TRACING)
    target_link_libraries(loader PRIVATE xpti)
    target_include_directories(loader PRIVATE ${xpti_SOURCE_DIR}/include)
    target_compile_definitions(loader PRIVATE XPTI_STATIC_LIBRARY)
endif()

if (UNIX)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(loader PRIVATE Threads::Threads)
endif()

if(WIN32)
    target_link_libraries(loader PRIVATE cfgmgr32.lib dbghelp)
endif()

if(UNIX)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/libur_loader.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/libur_loader.pc
        @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libur_loader.pc" DESTINATION "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" COMPONENT unified-runtime)
endif()

install(TARGETS loader
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT unified-runtime
)

target_sources(loader
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_object.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_loader.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_ldrddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_ldrddi.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_libapi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_libddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_lib.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_lib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/ur_valddi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/ur_validation_layer.cpp
)

if(UR_ENABLE_TRACING)
    target_sources(loader
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/tracing/ur_tracing_layer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/tracing/ur_trcddi.cpp
    )
endif()

if(WIN32)
    target_sources(loader
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/lib_init.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/platform_discovery_win.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/windows/loader_init.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/backtrace_win.cpp
    )
else()
    target_sources(loader
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/lib_init.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/platform_discovery_lin.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/linux/loader_init.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/layers/validation/backtrace_lin.cpp
    )
endif()
