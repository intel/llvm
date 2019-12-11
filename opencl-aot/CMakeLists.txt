cmake_minimum_required(VERSION 3.2)

# Supported configurations
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(ExternalProject)

if (NOT OpenCL_INCLUDE_DIRS)
    message("OpenCL_INCLUDE_DIRS is missed. Will try to download OpenCL headers from github.com")
    set(OpenCL_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc")
    ExternalProject_Add(opencl-headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
            GIT_TAG origin/master
            SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${CMAKE_CURRENT_BINARY_DIR}/inc/CL
            INSTALL_COMMAND ""
            STEP_TARGETS build
            COMMENT "Downloading OpenCL headers."
            )
else ()
    add_custom_target(opencl-headers ALL
            DEPENDS ${OpenCL_INCLUDE_DIRS}
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${CMAKE_CURRENT_BINARY_DIR}/inc/CL
            COMMENT "Copying OpenCL headers ..."
            )
endif ()

if (NOT OpenCL_LIBRARIES)
    message("OpenCL_LIBRARIES is missed. Will try to download OpenCL ICD Loader from github.com")
    if (MSVC)
        set(OpenCL_LIBRARIES
                "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}OpenCL${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(OPENCL_ICD_LOADER_WDK "-DOPENCL_ICD_LOADER_REQUIRE_WDK=OFF")
    else ()
        set(OpenCL_LIBRARIES
                "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}OpenCL${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(OPENCL_ICD_LOADER_WDK "")
    endif ()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/icd_build)
    ExternalProject_Add(opencl-icd
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
            GIT_TAG origin/master
            SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/icd"
            BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/icd_build"
            CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DOPENCL_ICD_LOADER_HEADERS_DIR=${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc
            -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            ${OPENCL_ICD_LOADER_WDK}
            STEP_TARGETS configure,build,install
            DEPENDS opencl-headers
            BUILD_BYPRODUCTS ${OpenCL_LIBRARIES}
            )
else ()
    file(GLOB ICD_LOADER_SRC "${OpenCL_LIBRARIES}*")
    file(COPY ${ICD_LOADER_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_target(opencl-icd DEPENDS ${OpenCL_LIBRARIES} COMMENT "Copying OpenCL ICD Loader ...")
endif ()

set(LLVM_LINK_COMPONENTS
        ${LLVM_TARGETS_TO_BUILD}
        Support
        )
include_directories(${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
link_directories(${CMAKE_CURRENT_BINARY_DIR}/lib
        )
file(GLOB TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

add_llvm_tool(opencl-aot ${TARGET_SOURCES})

add_dependencies(opencl-aot opencl-headers opencl-icd)

target_link_libraries(opencl-aot PRIVATE OpenCL)

install(FILES ${CMAKE_BINARY_DIR}/bin/opencl-aot${CMAKE_EXECUTABLE_SUFFIX}
        DESTINATION bin
        COMPONENT opencl-aot
        )
