set(UR_OFFLOAD_ADAPTER_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Offload adapter directory") # TODO

set(TARGET_NAME ur_adapter_offload)

set(UR_OFFLOAD_INSTALL_DIR "" CACHE PATH "Path to the directory containing libomptarget.so etc")
if (UR_OFFLOAD_INSTALL_DIR STREQUAL "")
    message(FATAL_ERROR "UR_OFFLOAD_INSTALL_DIR must be defined for the Offload adapter")
endif()

set(UR_OFFLOAD_INCLUDE_DIR "" CACHE PATH "Path to the directory containing LLVM headers")
if (UR_OFFLOAD_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR "UR_OFFLOAD_INCLUDE_DIR must be defined for the Offload adapter")
endif()

# For the PTX workaround we need to link with CUDA.
if (NOT TARGET cudadrv)
        find_package(CUDA 10.1 REQUIRED)
        add_library(cudadrv SHARED IMPORTED GLOBAL)
        set_target_properties(
                cudadrv PROPERTIES 
                IMPORTED_LOCATION             ${CUDA_CUDA_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
        )
endif()

add_ur_adapter(${TARGET_NAME}
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur2offload.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_link_libraries(${TARGET_NAME} PRIVATE
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        ${PROJECT_NAME}::umf
        ${UR_OFFLOAD_INSTALL_DIR}/lib/libLLVMOffload.so
        cudadrv
)

target_include_directories(${TARGET_NAME} PRIVATE
        "${UR_OFFLOAD_INCLUDE_DIR}/offload"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)
