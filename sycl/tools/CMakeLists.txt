add_subdirectory(sycl-ls)

if (SYCL_ENABLE_XPTI_TRACING)
  add_subdirectory(pi-trace)
endif()

# TODO: move each tool in its own sub-directory
add_executable(get_device_count_by_type get_device_count_by_type.cpp)
add_dependencies(get_device_count_by_type
  level-zero-loader
)

if(MSVC)
  set(LEVEL_ZERO_LIBRARY
    "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(LEVEL_ZERO_LIBRARY
    "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ze_loader${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

target_link_libraries(get_device_count_by_type
  PRIVATE
    OpenCL-Headers
    LevelZeroLoader::Headers
    OpenCL-ICD
    ${LEVEL_ZERO_LIBRARY}
    # The CUDA and ROCm for NVIDA plugins need cudadrv
    $<$<OR:$<BOOL:${SYCL_BUILD_PI_CUDA}>,$<AND:$<BOOL:${SYCL_BUILD_PI_ROCM}>,$<STREQUAL:${SYCL_BUILD_PI_ROCM_PLATFORM},NVIDIA>>>:cudadrv>
    # The ROCm for AMD plugin needs rocmdrv
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_ROCM}>,$<STREQUAL:${SYCL_BUILD_PI_ROCM_PLATFORM},AMD>>:rocmdrv>
    # The ROCm for NVIDIA plugin also needs cudart
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_ROCM}>,$<STREQUAL:${SYCL_BUILD_PI_ROCM_PLATFORM},NVIDIA>>:cudart>
)
target_compile_definitions(get_device_count_by_type
  PRIVATE
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:USE_PI_CUDA>
    $<$<BOOL:${SYCL_BUILD_PI_ROCM}>:USE_PI_ROCM>
    # For ROCm set the HIP define depending on the platform
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_ROCM}>,$<STREQUAL:${SYCL_BUILD_PI_ROCM_PLATFORM},AMD>>:__HIP_PLATFORM_AMD__>
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_ROCM}>,$<STREQUAL:${SYCL_BUILD_PI_ROCM_PLATFORM},NVIDIA>>:__HIP_PLATFORM_NVIDIA__>
)

if(SYCL_BUILD_PI_ROCM)
  target_include_directories(get_device_count_by_type
    PRIVATE
    ${SYCL_BUILD_PI_ROCM_INCLUDE_DIR})
endif()
