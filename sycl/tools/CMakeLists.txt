function(link_llvm_libs target)
  foreach(LIB ${ARGN})
    target_include_directories(${target} SYSTEM PRIVATE ${LLVM_MAIN_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${LIB})
  endforeach()
endfunction()

add_subdirectory(sycl-ls)

if (SYCL_ENABLE_XPTI_TRACING)
  if (UNIX)
    add_subdirectory(sycl-prof)
    add_subdirectory(sycl-trace)
    add_subdirectory(sycl-sanitize)
  endif()
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
    # The CUDA and HIP for NVIDA plugins need cudadrv
    $<$<OR:$<BOOL:${SYCL_BUILD_PI_CUDA}>,$<AND:$<BOOL:${SYCL_BUILD_PI_HIP}>,$<STREQUAL:${SYCL_BUILD_PI_HIP_PLATFORM},NVIDIA>>>:cudadrv>
    # The HIP for AMD plugin needs rocmdrv
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_HIP}>,$<STREQUAL:${SYCL_BUILD_PI_HIP_PLATFORM},AMD>>:rocmdrv>
    # The HIP for NVIDIA plugin also needs cudart
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_HIP}>,$<STREQUAL:${SYCL_BUILD_PI_HIP_PLATFORM},NVIDIA>>:cudart>
)
target_compile_definitions(get_device_count_by_type
  PRIVATE
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:USE_PI_CUDA>
    $<$<BOOL:${SYCL_BUILD_PI_HIP}>:USE_PI_HIP>
    # For HIP set defines depending on the platform
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_HIP}>,$<STREQUAL:${SYCL_BUILD_PI_HIP_PLATFORM},AMD>>:__HIP_PLATFORM_AMD__>
    $<$<AND:$<BOOL:${SYCL_BUILD_PI_HIP}>,$<STREQUAL:${SYCL_BUILD_PI_HIP_PLATFORM},NVIDIA>>:__HIP_PLATFORM_NVIDIA__>
)

if(SYCL_BUILD_PI_HIP)
  target_include_directories(get_device_count_by_type
    PRIVATE
    ${SYCL_BUILD_PI_HIP_INCLUDE_DIR})
endif()
