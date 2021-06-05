add_subdirectory(sycl-ls)

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
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:cudadrv>
)
target_compile_definitions(get_device_count_by_type
  PRIVATE
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:USE_PI_CUDA>
)
