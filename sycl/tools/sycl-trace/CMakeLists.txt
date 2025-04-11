add_executable(sycl-trace main.cpp)

target_include_directories(sycl-trace PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
)

target_compile_options(sycl-trace PRIVATE -fno-exceptions -fno-rtti)

link_llvm_libs(sycl-trace
  LLVMSupport
)

if ("level_zero" IN_LIST SYCL_ENABLE_BACKENDS)
  add_library(ze_trace_collector SHARED
    ze_trace_collector.cpp
  )
  set(EXTRA_TARGETS_TO_INSTALL
    ze_trace_collector
  )
  add_dependencies(sycl-trace ze_trace_collector)
endif()

if ("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  add_library(cuda_trace_collector SHARED
    cuda_trace_collector.cpp
  )
  set(EXTRA_TARGETS_TO_INSTALL
    ${EXTRA_TARGETS_TO_INSTALL}
    cuda_trace_collector
  )
  add_dependencies(sycl-trace cuda_trace_collector)
endif()

add_library(sycl_ur_trace_collector SHARED
  collector.cpp
  sycl_trace_collector.cpp
  ur_trace_collector.cpp
  verification_collector.cpp
)

find_package(Python3 REQUIRED)

# To get L0 loader
if ("level_zero" IN_LIST SYCL_ENABLE_BACKENDS)
  target_link_libraries(ze_trace_collector PRIVATE LevelZeroLoader-Headers)
  target_compile_definitions(ze_trace_collector PRIVATE SYCL_HAS_LEVEL_ZERO)
  target_link_libraries(ze_trace_collector PRIVATE xptifw)

  add_custom_target(ze-pretty-printers
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ze_printers.def
    )
  add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/ze_printers.def
    COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_ze_pretty_printers.py
    ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
    DEPENDS
    ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
    )

  add_custom_target(ze-api DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ze_api.def)
  add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/ze_api.def
    COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/ze_api_generator.py
    ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
    DEPENDS
    ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
    )

  add_dependencies(ze_trace_collector ze-api ze-pretty-printers)
  target_compile_definitions(sycl_ur_trace_collector PRIVATE SYCL_HAS_LEVEL_ZERO)
endif()

target_compile_definitions(sycl_ur_trace_collector PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(sycl_ur_trace_collector PRIVATE xptifw)
if (TARGET OpenCL-Headers)
  target_link_libraries(sycl_ur_trace_collector PRIVATE OpenCL-Headers)
endif()
target_link_libraries(sycl_ur_trace_collector PRIVATE UnifiedRuntime-Headers)

target_include_directories(sycl_ur_trace_collector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
    "${sycl_inc_dir}"
    "${sycl_src_dir}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

if(SYCL_BUILD_BACKEND_CUDA)

  find_package(CUDAToolkit 10.1 REQUIRED)

  target_compile_definitions(cuda_trace_collector PRIVATE USE_UR_CUDA)

  # CUDAToolkit_CUPTI_INCLUDE_DIR is provided by FindCUDAToolkit since CMake
  # 3.24, but we support versions back to 3.20. Find it if not found already.
  if (NOT CUDAToolkit_CUPTI_INCLUDE_DIR)
    find_path(CUDAToolkit_CUPTI_INCLUDE_DIR cupti.h PATHS
      "${CUDAToolkit_ROOT_DIR}/extras/CUPTI/include"
      ${CUDAToolkit_INCLUDE_DIRS}
      PATH_SUFFIXES "../extras/CUPTI/include"
                    "../../../extras/CUPTI/include"
      REQUIRED
      NO_DEFAULT_PATH)
  endif()

  target_include_directories(cuda_trace_collector
    PRIVATE
    ${CUDAToolkit_CUPTI_INCLUDE_DIR}
  )

  # FindCUDATookit fails to find CUPTI in CUDA installations from runfile.
  # See https://gitlab.kitware.com/cmake/cmake/-/issues/26770
  # Retry the search with extra paths if not found already.
  if (NOT CUDA_cupti_LIBRARY)
    get_filename_component(CUDAToolkit_LIBRARY_DIR ${CUDA_CUDART} DIRECTORY ABSOLUTE)
    find_library(CUDA_cupti_LIBRARY
      NAMES cupti
      HINTS ${CUDAToolkit_ROOT}
            ${CUDAToolkit_ROOT}/lib64
            ${CUDAToolkit_ROOT}/lib
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_LIBRARY_DIR}/..
            ${CUDAToolkit_LIBRARY_DIR}/../../..
            ${CUDAToolkit_LIBRARY_DIR}/../../../../..
            ENV CUDA_PATH
      PATH_SUFFIXES lib64 lib/x64 lib
                    extras/CUPTI/lib64/
                    extras/CUPTI/lib/
      REQUIRED
    )
  endif()
  target_link_libraries(cuda_trace_collector
    PRIVATE
      cudadrv
      ${CUDA_cupti_LIBRARY}
  )

  find_path(GEN_CUDA_META_H_DIR generated_cuda_meta.h PATHS
      "${CUDAToolkit_ROOT}/extras/CUPTI/include"
      "${CUDAToolkit_INCLUDE_DIRS}/../extras/CUPTI/include"
      "${CUDAToolkit_INCLUDE_DIRS}"
      NO_DEFAULT_PATH)

  if( EXISTS "${GEN_CUDA_META_H_DIR}/generated_cuda_meta.h" )
    add_custom_target(cuda-pretty-printers
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_printers.def)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_printers.def
      COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/generate_cuda_pretty_printers.py
      ${GEN_CUDA_META_H_DIR}/generated_cuda_meta.h
      DEPENDS
      ${GEN_CUDA_META_H_DIR}/generated_cuda_meta.h
    )
  else()
    message(WARNING "generated_cuda_meta.h not FOUND!")
    message(WARNING "CUDA printer definitions cannot be generated.")
    file(TOUCH "${CMAKE_CURRENT_BINARY_DIR}/cuda_printers.def")
    add_custom_target(cuda-pretty-printers)
  endif()

  add_dependencies(cuda_trace_collector cuda-pretty-printers)
  target_link_libraries(cuda_trace_collector PRIVATE xptifw)
  target_compile_definitions(sycl_ur_trace_collector PRIVATE USE_UR_CUDA)
endif()

add_dependencies(sycl-trace sycl_ur_trace_collector)
add_dependencies(sycl-toolchain sycl-trace)

include(GNUInstallDirs)
install(TARGETS sycl-trace sycl_ur_trace_collector ${EXTRA_TARGETS_TO_INSTALL}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-trace
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
)