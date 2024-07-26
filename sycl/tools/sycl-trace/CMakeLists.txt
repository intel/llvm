add_executable(sycl-trace main.cpp)

target_include_directories(sycl-trace PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
)

target_compile_options(sycl-trace PRIVATE -fno-exceptions -fno-rtti)

link_llvm_libs(sycl-trace
  LLVMSupport
)

if ("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  add_library(ze_trace_collector SHARED
    ze_trace_collector.cpp
  )
  set(EXTRA_TARGETS_TO_INSTALL
    ze_trace_collector
  )
  add_dependencies(sycl-trace ze_trace_collector)
endif()

if ("cuda" IN_LIST SYCL_ENABLE_PLUGINS)
  add_library(cuda_trace_collector SHARED
    cuda_trace_collector.cpp
  )
  set(EXTRA_TARGETS_TO_INSTALL
    ${EXTRA_TARGETS_TO_INSTALL}
    cuda_trace_collector
  )
  add_dependencies(sycl-trace cuda_trace_collector)
endif()

add_library(sycl_pi_trace_collector SHARED
  collector.cpp
  pi_trace_collector.cpp
  sycl_trace_collector.cpp
  verification_collector.cpp
)

find_package(Python3 REQUIRED)

add_custom_target(pi-pretty-printers
  DEPENDS 
  ${CMAKE_CURRENT_BINARY_DIR}/pi_printers.def
  ${CMAKE_CURRENT_BINARY_DIR}/pi_structs.hpp
  )

add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/pi_printers.def
  ${CMAKE_CURRENT_BINARY_DIR}/pi_structs.hpp
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_pi_pretty_printers.py
  ${sycl_inc_dir}/sycl/detail/pi.h
  DEPENDS ${sycl_inc_dir}/sycl/detail/pi.h
  )

# To get L0 loader
if ("level_zero" IN_LIST SYCL_ENABLE_PLUGINS)
  add_dependencies(ze_trace_collector pi_level_zero)

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
    pi_level_zero
    ${LEVEL_ZERO_INCLUDE_DIR}/ze_api.h
    )

  add_dependencies(ze_trace_collector ze-pretty-printers)
  target_compile_definitions(sycl_pi_trace_collector PRIVATE SYCL_HAS_LEVEL_ZERO)
endif()

target_compile_definitions(sycl_pi_trace_collector PRIVATE XPTI_CALLBACK_API_EXPORTS)
target_link_libraries(sycl_pi_trace_collector PRIVATE xptifw)
if (TARGET OpenCL-Headers)
  target_link_libraries(sycl_pi_trace_collector PRIVATE OpenCL-Headers)
endif()

target_include_directories(sycl_pi_trace_collector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
    "${sycl_inc_dir}"
    "${sycl_src_dir}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

add_dependencies(sycl_pi_trace_collector pi-pretty-printers)

if(SYCL_BUILD_BACKEND_CUDA)

  find_package(CUDA 10.1 REQUIRED)

  target_compile_definitions(cuda_trace_collector
    PRIVATE
      $<$<BOOL:${SYCL_BUILD_BACKEND_CUDA}>:USE_PI_CUDA>
  )

  # The following two if's can be removed when FindCUDA -> FindCUDAToolkit.
  # CUDA_CUPTI_INCLUDE_DIR -> CUDAToolkit_CUPTI_INCLUDE_DIR
  include(FindCUDACupti)
  if(NOT CUDA_CUPTI_INCLUDE_DIR)
     find_cuda_cupti_include_dir()
  endif()
  # CUDA_cupti_LIBRARY -> CUDAToolkit_cupti_LIBRARY
  if(NOT CUDA_cupti_LIBRARY)
     find_cuda_cupti_library()
  endif()

  target_include_directories(cuda_trace_collector
    PRIVATE
    ${CUDA_CUPTI_INCLUDE_DIR}
  )

  target_link_libraries(cuda_trace_collector
    PRIVATE
      cudadrv
      ${CUDA_cupti_LIBRARY}
  )

  find_path(GEN_CUDA_META_H_DIR generated_cuda_meta.h PATHS
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
      "${CUDA_INCLUDE_DIRS}/../extras/CUPTI/include"
      "${CUDA_INCLUDE_DIRS}"
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
      pi_cuda
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
  target_compile_definitions(sycl_pi_trace_collector
    PRIVATE
      $<$<BOOL:${SYCL_BUILD_BACKEND_CUDA}>:USE_PI_CUDA>
  )
endif()

add_dependencies(sycl-trace sycl_pi_trace_collector)
add_dependencies(sycl-toolchain sycl-trace)

include(GNUInstallDirs)
install(TARGETS sycl-trace sycl_pi_trace_collector ${EXTRA_TARGETS_TO_INSTALL}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-trace
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
)

