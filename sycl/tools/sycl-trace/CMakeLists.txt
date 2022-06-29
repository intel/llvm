add_executable(sycl-trace main.cpp)

target_include_directories(sycl-trace PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../xpti_helpers/"
)

target_compile_options(sycl-trace PRIVATE -fno-exceptions -fno-rtti)

link_llvm_libs(sycl-trace
  LLVMSupport
)

add_library(sycl_pi_trace_collector SHARED
  collector.cpp
  pi_trace_collector.cpp
  ze_trace_collector.cpp
  $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:cuda_trace_collector.cpp>
)

find_package(Python3 REQUIRED)

add_custom_target(pi-pretty-printers
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_pi_pretty_printers.py
  ${sycl_inc_dir}/sycl/detail/pi.h
  SOURCES ${sycl_inc_dir}/sycl/detail/pi.h
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/pi_structs.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/pi_printers.def
  )

# To get L0 loader
add_dependencies(sycl_pi_trace_collector pi_level_zero)

target_link_libraries(sycl_pi_trace_collector PRIVATE LevelZeroLoader::Headers)

add_custom_target(ze-pretty-printers
  COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_ze_pretty_printers.py
  ${SYCL_INCLUDE_BUILD_DIR}/sycl/level_zero/ze_api.h
  DEPENDS pi_level_zero
  BYPRODUCTS
    ${CMAKE_CURRENT_BINARY_DIR}/ze_printers.def
  )

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

add_dependencies(sycl_pi_trace_collector pi-pretty-printers ze-pretty-printers)

if(SYCL_BUILD_PI_CUDA)

  target_compile_definitions(sycl_pi_trace_collector
    PRIVATE
      $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:USE_PI_CUDA>
  )

  find_package(CUDAToolkit 10.1 REQUIRED)
  target_include_directories(sycl_pi_trace_collector
    PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
  )

  target_link_libraries(sycl_pi_trace_collector
    PRIVATE
      cudadrv
      ${CUDA_cupti_LIBRARY}
  )

  find_path (GEN_CUDA_META_H_DIR generated_cuda_meta.h
    PATH
      ${CUDAToolkit_INCLUDE_DIRS}
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
  )

  if( EXISTS "${GEN_CUDA_META_H_DIR}/generated_cuda_meta.h" )
  add_custom_target(cuda-pretty-printers
    COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_cuda_pretty_printers.py
    ${GEN_CUDA_META_H_DIR}/generated_cuda_meta.h
    DEPENDS pi_cuda
    BYPRODUCTS
      ${CMAKE_CURRENT_BINARY_DIR}/cuda_printers.def
    )
  else()
    message(WARNING "generated_cuda_meta.h not FOUND!")
    message(WARNING "CUDA printer definitions cannot be generated.")
    file(TOUCH "${CMAKE_CURRENT_BINARY_DIR}/cuda_printers.def")
    add_custom_target(cuda-pretty-printers)
  endif()

  add_dependencies(sycl_pi_trace_collector cuda-pretty-printers)

endif()

add_dependencies(sycl-trace sycl_pi_trace_collector)
add_dependencies(sycl-toolchain sycl-trace)

include(GNUInstallDirs)
install(TARGETS sycl-trace sycl_pi_trace_collector
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sycl-trace
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sycl-trace
)

