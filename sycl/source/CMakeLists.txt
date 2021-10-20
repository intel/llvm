configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.rc.in
  ${CMAKE_CURRENT_BINARY_DIR}/version.rc
  @ONLY)

if (SYCL_ENABLE_XPTI_TRACING)
  if (NOT EXISTS ${LLVM_EXTERNAL_XPTI_SOURCE_DIR})
    message (FATAL_ERROR "Undefined LLVM_EXTERNAL_XPTI_SOURCE_DIR variable: Must be set when XPTI tracing is set to ON")
  endif()
  include_directories(${LLVM_EXTERNAL_XPTI_SOURCE_DIR}/include)
endif()

function(add_sycl_rt_library LIB_NAME)
  # Add an optional argument so we can get the library name to
  # link with for Windows Debug version
  cmake_parse_arguments(ARG "" "XPTI_LIB" "COMPILE_OPTIONS;SOURCES" ${ARGN})

  set(LIB_OBJ_NAME ${LIB_NAME}_object)

  add_sycl_library(${LIB_OBJ_NAME} OBJECT
    SOURCES ${ARG_SOURCES}
    COMPILE_OPTIONS ${ARG_COMPILE_OPTIONS}
    COMPILE_DEFINITIONS
      __SYCL_INTERNAL_API
      __SYCL_BUILD_SYCL_DLL
      SYCL2020_DISABLE_DEPRECATION_WARNINGS
    INCLUDES
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${sycl_inc_dir}"
    LINK
      OpenCL-Headers
    HIDE_SYMBOLS
  )
  add_sycl_library(${LIB_NAME} SHARED
    SOURCES
      $<TARGET_OBJECTS:${LIB_OBJ_NAME}>
      ${CMAKE_CURRENT_BINARY_DIR}/version.rc
    LINK
      ${ARG_XPTI_LIB}
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
      # TODO remove dependency on OpenCL
      OpenCL-ICD
      "$<$<PLATFORM_ID:Windows>:shlwapi>"
    LINKER_SCRIPTS
      "${CMAKE_CURRENT_SOURCE_DIR}/abi_replacements_linux.txt"
    VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt"
  )

  add_dependencies(${LIB_OBJ_NAME}
    sycl-headers
  )

  if (SYCL_ENABLE_XPTI_TRACING)
    target_compile_definitions(${LIB_OBJ_NAME} PRIVATE XPTI_ENABLE_INSTRUMENTATION XPTI_STATIC_LIBRARY)
  endif()

  set_target_properties(${LIB_NAME} PROPERTIES
                        VERSION ${SYCL_VERSION_STRING}
                        SOVERSION ${SYCL_MAJOR_VERSION})

endfunction(add_sycl_rt_library)

set(SYCL_SOURCES
    "backend/opencl.cpp"
    "backend/level_zero.cpp"
    "backend.cpp"
    "detail/accessor_impl.cpp"
    "detail/allowlist.cpp"
    "detail/buffer_impl.cpp"
    "detail/builtins_common.cpp"
    "detail/builtins_geometric.cpp"
    "detail/builtins_integer.cpp"
    "detail/builtins_math.cpp"
    "detail/builtins_relational.cpp"
    "detail/pi.cpp"
    "detail/common.cpp"
    "detail/config.cpp"
    "detail/context_impl.cpp"
    "detail/device_binary_image.cpp"
    "detail/device_filter.cpp"
    "detail/device_impl.cpp"
    "detail/error_handling/enqueue_kernel.cpp"
    "detail/event_impl.cpp"
    "detail/filter_selector_impl.cpp"
    "detail/force_device.cpp"
    "detail/global_handler.cpp"
    "detail/helpers.cpp"
    "detail/handler_proxy.cpp"
    "detail/image_accessor_util.cpp"
    "detail/image_impl.cpp"
    "detail/kernel_impl.cpp"
    "detail/kernel_program_cache.cpp"
    "detail/memory_manager.cpp"
    "detail/platform_impl.cpp"
    "detail/program_impl.cpp"
    "detail/program_manager/program_manager.cpp"
    "detail/queue_impl.cpp"
    "detail/online_compiler/online_compiler.cpp"
    "detail/os_util.cpp"
    "detail/persistent_device_code_cache.cpp"
    "detail/platform_util.cpp"
    "detail/reduction.cpp"
    "detail/sampler_impl.cpp"
    "detail/stream_impl.cpp"
    "detail/scheduler/commands.cpp"
    "detail/scheduler/leaves_collection.cpp"
    "detail/scheduler/scheduler.cpp"
    "detail/scheduler/scheduler_helpers.cpp"
    "detail/scheduler/graph_processor.cpp"
    "detail/scheduler/graph_builder.cpp"
    "detail/spec_constant_impl.cpp"
    "detail/sycl_mem_obj_t.cpp"
    "detail/usm/usm_impl.cpp"
    "detail/util.cpp"
    "accessor.cpp"
    "context.cpp"
    "device.cpp"
    "device_selector.cpp"
    "event.cpp"
    "exception.cpp"
    "exception_list.cpp"
    "function_pointer.cpp"
    "half_type.cpp"
    "handler.cpp"
    "interop_handle.cpp"
    "interop_handler.cpp"
    "kernel.cpp"
    "kernel_bundle.cpp"
    "platform.cpp"
    "program.cpp"
    "queue.cpp"
    "sampler.cpp"
    "stream.cpp"
    "spirv_ops.cpp"
    "$<$<PLATFORM_ID:Windows>:detail/windows_pi.cpp>"
    "$<$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>:detail/posix_pi.cpp>"
    "$<$<PLATFORM_ID:Windows>:abi_replacements_windows.cpp>"
)

if (MSVC)
  # MSVC provides two incompatible build variants for its CRT: release and debug
  # To avoid potential issues in user code we also need to provide two kinds
  # of SYCL Runtime Library for release and debug configurations.
  if (SYCL_ENABLE_XPTI_TRACING)
    add_sycl_rt_library(sycld XPTI_LIB xptid COMPILE_OPTIONS "/MDd" SOURCES ${SYCL_SOURCES})
  else()
    add_sycl_rt_library(sycld COMPILE_OPTIONS "/MDd" SOURCES ${SYCL_SOURCES})
  endif()

  set(SYCL_EXTRA_OPTS "/MD")
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  add_sycl_rt_library(sycl XPTI_LIB xpti COMPILE_OPTIONS ${SYCL_EXTRA_OPTS}  SOURCES ${SYCL_SOURCES})
else()
  add_sycl_rt_library(sycl COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} SOURCES ${SYCL_SOURCES})
endif()

if (libdevice IN_LIST LLVM_ENABLE_PROJECTS)
  add_dependencies(sycl libsycldevice)
endif()

install(TARGETS ${SYCL_RT_LIBS}
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl
  RUNTIME DESTINATION "bin" COMPONENT sycl)
add_llvm_install_targets(install-sycl
  COMPONENT sycl
  DEPENDS ${SYCL_RT_LIBS}
  )

string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lower)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  set(sycl_lib sycld)
endif()

add_library(sycl-runtime INTERFACE)
if (WIN32 AND "${build_type_lower}" MATCHES "debug")
  target_link_libraries(sycl-runtime INTERFACE sycld)
else()
  target_link_libraries(sycl-runtime INTERFACE sycl)
endif()
