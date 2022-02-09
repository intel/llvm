#To-Do:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.rc.in
  ${CMAKE_CURRENT_BINARY_DIR}/version.rc
  @ONLY)

if (SYCL_ENABLE_XPTI_TRACING)
  if (NOT DEFINED LLVM_EXTERNAL_XPTI_SOURCE_DIR)
    message (FATAL_ERROR "Undefined LLVM_EXTERNAL_XPTI_SOURCE_DIR variable: Must be set when XPTI tracing is set to ON")
  endif()
  include_directories(${LLVM_EXTERNAL_XPTI_SOURCE_DIR}/include)
endif()

function(add_sycl_rt_library LIB_NAME)
  # Add an optional argument so we can get the library name to
  # link with for Windows Debug version
  cmake_parse_arguments(ARG "" "XPTI_LIB" "COMPILE_OPTIONS;SOURCES" ${ARGN})

  set(LIB_OBJ_NAME ${LIB_NAME}_object)

  add_library(${LIB_OBJ_NAME} OBJECT ${ARG_SOURCES})
  add_library(${LIB_NAME} SHARED
              $<TARGET_OBJECTS:${LIB_OBJ_NAME}>
              ${CMAKE_CURRENT_BINARY_DIR}/version.rc)

  if (SYCL_ENABLE_COVERAGE)
    target_compile_options(${LIB_OBJ_NAME} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
    target_compile_options(${LIB_NAME} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
    target_link_options(${LIB_NAME} PUBLIC
      -fprofile-instr-generate -fcoverage-mapping
    )
  endif()

  if (ARG_COMPILE_OPTIONS)
    target_compile_options(${LIB_OBJ_NAME} PRIVATE ${ARG_COMPILE_OPTIONS})
  endif()

  add_dependencies(${LIB_OBJ_NAME}
    sycl-headers
  )

  set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  if (SYCL_ENABLE_XPTI_TRACING)
    target_compile_definitions(${LIB_OBJ_NAME} PRIVATE XPTI_ENABLE_INSTRUMENTATION XPTI_STATIC_LIBRARY)
    target_link_libraries(${LIB_NAME} PRIVATE ${ARG_XPTI_LIB})
  endif()

  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE __SYCL_INTERNAL_API )

  if (MSVC)
    target_compile_definitions(${LIB_OBJ_NAME} PRIVATE __SYCL_BUILD_SYCL_DLL )
    target_link_libraries(${LIB_NAME} PRIVATE shlwapi)
    # Install stripped PDB
    add_stripped_pdb(${LIB_NAME})
  else()
    target_compile_options(${LIB_OBJ_NAME} PUBLIC
                           -fvisibility=hidden -fvisibility-inlines-hidden)
    set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
    set(abi_linker_script "${CMAKE_CURRENT_SOURCE_DIR}/abi_replacements_linux.txt")
    target_link_libraries(
      ${LIB_NAME} PRIVATE "-Wl,${abi_linker_script}")
    set_target_properties(${LIB_NAME} PROPERTIES LINK_DEPENDS ${abi_linker_script})
    target_link_libraries(
        ${LIB_NAME} PRIVATE "-Wl,--version-script=${linker_script}")
    set_target_properties(${LIB_NAME} PROPERTIES LINK_DEPENDS ${linker_script})
    if (SYCL_ENABLE_XPTI_TRACING)
      target_link_libraries(${LIB_NAME} PRIVATE dl)
    endif()
  endif()

  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE SYCL2020_DISABLE_DEPRECATION_WARNINGS)

  target_include_directories(
    ${LIB_OBJ_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${sycl_inc_dir}"
  )
  #TODO: Remove dependency on icd loader and opencl headers.
  target_link_libraries(${LIB_OBJ_NAME}
    PRIVATE OpenCL-Headers
  )

  find_package(Threads REQUIRED)

  target_link_libraries(${LIB_NAME}
    PRIVATE
      OpenCL-ICD
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
    PUBLIC
      $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:pi_cuda>
  )

  add_common_options(${LIB_NAME} ${LIB_OBJ_NAME})

  set_target_properties(${LIB_NAME} PROPERTIES
                        VERSION ${SYCL_VERSION_STRING}
                        SOVERSION ${SYCL_MAJOR_VERSION})

  check_cxx_compiler_flag(-Winstantiation-after-specialization
    HAS_INST_AFTER_SPEC)
  if (HAS_INST_AFTER_SPEC)
    target_compile_options(${LIB_OBJ_NAME} PRIVATE
      -Winstantiation-after-specialization)
  endif()
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
    "detail/xpti_registry.cpp"
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
    "esimd_emulator_device_interface.cpp"
    "$<$<PLATFORM_ID:Windows>:detail/windows_pi.cpp>"
    "$<$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>:detail/posix_pi.cpp>"
    "$<$<PLATFORM_ID:Windows>:abi_replacements_windows.cpp>"
)

if (MSVC)
  # MSVC provides two incompatible build variants for its CRT: release and debug
  # To avoid potential issues in user code we also need to provide two kinds
  # of SYCL Runtime Library for release and debug configurations.
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    string(REGEX REPLACE "/MDd" "" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/MTd" "" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/MD" "" ${flag_var} "${${flag_var}}")
    string(REGEX REPLACE "/MT" "" ${flag_var} "${${flag_var}}")
  endforeach()

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

install(TARGETS ${SYCL_RT_LIBS}
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl
  RUNTIME DESTINATION "bin" COMPONENT sycl)
