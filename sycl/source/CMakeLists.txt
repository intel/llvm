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

function(add_sycl_rt_library LIB_NAME LIB_OBJ_NAME)
  # Add an optional argument so we can get the library name to
  # link with for Windows Debug version
  cmake_parse_arguments(ARG "" "XPTI_LIB" "COMPILE_OPTIONS;SOURCES" ${ARGN})

  add_library(${LIB_OBJ_NAME} OBJECT ${ARG_SOURCES})
  add_library(${LIB_NAME} SHARED
              $<TARGET_OBJECTS:${LIB_OBJ_NAME}>
              ${CMAKE_CURRENT_BINARY_DIR}/version.rc)

  # Unlike for sycl library, for LLVMSupport we have only one version for a given build,
  # so, we link LLVMSupport lib to matching sycl version only.
  if (SYCL_ENABLE_STACK_PRINTING)
    if(NOT MSVC OR (CMAKE_BUILD_TYPE STREQUAL "Debug" AND ARG_COMPILE_OPTIONS MATCHES ".*MDd.*") OR
      (NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT ARG_COMPILE_OPTIONS MATCHES ".*MDd.*"))
        add_dependencies(${LIB_NAME} LLVMSupport)
        target_compile_definitions(${LIB_OBJ_NAME} PUBLIC ENABLE_STACK_TRACE)
        target_link_libraries(${LIB_NAME} PRIVATE LLVMSupport)
    endif()
  endif()

  # TODO: Enabled for MSVC
  if (NOT MSVC AND SYCL_LIB_WITH_DEBUG_SYMBOLS)
    separate_arguments(CMAKE_CXX_FLAGS_DEBUG_SEPARATED UNIX_COMMAND "${CMAKE_CXX_FLAGS_DEBUG}")
    target_compile_options(${LIB_NAME} PRIVATE ${CMAKE_CXX_FLAGS_DEBUG_SEPARATED})
  endif()

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

  target_include_directories(${LIB_OBJ_NAME} PRIVATE ${BOOST_UNORDERED_INCLUDE_DIRS})

  # pi_win_proxy_loader
  if (WIN32)
    include_directories(${LLVM_EXTERNAL_SYCL_SOURCE_DIR}/pi_win_proxy_loader)
    if(WIN_DUPE)
      target_link_libraries(${LIB_NAME} PUBLIC pi_win_proxy_loaderd)
    else()
      target_link_libraries(${LIB_NAME} PUBLIC pi_win_proxy_loader)
    endif()
  endif()

  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE __SYCL_INTERNAL_API )

  if (WIN32) 
    target_compile_definitions(${LIB_OBJ_NAME} PRIVATE __SYCL_BUILD_SYCL_DLL )
    target_link_libraries(${LIB_NAME} PRIVATE shlwapi)
  endif()

  if (MSVC)
    # Install stripped PDB
    add_stripped_pdb(${LIB_NAME})
  else()
    target_compile_options(${LIB_OBJ_NAME} PUBLIC
                           -fvisibility=hidden -fvisibility-inlines-hidden)
                           
    # Sycl math built-in macros cause a GCC 4.6 'note' to be output repeatedly.
    # => note: the ABI for passing parameters with 32-byte alignment has changed in GCC 4.6
    # Seems to be no way to suppress it except use -Wno-psabi
    target_compile_options(${LIB_OBJ_NAME} PUBLIC -Wno-psabi)

    if (UNIX AND NOT APPLE)
      set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
      target_link_libraries(
          ${LIB_NAME} PRIVATE "-Wl,--version-script=${linker_script}")
      set_target_properties(${LIB_NAME} PROPERTIES LINK_DEPENDS ${linker_script})
    endif()
    if (SYCL_ENABLE_XPTI_TRACING)
      target_link_libraries(${LIB_NAME} PRIVATE ${CMAKE_DL_LIBS})
    endif()
  endif()

  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE SYCL2020_DISABLE_DEPRECATION_WARNINGS)

  target_include_directories(
    ${LIB_OBJ_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${sycl_inc_dir}"
  )
  #TODO: Remove dependency on opencl headers.
  target_link_libraries(${LIB_OBJ_NAME}
    PRIVATE OpenCL-Headers
  )

  if(SYCL_ENABLE_KERNEL_FUSION)
    target_link_libraries(${LIB_NAME} PRIVATE sycl-fusion)
    target_link_libraries(${LIB_OBJ_NAME} PRIVATE sycl-fusion)
    if (BUILD_SHARED_LIBS)
     set_property(GLOBAL APPEND PROPERTY SYCL_TOOLCHAIN_INSTALL_COMPONENTS
       sycl-fusion)
    endif (BUILD_SHARED_LIBS)
  endif(SYCL_ENABLE_KERNEL_FUSION)

  find_package(Threads REQUIRED)

  target_link_libraries(${LIB_NAME}
    PRIVATE
      ${CMAKE_DL_LIBS}
      ${CMAKE_THREAD_LIBS_INIT}
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

set(SYCL_COMMON_SOURCES
    "backend/opencl.cpp"
    "backend/level_zero.cpp"
    "backend.cpp"
    "detail/accessor_impl.cpp"
    "detail/allowlist.cpp"
    "detail/bindless_images.cpp"
    "detail/buffer_impl.cpp"
    "detail/pi.cpp"
    "detail/common.cpp"
    "detail/composite_device/composite_device.cpp"
    "detail/config.cpp"
    "detail/context_impl.cpp"
    "detail/device_binary_image.cpp"
    "detail/device_filter.cpp"
    "detail/host_pipe_map.cpp"
    "detail/device_global_map.cpp"
    "detail/device_global_map_entry.cpp"
    "detail/device_impl.cpp"
    "detail/error_handling/error_handling.cpp"
    "detail/event_impl.cpp"
    "detail/filter_selector_impl.cpp"
    "detail/fusion/fusion_wrapper.cpp"
    "detail/fusion/fusion_wrapper_impl.cpp"
    "detail/global_handler.cpp"
    "detail/graph_impl.cpp"
    "detail/helpers.cpp"
    "detail/handler_proxy.cpp"
    "detail/image_accessor_util.cpp"
    "detail/image_impl.cpp"
    "detail/jit_compiler.cpp"
    "detail/jit_device_binaries.cpp"
    "detail/kernel_compiler/kernel_compiler_opencl.cpp"
    "detail/kernel_impl.cpp"
    "detail/kernel_program_cache.cpp"
    "detail/memory_manager.cpp"
    "detail/pipes.cpp"
    "detail/platform_impl.cpp"
    "detail/program_impl.cpp"
    "detail/program_manager/program_manager.cpp"
    "detail/queue_impl.cpp"
    "detail/online_compiler/online_compiler.cpp"
    "detail/os_util.cpp"
    "detail/persistent_device_code_cache.cpp"
    "detail/platform_util.cpp"
    "detail/preview_marker.cpp"
    "detail/reduction.cpp"
    "detail/sampler_impl.cpp"
    "detail/stream_impl.cpp"
    "detail/scheduler/commands.cpp"
    "detail/scheduler/leaves_collection.cpp"
    "detail/scheduler/scheduler.cpp"
    "detail/scheduler/graph_processor.cpp"
    "detail/scheduler/graph_builder.cpp"
    "detail/spec_constant_impl.cpp"
    "detail/sycl_mem_obj_t.cpp"
    "detail/usm/usm_impl.cpp"
    "detail/util.cpp"
    "detail/xpti_registry.cpp"
    "accessor.cpp"
    "buffer.cpp"
    "context.cpp"
    "device.cpp"
    "device_selector.cpp"
    "event.cpp"
    "exception.cpp"
    "exception_list.cpp"
    "handler.cpp"
    "image.cpp"
    "interop_handle.cpp"
    "kernel.cpp"
    "kernel_bundle.cpp"
    "platform.cpp"
    "queue.cpp"
    "sampler.cpp"
    "stream.cpp"
    "spirv_ops.cpp"
    "esimd_emulator_device_interface.cpp"
    "$<$<PLATFORM_ID:Windows>:detail/windows_pi.cpp>"
    "$<$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>:detail/posix_pi.cpp>"
)

set(SYCL_NON_PREVIEW_SOURCES "${SYCL_COMMON_SOURCES}"
    "detail/builtins_common.cpp"
    "detail/builtins_geometric.cpp"
    "detail/builtins_integer.cpp"
    "detail/builtins_math.cpp"
    "detail/builtins_relational.cpp"
)


set(SYCL_PREVIEW_SOURCES "${SYCL_COMMON_SOURCES}"
    "builtins/common_functions.cpp"
    "builtins/geometric_functions.cpp"
    "builtins/half_precision_math_functions.cpp"
    "builtins/integer_functions.cpp"
    "builtins/math_functions.cpp"
    "builtins/native_math_functions.cpp"
    "builtins/relational_functions.cpp"
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

  set(WIN_DUPE "1")
  if (SYCL_ENABLE_XPTI_TRACING)
    add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}d sycld_object XPTI_LIB xptid COMPILE_OPTIONS "/MDd" SOURCES ${SYCL_NON_PREVIEW_SOURCES})
    if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
      add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}-previewd sycl-previewd_object XPTI_LIB xptid COMPILE_OPTIONS "/MDd" "/D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES})
    endif()
  else()
    add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}d sycld_object COMPILE_OPTIONS "/MDd" SOURCES ${SYCL_NON_PREVIEW_SOURCES})
    if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
      add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}-previewd sycl-previewd_object COMPILE_OPTIONS "/MDd" "/D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES})
    endif()
  endif()
  unset(WIN_DUPE)
  add_library(sycld ALIAS sycl${SYCL_MAJOR_VERSION}d)
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_library(sycl-previewd ALIAS sycl${SYCL_MAJOR_VERSION}-previewd)
  endif()

  set(SYCL_EXTRA_OPTS "/MD")
endif()

check_cxx_compiler_flag(-fsemantic-interposition HAS_SEMANTIC_INTERPOSITION_FLAG)
if (HAS_SEMANTIC_INTERPOSITION_FLAG)
  # See https://github.com/llvm/llvm-project/issues/58295.
  set_source_files_properties(device_selector.cpp PROPERTIES COMPILE_FLAGS -fsemantic-interposition)
endif()

if (WIN32)
set(LIB_NAME "sycl${SYCL_MAJOR_VERSION}")
else()
set(LIB_NAME "sycl")
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  add_sycl_rt_library(${LIB_NAME} sycl_object XPTI_LIB xpti COMPILE_OPTIONS ${SYCL_EXTRA_OPTS}  SOURCES ${SYCL_NON_PREVIEW_SOURCES})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_sycl_rt_library(${LIB_NAME}-preview sycl-preview_object XPTI_LIB xpti COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} "-D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES})
  endif()
else()
  add_sycl_rt_library(${LIB_NAME} sycl_object COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} SOURCES ${SYCL_NON_PREVIEW_SOURCES})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_sycl_rt_library(${LIB_NAME}-preview sycl-preview_object COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} "-D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES})
  endif()
endif()

if (WIN32)
  add_library(sycl ALIAS ${LIB_NAME})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_library(sycl-preview ALIAS sycl${SYCL_MAJOR_VERSION}-preview)
  endif()
endif()

install(TARGETS ${SYCL_RT_LIBS}
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl
  RUNTIME DESTINATION "bin" COMPONENT sycl)
