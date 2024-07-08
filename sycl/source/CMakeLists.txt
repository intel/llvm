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
  cmake_parse_arguments(ARG "" "XPTI_LIB;IMPLIB_NAME" "COMPILE_OPTIONS;SOURCES" ${ARGN})

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
      set(MANIFEST_FILE_NAME "sycld.manifest")
    else()
      target_link_libraries(${LIB_NAME} PUBLIC pi_win_proxy_loader)
      set(MANIFEST_FILE_NAME "sycl.manifest")
    endif()
    # Embed manifest into the sycl.dll where pi_win_proxy_loader.dll is described as sycl.dll's private dll and will always be loaded from the same directory.
    # 0x2000: LOAD_LIBRARY_SAFE_CURRENT_DIRS flag. Using this flag means that loading dependency DLLs (of sycl.dll)
    # from the current directory is only allowed if it is under a directory in the Safe load list.
    target_link_options(${LIB_NAME} PRIVATE /DEPENDENTLOADFLAG:0x2000 /MANIFEST:NO /MANIFEST:EMBED /MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/${MANIFEST_FILE_NAME})
  endif()

  target_compile_definitions(${LIB_OBJ_NAME} PRIVATE __SYCL_INTERNAL_API )

  if (WIN32) 
    target_compile_definitions(${LIB_OBJ_NAME} PRIVATE __SYCL_BUILD_SYCL_DLL )
    target_link_libraries(${LIB_NAME} PRIVATE shlwapi)
    if (ARG_IMPLIB_NAME)
      add_custom_command(
        TARGET ${LIB_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${LIB_NAME}.lib ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ARG_IMPLIB_NAME}.lib
        COMMENT "Creating version-agnostic copy of the import library.")
      install(
        FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${ARG_IMPLIB_NAME}.lib
        DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl)
    endif()
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
    if(NOT DEFINED LLVM_EXTERNAL_SYCL_FUSION_SOURCE_DIR)
      message(FATAL_ERROR "Undefined LLVM_EXTERNAL_SYCL_FUSION_SOURCE_DIR variable: Must be set when SYCL fusion is enabled")
    endif()
    set(SYCL_FUSION_INCLUDE_DIRS 
        ${LLVM_EXTERNAL_SYCL_FUSION_SOURCE_DIR}/common/include
        ${LLVM_EXTERNAL_SYCL_FUSION_SOURCE_DIR}/jit-compiler/include)
    add_dependencies(${LIB_NAME} sycl-fusion)
    add_dependencies(${LIB_OBJ_NAME} sycl-fusion)
    target_include_directories(${LIB_NAME} PRIVATE ${SYCL_FUSION_INCLUDE_DIRS})
    target_include_directories(${LIB_OBJ_NAME} PRIVATE ${SYCL_FUSION_INCLUDE_DIRS})
    set_property(GLOBAL APPEND PROPERTY SYCL_TOOLCHAIN_INSTALL_COMPONENTS
       sycl-fusion)
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
    "builtins/common_functions.cpp"
    "builtins/geometric_functions.cpp"
    "builtins/half_precision_math_functions.cpp"
    "builtins/integer_functions.cpp"
    "builtins/math_functions.cpp"
    "builtins/native_math_functions.cpp"
    "builtins/relational_functions.cpp"
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
    "detail/kernel_compiler/kernel_compiler_sycl.cpp"
    "detail/kernel_impl.cpp"
    "detail/kernel_program_cache.cpp"
    "detail/memory_manager.cpp"
    "detail/pipes.cpp"
    "detail/platform_impl.cpp"
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
    "enqueue_functions.cpp"
    "event.cpp"
    "exception.cpp"
    "exception_list.cpp"
    "handler.cpp"
    "image.cpp"
    "interop_handle.cpp"
    "kernel.cpp"
    "kernel_bundle.cpp"
    "physical_mem.cpp"
    "platform.cpp"
    "queue.cpp"
    "sampler.cpp"
    "stream.cpp"
    "spirv_ops.cpp"
    "virtual_mem.cpp"
    "$<$<PLATFORM_ID:Windows>:detail/windows_pi.cpp>"
    "$<$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>:detail/posix_pi.cpp>"
)

set(SYCL_NON_PREVIEW_SOURCES "${SYCL_COMMON_SOURCES}"
)


set(SYCL_PREVIEW_SOURCES "${SYCL_COMMON_SOURCES}"
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
    add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}d sycld_object XPTI_LIB xptid COMPILE_OPTIONS "/MDd" SOURCES ${SYCL_NON_PREVIEW_SOURCES} IMPLIB_NAME sycld)
    if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
      add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}-previewd sycl-previewd_object XPTI_LIB xptid COMPILE_OPTIONS "/MDd" "/D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES} IMPLIB_NAME sycl-previewd)
    endif()
  else()
    add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}d sycld_object COMPILE_OPTIONS "/MDd" SOURCES ${SYCL_NON_PREVIEW_SOURCES} IMPLIB_NAME sycld)
    if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
      add_sycl_rt_library(sycl${SYCL_MAJOR_VERSION}-previewd sycl-previewd_object COMPILE_OPTIONS "/MDd" "/D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES} IMPLIB_NAME sycl-previewd)
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
# Version-agnostic name of the import library, has effect on Windows only.
set(IMPLIB_NAME "sycl")

if (SYCL_ENABLE_XPTI_TRACING)
  add_sycl_rt_library(${LIB_NAME} sycl_object XPTI_LIB xpti COMPILE_OPTIONS ${SYCL_EXTRA_OPTS}  SOURCES ${SYCL_NON_PREVIEW_SOURCES} IMPLIB_NAME ${IMPLIB_NAME})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_sycl_rt_library(${LIB_NAME}-preview sycl-preview_object XPTI_LIB xpti COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} "-D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES} IMPLIB_NAME ${IMPLIB_NAME}-preview)
  endif()
else()
  add_sycl_rt_library(${LIB_NAME} sycl_object COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} SOURCES ${SYCL_NON_PREVIEW_SOURCES} IMPLIB_NAME ${IMPLIB_NAME})
  if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
    add_sycl_rt_library(${LIB_NAME}-preview sycl-preview_object COMPILE_OPTIONS ${SYCL_EXTRA_OPTS} "-D__INTEL_PREVIEW_BREAKING_CHANGES" SOURCES ${SYCL_PREVIEW_SOURCES} IMPLIB_NAME ${IMPLIB_NAME}-preview)
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
