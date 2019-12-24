#To-Do:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

function(add_sycl_rt_library LIB_NAME)

  add_library(${LIB_NAME} SHARED ${ARGN})

  #To-Do: Remove dependency on icd loader and opencl headers.
  add_dependencies(${LIB_NAME}
    ocl-icd
    ocl-headers
    sycl-headers
  )

  set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

  if (MSVC)
      target_compile_definitions(${LIB_NAME} PRIVATE __SYCL_BUILD_SYCL_DLL )
      target_link_libraries(${LIB_NAME} PRIVATE shlwapi)
  else()
      set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
      target_link_libraries(
        ${LIB_NAME} PRIVATE "-Wl,--version-script=${linker_script}")
      set_target_properties(${LIB_NAME} PROPERTIES LINK_DEPENDS ${linker_script})
  endif()

  target_include_directories(
      ${LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} "${sycl_inc_dir}")
  target_link_libraries(${LIB_NAME}
      PRIVATE OpenCL::Headers
      PRIVATE ${OpenCL_LIBRARIES}
      PRIVATE ${CMAKE_DL_LIBS}
  )

  add_common_options(${LIB_NAME})

endfunction(add_sycl_rt_library)

set(SYCL_SOURCES
    "${sycl_inc_dir}/CL/sycl.hpp"
    "detail/accessor_impl.cpp"
    "detail/builtins_common.cpp"
    "detail/builtins_geometric.cpp"
    "detail/builtins_integer.cpp"
    "detail/builtins_math.cpp"
    "detail/builtins_relational.cpp"
    "detail/pi.cpp"
    "detail/common.cpp"
    "detail/config.cpp"
    "detail/context_impl.cpp"
    "detail/device_impl.cpp"
    "detail/device_info.cpp"
    "detail/error_handling/enqueue_kernel.cpp"
    "detail/event_impl.cpp"
    "detail/force_device.cpp"
    "detail/helpers.cpp"
    "detail/image_accessor_util.cpp"
    "detail/image_impl.cpp"
    "detail/kernel_impl.cpp"
    "detail/kernel_info.cpp"
    "detail/memory_manager.cpp"
    "detail/platform_impl.cpp"
    "detail/platform_info.cpp"
    "detail/program_impl.cpp"
    "detail/program_manager/program_manager.cpp"
    "detail/queue_impl.cpp"
    "detail/os_util.cpp"
    "detail/platform_util.cpp"
    "detail/sampler_impl.cpp"
    "detail/stream_impl.cpp"
    "detail/scheduler/commands.cpp"
    "detail/scheduler/scheduler.cpp"
    "detail/scheduler/graph_processor.cpp"
    "detail/scheduler/graph_builder.cpp"
    "detail/usm/clusm.cpp"
    "detail/usm/usm_dispatch.cpp"
    "detail/usm/usm_impl.cpp"
    "detail/util.cpp"
    "context.cpp"
    "device.cpp"
    "device_selector.cpp"
    "event.cpp"
    "exception.cpp"
    "exception_list.cpp"
    "half_type.cpp"
    "kernel.cpp"
    "platform.cpp"
    "queue.cpp"
    "ordered_queue.cpp"
    "sampler.cpp"
    "stream.cpp"
    "spirv_ops.cpp"
    "$<$<PLATFORM_ID:Windows>:detail/windows_pi.cpp>"
    "$<$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>:detail/posix_pi.cpp>"
)

add_sycl_rt_library(sycl ${SYCL_SOURCES})

if (MSVC)
  # MSVC provides two incompatible build variants for its CRT: release and debug
  # To avoid potential issues in user code we also need to provide two kinds
  # of SYCL Runtime Library for release and debug configurations.
  set(SYCL_CXX_FLAGS "")
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MDd" "" SYCL_CXX_FLAGS "${SYCL_CXX_FLAGS}")
    string(REPLACE "/MTd" "" SYCL_CXX_FLAGS "${SYCL_CXX_FLAGS}")
  else()
    if (CMAKE_BUILD_TYPE MATCHES "Release")
      set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    elseif (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
      set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS_MINSIZEREL}")
    elseif (CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
      set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    endif()
    string(REPLACE "/MD" "" SYCL_CXX_FLAGS "${SYCL_CXX_FLAGS}")
    string(REPLACE "/MT" "" SYCL_CXX_FLAGS "${SYCL_CXX_FLAGS}")
  endif()

  # target_compile_options requires list of options, not a string
  string(REPLACE " " ";" SYCL_CXX_FLAGS "${SYCL_CXX_FLAGS}")

  set(SYCL_CXX_FLAGS_RELEASE "${SYCL_CXX_FLAGS};/MD")
  set(SYCL_CXX_FLAGS_DEBUG "${SYCL_CXX_FLAGS};/MDd")

  # CMake automatically applies these flags to all targets. To override this
  # behavior, options lists are reset.
  set(CMAKE_CXX_FLAGS_RELEASE "")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "")
  set(CMAKE_CXX_FLAGS_DEBUG "")

  target_compile_options(sycl PRIVATE ${SYCL_CXX_FLAGS_RELEASE})

  add_sycl_rt_library(sycld ${SYCL_SOURCES})
  target_compile_options(sycld PRIVATE ${SYCL_CXX_FLAGS_DEBUG})
endif()

install(TARGETS ${SYCL_RT_LIBS}
    ARCHIVE DESTINATION "lib" COMPONENT sycl
    LIBRARY DESTINATION "lib" COMPONENT sycl
    RUNTIME DESTINATION "bin" COMPONENT sycl)
add_subdirectory(detail/devicelib)
