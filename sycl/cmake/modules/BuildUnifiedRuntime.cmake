# Builds in-tree UR

# TODO: taken from sycl/plugins/CMakeLists.txt - maybe we should handle this
# within UR (although it is an obscure warning that the build system here
# seems to specifically enable)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()


option(SYCL_UR_BUILD_TESTS "Build tests for UR" OFF)
set(UR_BUILD_TESTS "${SYCL_UR_BUILD_TESTS}" CACHE BOOL "" FORCE)
# UR tests require the examples to be built
set(UR_BUILD_EXAMPLES "${SYCL_UR_BUILD_TESTS}" CACHE BOOL "" FORCE)

option(SYCL_UR_FORMAT_CPP_STYLE "Format code style of UR C++ sources" OFF)
set(UR_FORMAT_CPP_STYLE "${SYCL_UR_FORMAT_CPP_STYLE}" CACHE BOOL "" FORCE)

option(SYCL_UR_ENABLE_ASSERTIONS "Enable assertions for all UR build types" OFF)
set(UR_ENABLE_ASSERTIONS "${SYCL_UR_ENABLE_ASSERTIONS}" CACHE BOOL "" FORCE)

# Here we override the defaults to unified-runtime
set(UR_BUILD_XPTI_LIBS OFF CACHE BOOL "")
set(UR_ENABLE_SYMBOLIZER ON CACHE BOOL "Enable symbolizer for sanitizer layer.")
set(UR_ENABLE_TRACING ON CACHE BOOL "")

set(UR_EXTERNAL_DEPENDENCIES "sycl-headers" CACHE STRING
  "List of external CMake targets for executables/libraries to depend on" FORCE)

# Force fetch Level Zero loader and headers from github.com
option(SYCL_UR_FORCE_FETCH_LEVEL_ZERO "Force fetching Level Zero even if preinstalled loader is found" OFF)
set(UR_FORCE_FETCH_LEVEL_ZERO "${SYCL_UR_FORCE_FETCH_LEVEL_ZERO}" CACHE BOOL "" FORCE)

if("level_zero" IN_LIST SYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_L0 ON)
endif()
if("level_zero_v2" IN_LIST SYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_L0_V2 ON)
endif()
if("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_CUDA ON)
endif()
if("hip" IN_LIST SYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_HIP ON)
  if (SYCL_ENABLE_EXTENSION_JIT)
    set(UR_ENABLE_COMGR ON)
  endif()
endif()
if("opencl" IN_LIST SYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_OPENCL ON)
endif()
if("native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_NATIVE_CPU ON)
endif()

# Disable errors from warnings while building the UR.
# And remember origin flags before doing that.
set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX-")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX-")
  # FIXME: Unified runtime build fails with /DUNICODE
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /UUNICODE")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /UUNICODE")
  # USE_Z7 forces use of /Z7 instead of /Zi which is broken with sccache
  set(USE_Z7 ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error")
endif()

set(UR_INTREE_SOURCE_DIR "${LLVM_SOURCE_DIR}/../unified-runtime")
cmake_path(NORMAL_PATH UR_INTREE_SOURCE_DIR OUTPUT_VARIABLE UR_INTREE_SOURCE_DIR)

if(NOT IS_DIRECTORY "${UR_INTREE_SOURCE_DIR}")
  message(FATAL_ERROR "unified-runtime folder not found at repo root")
endif()

set(UR_INTREE_BINARY_DIR ${LLVM_BINARY_DIR}/unified-runtime)
set(UNIFIED_RUNTIME_SOURCE_DIR
  "${UR_INTREE_SOURCE_DIR}" CACHE PATH
  "Path to Unified Runtime Headers" FORCE)
set(UMF_BUILD_EXAMPLES OFF CACHE INTERNAL "EXAMPLES")
# Due to the use of dependentloadflag and no installer for UMF and hwloc we need
# to link statically on windows
if(WIN32)
  set(UMF_BUILD_SHARED_LIBRARY OFF CACHE INTERNAL "Build UMF shared library")
  set(UMF_LINK_HWLOC_STATICALLY ON CACHE INTERNAL "static HWLOC")
endif()
add_subdirectory(${UNIFIED_RUNTIME_SOURCE_DIR} ${UR_INTREE_BINARY_DIR})

# Restore original flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
set(UNIFIED_RUNTIME_SRC_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source")
set(UNIFIED_RUNTIME_COMMON_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source/common")

add_library(UnifiedRuntimeLoader ALIAS ur_loader)
add_library(UnifiedRuntimeCommon ALIAS ur_common)
add_library(UnifiedMemoryFramework ALIAS ur_umf)

add_library(UnifiedRuntime-Headers INTERFACE)

target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

if(TARGET UnifiedRuntimeLoader)
  # Install the UR loader.
  install(TARGETS ur_loader
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-runtime-loader
    ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-runtime-loader
    RUNTIME DESTINATION "bin" COMPONENT unified-runtime-loader
  )
endif()

add_custom_target(UnifiedRuntimeAdapters)

function(add_sycl_ur_adapter NAME)
  add_dependencies(UnifiedRuntimeAdapters ur_adapter_${NAME})

  install(TARGETS ur_adapter_${NAME}
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT ur_adapter_${NAME}
    RUNTIME DESTINATION "bin" COMPONENT ur_adapter_${NAME})

  set(manifest_file
    ${CMAKE_CURRENT_BINARY_DIR}/install_manifest_ur_adapter_${NAME}.txt)
  add_custom_command(OUTPUT ${manifest_file}
    COMMAND "${CMAKE_COMMAND}"
    "-DCMAKE_INSTALL_COMPONENT=ur_adapter_${NAME}"
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    COMMENT "Deploying component ur_adapter_${NAME}"
    USES_TERMINAL
  )
  add_custom_target(install-sycl-ur-adapter-${NAME}
    DEPENDS ${manifest_file} ur_adapter_${NAME}
  )

  set_property(GLOBAL APPEND PROPERTY
    SYCL_TOOLCHAIN_INSTALL_COMPONENTS ur_adapter_${NAME})
endfunction()

if("level_zero" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(level_zero)

  # TODO: L0 adapter does other... things in its cmake - make sure they get
  # added to the new build system
endif()

if("level_zero_v2" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(level_zero_v2)
endif()

if("cuda" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(cuda)
endif()

if("hip" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(hip)
endif()

if("opencl" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(opencl)
endif()

if("native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(native_cpu)

  # Deal with OCK option
  option(NATIVECPU_USE_OCK "Use the oneAPI Construction Kit for Native CPU" ON)

  if(NATIVECPU_USE_OCK)
    message(STATUS "Compiling Native CPU adapter with OCK support.")
    target_compile_definitions(ur_adapter_native_cpu PRIVATE NATIVECPU_USE_OCK)
  else()
    message(WARNING "Compiling Native CPU adapter without OCK support.
    Some valid SYCL programs may not build or may have low performance.")
  endif()
endif()

if("offload" IN_LIST SYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(offload)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  # On Windows, also build/install debug libraries with the d suffix that are
  # compiled with /MDd so users can link against these in debug builds.
  include(ExternalProject)
  set(URD_BINARY_DIR ${CMAKE_BINARY_DIR}/unified-runtimed)
  set(URD_INSTALL_DIR ${URD_BINARY_DIR}/install)

  # This creates a subbuild which can be used in dependencies with the
  # unified-runtimed target. It invokes the install-unified-runtime-libraries
  # target to install the UR runtime libraries.
  ExternalProject_Add(unified-runtimed
    SOURCE_DIR ${UNIFIED_RUNTIME_SOURCE_DIR}
    BINARY_DIR ${URD_BINARY_DIR}
    INSTALL_DIR ${URD_INSTALL_DIR}
    INSTALL_COMMAND ${CMAKE_COMMAND}
      --build <BINARY_DIR> --config Debug
      --target install-unified-runtime-libraries
    CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=Debug
      -DCMAKE_INSTALL_PREFIX:STRING=<INSTALL_DIR>
      # Enable d suffix on libraries
      -DUR_USE_DEBUG_POSTFIX:BOOL=ON
      # Don't build unnecessary targets in subbuild.
      -DUR_BUILD_EXAMPLES:BOOL=OFF
      -DUR_BUILD_TESTS:BOOL=OFF
      -DUR_BUILD_TOOLS:BOOL=OFF
      # Sanitizer layer is not supported on Windows.
      -DUR_ENABLE_SYMBOLIZER:BOOL=OFF
      # Inherit settings from parent build.
      -DUR_ENABLE_TRACING:BOOL=${UR_ENABLE_TRACING}
      -DUR_ENABLE_COMGR:BOOL=${UR_ENABLE_COMGR}
      -DUR_BUILD_ADAPTER_L0:BOOL=${UR_BUILD_ADAPTER_L0}
      -DUR_BUILD_ADAPTER_L0_V2:BOOL=${UR_BUILD_ADAPTER_L0_V2}
      -DUR_BUILD_ADAPTER_OPENCL:BOOL=${UR_BUILD_ADAPTER_OPENCL}
      -DUR_BUILD_ADAPTER_CUDA:BOOL=${UR_BUILD_ADAPTER_CUDA}
      -DUR_BUILD_ADAPTER_HIP:BOOL=${UR_BUILD_ADAPTER_HIP}
      -DUR_BUILD_ADAPTER_NATIVE_CPU:BOOL=${UR_BUILD_ADAPTER_NATIVE_CPU}
      -DUMF_BUILD_EXAMPLES:BOOL=${UMF_BUILD_EXAMPLES}
      -DUMF_BUILD_SHARED_LIBRARY:BOOL=${UMF_BUILD_SHARED_LIBRARY}
      -DUMF_LINK_HWLOC_STATICALLY:BOOL=${UMF_LINK_HWLOC_STATICALLY}
      -DUMF_DISABLE_HWLOC:BOOL=${UMF_DISABLE_HWLOC}
      # Enable d suffix in UMF
      -DUMF_USE_DEBUG_POSTFIX:BOOL=ON
  )

  # Copy the debug UR runtime libraries to <build>/bin & <build>/lib for use in
  # the parent build, e.g. integration testing.
  set(URD_COPY_FILES)
  macro(urd_copy_library_to_build library shared)
    if(${shared})
      list(APPEND URD_COPY_FILES
        ${LLVM_BINARY_DIR}/bin/${library}.dll
      )
      add_custom_command(
        OUTPUT
          ${LLVM_BINARY_DIR}/bin/${library}.dll
        COMMAND ${CMAKE_COMMAND} -E copy
          ${URD_INSTALL_DIR}/bin/${library}.dll
          ${LLVM_BINARY_DIR}/bin/${library}.dll
      )
    endif()

    list(APPEND URD_COPY_FILES
      ${LLVM_BINARY_DIR}/lib/${library}.lib
    )
    add_custom_command(
      OUTPUT
        ${LLVM_BINARY_DIR}/lib/${library}.lib
      COMMAND ${CMAKE_COMMAND} -E copy
        ${URD_INSTALL_DIR}/lib/${library}.lib
        ${LLVM_BINARY_DIR}/lib/${library}.lib
    )
  endmacro()

  urd_copy_library_to_build(ur_loaderd "NOT;${UR_STATIC_LOADER}")
  foreach(adatper ${SYCL_ENABLE_BACKENDS})
    if(adapter MATCHES "level_zero")
      set(shared "NOT;${UR_STATIC_ADAPTER_L0}")
    else()
      set(shared TRUE)
    endif()
    urd_copy_library_to_build(ur_adapter_${adatper}d "${shared}")
  endforeach()
  # Also copy umfd.dll/umfd.lib
  urd_copy_library_to_build(umfd ${UMF_BUILD_SHARED_LIBRARY})

  add_custom_target(unified-runtimed-build ALL DEPENDS ${URD_COPY_FILES})
  add_dependencies(unified-runtimed-build unified-runtimed)

  # Add the debug UR runtime libraries to the parent install.
  install(
    FILES ${URD_INSTALL_DIR}/bin/ur_loaderd.dll
    DESTINATION "bin" COMPONENT unified-runtime-loader)
  foreach(adapter ${SYCL_ENABLE_BACKENDS})
    install(
      FILES ${URD_INSTALL_DIR}/bin/ur_adapter_${adapter}d.dll
      DESTINATION "bin" COMPONENT ur_adapter_${adapter})
    add_dependencies(install-sycl-ur-adapter-${adapter} unified-runtimed)
  endforeach()
  if(UMF_BUILD_SHARED_LIBRARY)
    # Also install umfd.dll
    install(
      FILES ${URD_INSTALL_DIR}/bin/umfd.dll
      DESTINATION "bin" COMPONENT unified-memory-framework)
  endif()
endif()

if(TARGET umf)
  install(TARGETS umf
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-memory-framework
    ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-memory-framework
    RUNTIME DESTINATION "bin" COMPONENT unified-memory-framework)
endif()
