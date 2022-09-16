

# PI Esimd CPU  library
# Create Shared library for libpi_esimd_emulator.so.

include(ExternalProject)

include_directories("${sycl_inc_dir}")
include_directories("${SYCL_SOURCE_DIR}/source/")
# FIXME/TODO: 'pi.h' is included in 'pi_esimd_emulator.cpp', and CL_*_INTEL
# and CL_*_KHR definitions in 'pi.h' are from
# ${OPENCL_INCLUDE}. Remove build dependency on OpenCL
include_directories(${OpenCL_INCLUDE_DIR})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install)

if (WIN32)
  set(LIBCM ${LLVM_BINARY_DIR}/pi_esimd_emulator_deps/lib/libcm${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(LIBIGFXCMRT_EMU ${LLVM_BINARY_DIR}/pi_esimd_emulator_deps/lib/igfxcmrt64_emu${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(LIBCM ${LLVM_BINARY_DIR}/pi_esimd_emulator_deps/lib/libcm${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(LIBIGFXCMRT_EMU ${LLVM_BINARY_DIR}/pi_esimd_emulator_deps/lib/libigfxcmrt_emu${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

set (DEFAULT_CM_EMU_PREBUILT_PACKAGE "https://github.com/intel/cm-cpu-emulation/releases/download/v2022-07-06/intel-cmemu-1.0.23.u20.04-release.x86_64.tar.xz")
set (DEFAULT_CM_EMU_SOURCE_URL "https://github.com/intel/cm-cpu-emulation.git")

if ((DEFINED USE_DEFAULT_CM_EMU_SOURCE) OR (DEFINED USE_LOCAL_CM_EMU_SOURCE))
  if (DEFINED USE_CM_EMU_PREBUILT_PACKAGE)
    message(FATAL_ERROR "Configuration failure : Pre-built package and On-line building of CM_EMU library package cannot be applied together")
  endif()
  if ((DEFINED USE_DEFAULT_CM_EMU_SOURCE) AND (DEFINED USE_LOCAL_CM_EMU_SOURCE))
    message(FATAL_ERROR "Configuration failure : Either default CM_EMU source or local source can be used for on-line CM building - not together")
  endif()
  if (WIN32)
    message(FATAL_ERROR "Configuration failure : Online-building of CM_EMU library is not supported under Windows environment")
  endif()

  if (DEFINED USE_DEFAULT_CM_EMU_SOURCE)
    message(STATUS "CM_EMU library package will be built online with source codes downloaded from ${DEFAULT_CM_EMU_SOURCE_URL}")
    ExternalProject_Add(cm-emu
      GIT_REPOSITORY    ${DEFAULT_CM_EMU_SOURCE_URL}
      GIT_TAG           f9d167edbcb995e713d5a8
      BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build
      INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
      CMAKE_ARGS        -DLIBVA_INSTALL_PATH=/usr
                        -D__SYCL_EXPLICIT_SIMD_PLUGIN__=true
                        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
    )
  elseif (DEFINED USE_LOCAL_CM_EMU_SOURCE)
    message(STATUS "CM_EMU library package will be built online with source codes from ${USE_LOCAL_CM_EMU_SOURCE}")
    ExternalProject_Add(cm-emu
      DOWNLOAD_COMMAND  ""
      SOURCE_DIR        ${USE_LOCAL_CM_EMU_SOURCE}
      BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build
      INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
      CMAKE_ARGS        -DLIBVA_INSTALL_PATH=/usr
                        -D__SYCL_EXPLICIT_SIMD_PLUGIN__=true
                        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
    )
  endif()
else()
  set(ACTUAL_CM_EMU_PREBUILT_PACKAGE ${DEFAULT_CM_EMU_PREBUILT_PACKAGE})
  if (DEFINED USE_CM_EMU_PREBUILT_PACKAGE)
    set(ACTUAL_CM_EMU_PREBUILT_PACKAGE ${USE_CM_EMU_PREBUILT_PACKAGE})
  else()
    message(STATUS "Neither of USE_DEFAULT_CM_EMU_SOURCE, USE_LOCAL_CM_EMU_SOURCE, USE_CM_EMU_PREBUILT_PACKAGE is set, using prebuilt libCM from ${DEFAULT_CM_EMU_PREBUILT_PACKAGE}")
  endif()
  file (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install)
  ExternalProject_Add(cm-emu
    URL               ${ACTUAL_CM_EMU_PREBUILT_PACKAGE}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    UPDATE_COMMAND    ""
    INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
    INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/cm-emu-prefix/src/cm-emu/ <INSTALL_DIR>
    BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
  )
endif()

ExternalProject_Add_Step(cm-emu llvminstall
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LLVM_BINARY_DIR}/pi_esimd_emulator_deps && ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}/pi_esimd_emulator_deps
  COMMENT "Installing cm-emu into the LLVM binary directory"
  DEPENDEES install
)

include_directories(${LLVM_BINARY_DIR}/pi_esimd_emulator_deps/include/igfxcmrt_emu)
include_directories(${LLVM_BINARY_DIR}/pi_esimd_emulator_deps/include/libcm/cm)

# Compilation flag to exclude lines in header files imported from CM
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SYCL_EXPLICIT_SIMD_PLUGIN__")

set(CMAKE_CXX_STANDARD 17)

# Compilation option modification to prevent build termination caused by
# warnings from CM-imported files
if (WIN32)
string(REPLACE "/W4" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
string(REPLACE "-pedantic" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_sycl_plugin(esimd_emulator
  SOURCES
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "pi_esimd_emulator.cpp"
  LIBRARIES
    "${LEVEL_ZERO_LOADER}"
    ${LIBCM}
    ${LIBIGFXCMRT_EMU}
    #TODO remove cyclic dependency
    sycl
)

add_dependencies(pi_esimd_emulator cm-emu)

set_target_properties(pi_esimd_emulator PROPERTIES LINKER_LANGUAGE CXX)

# Copy CM Header files to $(INSTALL)/include/sycl/CL/
install(DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/include/libcm/cm/
        DESTINATION  ${SYCL_INCLUDE_DIR}/sycl/CL
        COMPONENT    libcmrt-headers
        FILES_MATCHING PATTERN "*.h"
)

# Copy '.so' files to '$(INSTALL)/lib'
if (WIN32)
  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/lib/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          COMPONENT   libcmrt-libs
          FILES_MATCHING PATTERN "*.lib"
  )
  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/bin/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
          COMPONENT   libcmrt-dlls
          FILES_MATCHING PATTERN "*.dll"
  )
else()
  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/lib/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          COMPONENT   libcmrt-sos
          FILES_MATCHING PATTERN "*.so"
  )
endif()
