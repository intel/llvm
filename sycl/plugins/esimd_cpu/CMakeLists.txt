
# PI Esimd CPU  library
# Create Shared library for libpi_esimd_cpu.so.

include_directories("${sycl_inc_dir}")
include_directories(${OPENCL_INCLUDE})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libva_build)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libva_install)
ExternalProject_Add(libva
  GIT_REPOSITORY    https://github.com/intel/libva.git
  BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/libva_build
  INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/libva_install
  CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/libva-prefix/src/libva && ./autogen.sh --prefix=${CMAKE_CURRENT_BINARY_DIR}/libva_install
  BUILD_COMMAND     cd ${CMAKE_CURRENT_BINARY_DIR}/libva-prefix/src/libva && make -j
  INSTALL_COMMAND   cd ${CMAKE_CURRENT_BINARY_DIR}/libva-prefix/src/libva && make install
)
ExternalProject_Add_Step(libva llvminstall
  COMMAND mkdir -p ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps && ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps
  COMMENT "Installing libva into the LLVM binary directory"
  DEPENDEES install
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_build)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_install)
ExternalProject_Add(level-zero-loader_esimd_cpu
  GIT_REPOSITORY    https://github.com/oneapi-src/level-zero.git
  GIT_TAG           v1.0.22
  BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_build
  INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_install
  CMAKE_ARGS        -DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIRS}
                    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                    -DCMAKE_INSTALL_LIBDIR:PATH=lib${LLVM_LIBDIR_SUFFIX}
                    ${AUX_CMAKE_FLAGS}
  STEP_TARGETS      configure,build,install
  DEPENDS           ocl-headers
)
ExternalProject_Add_Step(level-zero-loader_esimd_cpu llvminstall
  COMMAND mkdir -p ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps && ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps
  COMMENT "Installing level-zero-loader into the LLVM binary directory"
  DEPENDEES install
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install)
set(LIBCM ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/lib/libcm.so)
set(LIBIGFXCMRT_EMU ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/lib/libigfxcmrt_emu.so)
if (NOT DEFINED CM_LOCAL_SOURCE_DIR)
  ExternalProject_Add(cm-emu
    GIT_REPOSITORY    https://github.com/intel/cm-cpu-emulation.git
    BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build
    INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
    CMAKE_ARGS        -DLIBVA_INSTALL_PATH=${CMAKE_CURRENT_BINARY_DIR}/libva_install
                      -DLevelZero_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_install/include
                      -DLevelZero_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_install/lib
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    DEPENDS           level-zero-loader_esimd_cpu libva
    BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
  )
else ()
  ExternalProject_Add(cm-emu
    URL               ${CM_LOCAL_SOURCE_DIR}
    BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build
    INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
    CMAKE_ARGS        -DLIBVA_INSTALL_PATH=${CMAKE_CURRENT_BINARY_DIR}/libva_install
                      -DLevelZero_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_install/include
                      -DLevelZero_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/level-zero-loader_install/lib
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    DEPENDS           level-zero-loader_esimd_cpu libva
    BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
  )
endif ()
ExternalProject_Add_Step(cm-emu llvminstall
  COMMAND mkdir -p ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps && ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps
  COMMENT "Installing cm-emu into the LLVM binary directory"
  DEPENDEES install
)

include_directories(${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/include/igfxcmrt_emu)

# Compilation flag to exclude lines in header files imported from CM
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SYCL_EXPLICIT_SIMD_PLUGIN__")

# Compilation option modification to prevent build termination caused by
# warnings from CM-imported files
string(REPLACE "-std=c++14" "-std=c++1z" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Werror=date\-time" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Werror" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(pi_esimd_cpu SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "pi_esimd_cpu.cpp"
)

add_dependencies(pi_esimd_cpu ocl-headers)
add_dependencies(pi_esimd_cpu cm-emu)
add_dependencies(sycl-toolchain pi_esimd_cpu)

target_link_libraries(pi_esimd_cpu PRIVATE sycl ${LIBCM} ${LIBIGFXCMRT_EMU})
set_target_properties(pi_esimd_cpu PROPERTIES LINKER_LANGUAGE CXX)

add_common_options(pi_esimd_cpu)

install(TARGETS pi_esimd_cpu
        LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_esimd_cpu
        RUNTIME DESTINATION "bin" COMPONENT pi_esimd_cpu)

# Copy CM Header files to $(INSTALL)/include/sycl/CL/
install(DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/include/libcm/cm/
        DESTINATION  ${SYCL_INCLUDE_DEPLOY_DIR}/CL
        COMPONENT    libcmrt-headers
        FILES_MATCHING PATTERN "*.h"
)

# Copy '.so' files to '$(INSTALL)/lib'
install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/lib/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        COMPONENT   libcmrt-sos
        FILES_MATCHING PATTERN "*.so"
)
