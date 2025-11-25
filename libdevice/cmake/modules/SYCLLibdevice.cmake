include(CheckCXXCompilerFlag)
set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(obj-new-offload_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (MSVC)
  set(obj-suffix obj)
  set(obj-new-offload-suffix new.obj)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(install_dest_spv bin)
  set(devicelib_host_static_obj sycl-devicelib-host.lib)
  set(devicelib_host_static_obj-new-offload sycl-devicelib-host.new.lib)
else()
  set(obj-suffix o)
  set(obj-new-offload-suffix new.o)
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(install_dest_spv lib${LLVM_LIBDIR_SUFFIX})
  set(devicelib_host_static_obj libsycl-devicelib-host.a)
  set(devicelib_host_static_obj-new-offload libsycl-devicelib-host.new.a)
endif()
set(spv-suffix spv)
set(bc-suffix bc)
set(bc_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(install_dest_obj lib${LLVM_LIBDIR_SUFFIX})
set(install_dest_obj-new-offload lib${LLVM_LIBDIR_SUFFIX})
set(install_dest_bc lib${LLVM_LIBDIR_SUFFIX})

string(CONCAT sycl_targets_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown,"
  "spir64_gen-unknown-unknown,"
  "spir64-unknown-unknown,"
  "spirv64-unknown-unknown")

string(CONCAT sycl_pvc_target_opt
  "-fsycl-targets="
  "intel_gpu_pvc")

string(CONCAT sycl_cpu_target_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown")

string(CONCAT sycl_dg2_target_opt
  "-fsycl-targets="
  "spir64_gen-unknown-unknown")

set(compile_opts
  # suppress an error about SYCL_EXTERNAL being used for
  # a function with a raw pointer parameter.
  -Wno-sycl-strict
  # Disable warnings for the host compilation, where
  # we declare all functions as 'static'.
  -Wno-undefined-internal
  -sycl-std=2020
  --target=${LLVM_HOST_TRIPLE}
  )

set(SYCL_LIBDEVICE_GCC_TOOLCHAIN "" CACHE PATH "Path to GCC installation")

set(SYCL_LIBDEVICE_CXX_FLAGS "" CACHE STRING "C++ compiler flags for SYCL libdevice")
if(NOT SYCL_LIBDEVICE_CXX_FLAGS STREQUAL "")
  separate_arguments(SYCL_LIBDEVICE_CXX_FLAGS NATIVE_COMMAND ${SYCL_LIBDEVICE_CXX_FLAGS})
endif()

if (NOT SYCL_LIBDEVICE_GCC_TOOLCHAIN STREQUAL "")
  list(APPEND SYCL_LIBDEVICE_CXX_FLAGS "--gcc-install-dir=${SYCL_LIBDEVICE_GCC_TOOLCHAIN}")
endif()

if(NOT SYCL_LIBDEVICE_CXX_FLAGS STREQUAL "")
  list(APPEND compile_opts ${SYCL_LIBDEVICE_CXX_FLAGS})
endif()

if(LLVM_LIBCXX_USED)
  list(APPEND compile_opts "-stdlib=libc++")
endif()

if (WIN32)
  list(APPEND compile_opts "-std=c++17")
  list(APPEND compile_opts -D_ALLOW_RUNTIME_LIBRARY_MISMATCH)
  list(APPEND compile_opts -D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

add_custom_target(libsycldevice)

set(filetypes obj obj-new-offload spv bc)
set(filetypes_no_spv obj obj-new-offload bc)

foreach(filetype IN LISTS filetypes)
  add_custom_target(libsycldevice-${filetype})
  add_dependencies(libsycldevice libsycldevice-${filetype})
endforeach()

# For NVPTX and AMDGCN each device libary is compiled into a single bitcode
# file and all files created this way are linked into one large bitcode
# library.
# Additional compilation options are needed for compiling each device library.
set(full_build_archs)
if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND full_build_archs nvptx64-nvidia-cuda)
  set(compile_opts_nvptx64-nvidia-cuda "-fsycl-targets=nvptx64-nvidia-cuda"
  "-Xsycl-target-backend" "--cuda-gpu-arch=sm_50" "-nocudalib" "-fno-sycl-libspirv" "-Wno-unsafe-libspirv-not-linked")
  set(opt_flags_nvptx64-nvidia-cuda "-O3" "--nvvm-reflect-enable=false")
endif()
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND full_build_archs amdgcn-amd-amdhsa)
  set(compile_opts_amdgcn-amd-amdhsa "-nogpulib" "-fsycl-targets=amdgcn-amd-amdhsa"
  "-Xsycl-target-backend" "--offload-arch=gfx942" "-fno-sycl-libspirv" "-Wno-unsafe-libspirv-not-linked")
  set(opt_flags_amdgcn-amd-amdhsa "-O3" "--amdgpu-oclc-reflect-enable=false")
endif()


set(spv_device_compile_opts -fsycl-device-only -fsycl-device-obj=spirv)
set(bc_device_compile_opts -fsycl-device-only -fsycl-device-obj=llvmir)
set(obj-new-offload_device_compile_opts -fsycl -c --offload-new-driver
  -foffload-lto=thin ${sycl_targets_opt})
set(obj_device_compile_opts -fsycl -c ${sycl_targets_opt} --no-offload-new-driver)

# Compiles and installs a single device library.
#
# Arguments:
# * FILETYPE <string>
#     Specifies the output file type of the compilation and its repsective
#     installation directory.
#     Adds a new target that the libsycldevice-FILETYPE target will depend on.
# * SRC <string> ...
#    Source code files needed for the compilation.
# * EXTRA_OPTS <string> ...
#     List of extra compiler options to use.
#     Note that the ones specified by the compile_opts var are always used.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the clang target for compiling.
function(compile_lib filename)
  cmake_parse_arguments(ARG
    ""
    "FILETYPE"
    "SRC;EXTRA_OPTS;DEPENDENCIES"
    ${ARGN})
    set(compile_opt_list ${compile_opts}
                         ${${ARG_FILETYPE}_device_compile_opts}
                         ${ARG_EXTRA_OPTS})
    compile_lib_ext(${filename}
      FILETYPE ${ARG_FILETYPE}
      SRC ${ARG_SRC}
      DEPENDENCIES ${ARG_DEPENDENCIES}
      OPTS ${compile_opt_list})
endfunction()

function(compile_lib_ext filename)
  cmake_parse_arguments(ARG
    ""
    "FILETYPE"
    "SRC;OPTS;DEPENDENCIES"
    ${ARGN})

  set(devicelib-file
    ${${ARG_FILETYPE}_binary_dir}/${filename}.${${ARG_FILETYPE}-suffix})

  add_custom_command(
    OUTPUT ${devicelib-file}
    COMMAND ${clang_exe} -I ${PROJECT_BINARY_DIR}/include
            ${ARG_OPTS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC} -o ${devicelib-file}
    MAIN_DEPENDENCY ${ARG_SRC}
    DEPENDS ${ARG_DEPENDENCIES}
    VERBATIM
  )
  set(devicelib-${ARG_FILETYPE}-target ${filename}-${ARG_FILETYPE})
  add_custom_target(${devicelib-${ARG_FILETYPE}-target}
    DEPENDS ${devicelib-file})
  add_dependencies(libsycldevice-${ARG_FILETYPE}
    ${devicelib-${ARG_FILETYPE}-target})

  install( FILES ${devicelib-file}
           DESTINATION ${install_dest_${ARG_FILETYPE}}
           COMPONENT libsycldevice)
endfunction()

# Links together one or more bytecode files
#
# Arguments:
# * INTERNALIZE
#     Set if -internalize flag should be passed when linking
# * TARGET <string>
#     Custom target to create
# * INPUT <string> ...
#     List of bytecode files to link together
# * RSP_DIR <string>
#     Directory where a response file should be placed
#     (Only needed for WIN32 or CYGWIN)
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(link_bc)
  cmake_parse_arguments(ARG
    "INTERNALIZE"
    "TARGET;RSP_DIR"
    "INPUTS;DEPENDENCIES"
    ${ARGN}
  )

  set( LINK_INPUT_ARG ${ARG_INPUTS} )
  if( WIN32 OR CYGWIN )
    # Create a response file in case the number of inputs exceeds command-line
    # character limits on certain platforms.
    file( TO_CMAKE_PATH ${ARG_RSP_DIR}/${ARG_TARGET}.rsp RSP_FILE )
    # Turn it into a space-separate list of input files
    list( JOIN ARG_INPUTS " " RSP_INPUT )
    file( GENERATE OUTPUT ${RSP_FILE} CONTENT ${RSP_INPUT} )
    # Ensure that if this file is removed, we re-run CMake
    set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${RSP_FILE}
    )
    set( LINK_INPUT_ARG "@${RSP_FILE}" )
  endif()

  if( ARG_INTERNALIZE )
    set( link_flags --internalize --only-needed )
  endif()

  add_custom_command(
    OUTPUT ${ARG_TARGET}.bc
    COMMAND ${llvm-link_exe} ${link_flags} -o ${ARG_TARGET}.bc ${LINK_INPUT_ARG}
    DEPENDS ${llvm-link_target} ${ARG_DEPENDENCIES} ${ARG_INPUTS} ${RSP_FILE}
  )

  add_custom_target( ${ARG_TARGET} ALL DEPENDS ${ARG_TARGET}.bc )
  set_target_properties( ${ARG_TARGET} PROPERTIES
    TARGET_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.bc
  )
endfunction()

# Runs opt and prepare-builtins on a bitcode file specified by lib_tgt
#
# ARGUMENTS:
# * LIB_TGT string
#     Target name that becomes dependent on the out file named LIB_TGT.bc
# * IN_FILE string
#     Target name of the input bytecode file
# * OUT_DIR string
#     Name of the directory where the output should be placed
# *  DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(process_bc out_file)
  cmake_parse_arguments(ARG
    ""
    "LIB_TGT;IN_FILE;OUT_DIR"
    "OPT_FLAGS;DEPENDENCIES"
    ${ARGN})
  add_custom_command( OUTPUT ${ARG_LIB_TGT}.bc
    COMMAND ${opt_exe} ${ARG_OPT_FLAGS} -o ${ARG_LIB_TGT}.bc
    ${ARG_IN_FILE}
    DEPENDS ${opt_target} ${ARG_IN_FILE} ${ARG_DEPENDENCIES}
  )
  add_custom_target( ${ARG_LIB_TGT}
    ALL DEPENDS ${ARG_LIB_TGT}.bc
  )
  set_target_properties( ${ARG_LIB_TGT}
    PROPERTIES TARGET_FILE ${ARG_LIB_TGT}.bc
  )

  set( builtins_opt_lib $<TARGET_PROPERTY:${ARG_LIB_TGT},TARGET_FILE> )

  # Add prepare target
  # FIXME: prepare_builtins_exe comes from having included libclc before this.
  # This is brittle.
  add_custom_command( OUTPUT ${ARG_OUT_DIR}/${out_file}
    COMMAND ${prepare_builtins_exe} -o ${ARG_OUT_DIR}/${out_file}
      ${builtins_opt_lib}
      DEPENDS ${builtins_opt_lib} ${ARG_LIB_TGT} ${prepare_builtins_target} )
  add_custom_target( prepare-${out_file} ALL
    DEPENDS ${ARG_OUT_DIR}/${out_file}
  )
  set_target_properties( prepare-${out_file}
    PROPERTIES TARGET_FILE ${ARG_OUT_DIR}/${out_file}
  )
endfunction()

# Appends a list to a global property.
#
# Arguments:
# * PROPERTY_NAME <string>
#     The name of the property to append to.
function(append_to_property list)
  cmake_parse_arguments(ARG
    ""
    "PROPERTY_NAME"
    ""
    ${ARGN})
  get_property(new_property GLOBAL PROPERTY ${ARG_PROPERTY_NAME})
  list(APPEND new_property ${list})
  set_property(GLOBAL PROPERTY ${ARG_PROPERTY_NAME} ${new_property})
endfunction()

# Creates device libaries for all filetypes.
# Adds bitcode library files additionally for each devicelib build arch and
# adds the created file to an arch specific global property.
#
# Arguments:
# * SRC <string> ...
#    Source code files needed for the compilation.
# * EXTRA_OPTS <string> ...
#     List of extra compiler options to use.
#     Note that the ones specified by the compile_opts var are always used.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the clang target for compiling.
function(add_devicelibs filename)
  cmake_parse_arguments(ARG
    ""
    ""
    "SRC;EXTRA_OPTS;DEPENDENCIES;BUILD_ARCHS;FILETYPES"
    ${ARGN})
  if(ARG_FILETYPES)
    set(devicelib_filetypes "${ARG_FILETYPES}")
  else()
    set(devicelib_filetypes "${filetypes}")
  endif()
  foreach(filetype IN LISTS devicelib_filetypes)
    compile_lib(${filename}
      FILETYPE ${filetype}
      SRC ${ARG_SRC}
      DEPENDENCIES ${ARG_DEPENDENCIES}
      EXTRA_OPTS ${ARG_EXTRA_OPTS})
  endforeach()

  foreach(arch IN LISTS ARG_BUILD_ARCHS)
    compile_lib(${filename}-${arch}
      FILETYPE bc
      SRC ${ARG_SRC}
      DEPENDENCIES ${ARG_DEPENDENCIES}
      EXTRA_OPTS ${ARG_EXTRA_OPTS} ${bc_device_compile_opts}
                 ${compile_opts_${arch}})

    append_to_property(${bc_binary_dir}/${filename}-${arch}.bc
      PROPERTY_NAME BC_DEVICE_LIBS_${arch})
  endforeach()
endfunction()

# For native builds, sycl-compiler will already include everything we need.
# For cross builds, we also need native versions of the tools.
set(sycl-compiler_deps
  sycl-compiler ${clang_target} ${append-file_target}
  ${clang-offload-bundler_target} ${llvm-offload-binary_target}
  ${file-table-tform_target} ${llvm-foreach_target} ${llvm-spirv_target}
  ${sycl-post-link_target})
set(crt_obj_deps wrapper.h device.h spirv_vars.h malloc.hpp ${sycl-compiler_deps})
set(complex_obj_deps device_complex.h device.h ${sycl-compiler_deps})
set(cmath_obj_deps device_math.h device.h ${sycl-compiler_deps})
set(imf_obj_deps device_imf.hpp imf_half.hpp imf_bf16.hpp imf_rounding_op.hpp imf_impl_utils.hpp device.h ${sycl-compiler_deps})
set(itt_obj_deps device_itt.h spirv_vars.h device.h ${sycl-compiler_deps})
set(bfloat16_obj_deps sycl-headers ${sycl-compiler_deps})
if (NOT MSVC AND UR_SANITIZER_INCLUDE_DIR)
  set(asan_obj_deps
    device.h atomic.hpp spirv_vars.h
    ${UR_SANITIZER_INCLUDE_DIR}/asan/asan_libdevice.hpp
    include/asan_rtl.hpp
    include/sanitizer_defs.hpp
    include/spir_global_var.hpp
    include/sanitizer_utils.hpp
    ${sycl-compiler_deps})

  set(sanitizer_generic_compile_opts ${compile_opts}
                            -fno-sycl-instrument-device-code
                            -I${UR_SANITIZER_INCLUDE_DIR}
                            -I${CMAKE_CURRENT_SOURCE_DIR})

  set(sanitizer_pvc_compile_opts_obj -fsycl -c --no-offload-new-driver
                                ${sanitizer_generic_compile_opts}
                                ${sycl_pvc_target_opt}
                                -D__LIBDEVICE_PVC__)

  set(sanitizer_cpu_compile_opts_obj -fsycl -c --no-offload-new-driver
                                ${sanitizer_generic_compile_opts}
                                ${sycl_cpu_target_opt}
                                -D__LIBDEVICE_CPU__)

  set(sanitizer_dg2_compile_opts_obj -fsycl -c --no-offload-new-driver
                                ${sanitizer_generic_compile_opts}
                                ${sycl_dg2_target_opt}
                                -D__LIBDEVICE_DG2__)

  set(sanitizer_pvc_compile_opts_bc  ${bc_device_compile_opts}
                                ${sanitizer_generic_compile_opts}
                                -D__LIBDEVICE_PVC__)

  set(sanitizer_cpu_compile_opts_bc  ${bc_device_compile_opts}
                                ${sanitizer_generic_compile_opts}
                                -D__LIBDEVICE_CPU__)

  set(sanitizer_dg2_compile_opts_bc  ${bc_device_compile_opts}
                                ${sanitizer_generic_compile_opts}
                                -D__LIBDEVICE_DG2__)

  set(sanitizer_pvc_compile_opts_obj-new-offload -fsycl -c --offload-new-driver
                                            -foffload-lto=thin
                                            ${sanitizer_generic_compile_opts}
                                            ${sycl_pvc_target_opt}
                                            -D__LIBDEVICE_PVC__)

  set(sanitizer_cpu_compile_opts_obj-new-offload -fsycl -c --offload-new-driver
                                            -foffload-lto=thin
                                            ${sanitizer_generic_compile_opts}
                                            ${sycl_cpu_target_opt}
                                            -D__LIBDEVICE_CPU__)

  set(sanitizer_dg2_compile_opts_obj-new-offload -fsycl -c --offload-new-driver
                                            -foffload-lto=thin
                                            ${sanitizer_generic_compile_opts}
                                            ${sycl_dg2_target_opt}
                                            -D__LIBDEVICE_DG2__)
  
  set(msan_obj_deps
    device.h atomic.hpp spirv_vars.h
    ${UR_SANITIZER_INCLUDE_DIR}/msan/msan_libdevice.hpp
    include/msan_rtl.hpp
    include/sanitizer_defs.hpp
    include/spir_global_var.hpp
    include/sanitizer_utils.hpp
    ${sycl-compiler_deps})

  set(tsan_obj_deps
    device.h atomic.hpp spirv_vars.h
    ${UR_SANITIZER_INCLUDE_DIR}/tsan/tsan_libdevice.hpp
    include/tsan_rtl.hpp
    include/sanitizer_defs.hpp
    include/spir_global_var.hpp
    include/sanitizer_utils.hpp
    ${sycl-compiler_deps})
endif()

if("native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  if (NOT DEFINED NATIVE_CPU_DIR)
    message( FATAL_ERROR "Undefined UR variable NATIVE_CPU_DIR. The name may have changed." )
  endif()
  # Include NativeCPU UR adapter path to enable finding header file with state struct.
  # libsycl-nativecpu_utils is only needed as BC file by NativeCPU.
  add_custom_command(
    OUTPUT ${bc_binary_dir}/nativecpu_utils.bc
    COMMAND ${clang_exe} ${compile_opts} ${bc_device_compile_opts}
      -fsycl-targets=native_cpu -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked
      -I ${PROJECT_BINARY_DIR}/include -I ${NATIVE_CPU_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/nativecpu_utils.cpp
      -o ${bc_binary_dir}/nativecpu_utils.bc
    MAIN_DEPENDENCY nativecpu_utils.cpp
    DEPENDS sycl-headers ${sycl-compiler_deps}
    VERBATIM)
  add_custom_target(nativecpu_utils-bc DEPENDS ${bc_binary_dir}/nativecpu_utils.bc)
  process_bc(libsycl-nativecpu_utils.bc
    LIB_TGT libsycl-nativecpu_utils
    IN_FILE ${bc_binary_dir}/nativecpu_utils.bc
    OUT_DIR ${bc_binary_dir})
  add_custom_target(libsycl-nativecpu_utils-bc DEPENDS ${bc_binary_dir}/libsycl-nativecpu_utils.bc)
  add_dependencies(libsycldevice-bc libsycl-nativecpu_utils-bc)
  install(FILES ${bc_binary_dir}/libsycl-nativecpu_utils.bc
          DESTINATION ${install_dest_bc}
          COMPONENT libsycldevice)
endif()

check_cxx_compiler_flag(-Wno-invalid-noreturn HAS_NO_INVALID_NORETURN_WARN_FLAG)
# Add all device libraries for each filetype except for the Intel math function
# ones.
add_devicelibs(libsycl-itt-stubs
  SRC itt_stubs.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${itt_obj_deps})
add_devicelibs(libsycl-itt-compiler-wrappers
  SRC itt_compiler_wrappers.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${itt_obj_deps})
add_devicelibs(libsycl-itt-user-wrappers
  SRC itt_user_wrappers.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${itt_obj_deps})

add_devicelibs(libsycl-crt
  SRC crt_wrapper.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${crt_obj_deps}
  EXTRA_OPTS $<$<BOOL:${HAS_NO_INVALID_NORETURN_WARN_FLAG}>:-Wno-invalid-noreturn>)

add_devicelibs(libsycl-complex
  SRC complex_wrapper.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${complex_obj_deps})
add_devicelibs(libsycl-complex-fp64
  SRC complex_wrapper_fp64.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${complex_obj_deps} )
add_devicelibs(libsycl-cmath
  SRC cmath_wrapper.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${cmath_obj_deps})
add_devicelibs(libsycl-cmath-fp64
  SRC cmath_wrapper_fp64.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${cmath_obj_deps} )
set(imf_build_archs)
add_devicelibs(libsycl-imf
  SRC imf_wrapper.cpp
  DEPENDENCIES ${imf_obj_deps}
  BUILD_ARCHS ${imf_build_archs})
add_devicelibs(libsycl-imf-fp64
  SRC imf_wrapper_fp64.cpp
  DEPENDENCIES ${imf_obj_deps}
  BUILD_ARCHS ${imf_build_archs})
add_devicelibs(libsycl-imf-bf16
  SRC imf_wrapper_bf16.cpp
  DEPENDENCIES ${imf_obj_deps}
  BUILD_ARCHS ${imf_build_archs})
add_devicelibs(libsycl-bfloat16
  SRC bfloat16_wrapper.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${cmath_obj_deps})
if(MSVC)
  add_devicelibs(libsycl-msvc-math
    SRC msvc_math.cpp
    BUILD_ARCHS ${full_build_archs}
    DEPENDENCIES ${cmath_obj_deps})
else()
  if(UR_SANITIZER_INCLUDE_DIR)
    set(sanitizer_build_archs)
    # asan jit
    add_devicelibs(libsycl-asan
      SRC sanitizer/asan_rtl.cpp
      DEPENDENCIES ${asan_obj_deps}
      BUILD_ARCHS ${sanitizer_build_archs}
      FILETYPES "${filetypes_no_spv}"
      EXTRA_OPTS -fno-sycl-instrument-device-code
                 -I${UR_SANITIZER_INCLUDE_DIR}
                 -I${CMAKE_CURRENT_SOURCE_DIR})

    # asan aot
    set(asan_devicetypes pvc cpu dg2)

    foreach(asan_ft IN LISTS filetypes_no_spv)
      foreach(asan_device IN LISTS asan_devicetypes)
        compile_lib_ext(libsycl-asan-${asan_device}
                        SRC sanitizer/asan_rtl.cpp
                        FILETYPE ${asan_ft}
                        DEPENDENCIES ${asan_obj_deps}
                        OPTS ${sanitizer_${asan_device}_compile_opts_${asan_ft}})
      endforeach()
    endforeach()

    # msan jit
    add_devicelibs(libsycl-msan
      SRC sanitizer/msan_rtl.cpp
      DEPENDENCIES ${msan_obj_deps}
      BUILD_ARCHS ${sanitizer_build_archs}
      FILETYPES "${filetypes_no_spv}"
      EXTRA_OPTS -fno-sycl-instrument-device-code
                 -I${UR_SANITIZER_INCLUDE_DIR}
                 -I${CMAKE_CURRENT_SOURCE_DIR})

    # msan aot
    set(msan_devicetypes pvc cpu)

    foreach(msan_ft IN LISTS filetypes_no_spv)
      foreach(msan_device IN LISTS msan_devicetypes)
        compile_lib_ext(libsycl-msan-${msan_device}
                        SRC sanitizer/msan_rtl.cpp
                        FILETYPE ${msan_ft}
                        DEPENDENCIES ${msan_obj_deps}
                        OPTS ${sanitizer_${msan_device}_compile_opts_${msan_ft}})
      endforeach()
    endforeach()

    # tsan jit
    add_devicelibs(libsycl-tsan
      SRC sanitizer/tsan_rtl.cpp
      DEPENDENCIES ${tsan_obj_deps}
      BUILD_ARCHS ${sanitizer_build_archs}
      FILETYPES "${filetypes_no_spv}"
      EXTRA_OPTS -fno-sycl-instrument-device-code
                 -I${UR_SANITIZER_INCLUDE_DIR}
                 -I${CMAKE_CURRENT_SOURCE_DIR})

    set(tsan_devicetypes pvc cpu)

    foreach(tsan_ft IN LISTS filetypes_no_spv)
      foreach(tsan_device IN LISTS tsan_devicetypes)
        compile_lib_ext(libsycl-tsan-${tsan_device}
                        SRC sanitizer/tsan_rtl.cpp
                        FILETYPE ${tsan_ft}
                        DEPENDENCIES ${tsan_obj_deps}
                        OPTS ${sanitizer_${tsan_device}_compile_opts_${tsan_ft}})
      endforeach()
    endforeach()

  endif()
endif()

add_devicelibs(libsycl-fallback-cassert
  SRC fallback-cassert.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${crt_obj_deps}
  EXTRA_OPTS -fno-sycl-instrument-device-code)
add_devicelibs(libsycl-fallback-cstring
  SRC fallback-cstring.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${crt_obj_deps})
add_devicelibs(libsycl-fallback-complex
  SRC fallback-complex.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${complex_obj_deps})
add_devicelibs(libsycl-fallback-complex-fp64
  SRC fallback-complex-fp64.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${complex_obj_deps})
add_devicelibs(libsycl-fallback-cmath
  SRC fallback-cmath.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${cmath_obj_deps})
add_devicelibs(libsycl-fallback-cmath-fp64
  SRC fallback-cmath-fp64.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${cmath_obj_deps})
add_devicelibs(libsycl-fallback-bfloat16
  SRC fallback-bfloat16.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${bfloat16_obj_deps})
add_devicelibs(libsycl-native-bfloat16
  SRC bfloat16_wrapper.cpp
  BUILD_ARCHS ${full_build_archs}
  DEPENDENCIES ${bfloat16_obj_deps})

# Create dependency and source lists for Intel math function libraries.
file(MAKE_DIRECTORY ${obj_binary_dir}/libdevice)
set(imf_fallback_src_dir ${obj_binary_dir}/libdevice)
set(imf_src_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(imf_fallback_fp32_deps device.h device_imf.hpp imf_half.hpp imf_rounding_op.hpp imf_impl_utils.hpp
                           imf_utils/integer_misc.cpp
                           imf_utils/float_convert.cpp
                           imf_utils/half_convert.cpp
                           imf_utils/simd_emulate.cpp
                           imf_utils/fp32_round.cpp
                           imf/imf_inline_fp32.cpp
                           imf/imf_fp32_dl.cpp)
set(imf_fallback_fp64_deps device.h device_imf.hpp imf_half.hpp imf_rounding_op.hpp imf_impl_utils.hpp
                           imf_utils/double_convert.cpp imf_utils/fp64_round.cpp
                           imf/imf_inline_fp64.cpp
                           imf/imf_fp64_dl.cpp)
set(imf_fallback_bf16_deps device.h device_imf.hpp imf_bf16.hpp
                           imf_utils/bfloat16_convert.cpp
                           imf/imf_inline_bf16.cpp)

set(imf_fp32_fallback_src ${imf_fallback_src_dir}/imf_fp32_fallback.cpp)
set(imf_fp64_fallback_src ${imf_fallback_src_dir}/imf_fp64_fallback.cpp)
set(imf_bf16_fallback_src ${imf_fallback_src_dir}/imf_bf16_fallback.cpp)

set(imf_host_cxx_flags -c
  --target=${LLVM_HOST_TRIPLE}
  -D__LIBDEVICE_HOST_IMPL__
  ${SYCL_LIBDEVICE_CXX_FLAGS}
)

if(LLVM_LIBCXX_USED)
  list(APPEND  imf_host_cxx_flags "-stdlib=libc++")
endif()

if (WIN32)
  list(APPEND imf_host_cxx_flags "-std=c++17")
endif()

macro(mangle_name str output)
  string(STRIP "${str}" strippedStr)
  string(REGEX REPLACE "^/" "" strippedStr "${strippedStr}")
  string(REGEX REPLACE "^-+" "" strippedStr "${strippedStr}")
  string(REGEX REPLACE "-+$" "" strippedStr "${strippedStr}")
  string(REPLACE "-" "_" strippedStr "${strippedStr}")
  string(REPLACE "=" "_EQ_" strippedStr "${strippedStr}")
  string(REPLACE "+" "X" strippedStr "${strippedStr}")
  string(TOUPPER "${strippedStr}" ${output})
endmacro()

# Add a list of flags to 'imf_host_cxx_flags'.
macro(add_imf_host_cxx_flags_compile_flags)
  foreach(f ${ARGN})
    list(APPEND imf_host_cxx_flags ${f})
  endforeach()
endmacro()

# If 'condition' is true then add the specified list of flags to
# 'imf_host_cxx_flags'
macro(add_imf_host_cxx_flags_compile_flags_if condition)
  if (${condition})
    add_imf_host_cxx_flags_compile_flags(${ARGN})
  endif()
endmacro()

# For each specified flag, add that flag to 'imf_host_cxx_flags' if the
# flag is supported by the C++ compiler.
macro(add_imf_host_cxx_flags_compile_flags_if_supported)
  foreach(flag ${ARGN})
      mangle_name("${flag}" flagname)
      check_cxx_compiler_flag("${flag}" "CXX_SUPPORTS_${flagname}_FLAG")
      add_imf_host_cxx_flags_compile_flags_if(CXX_SUPPORTS_${flagname}_FLAG ${flag})
  endforeach()
endmacro()


if (NOT WIN32)
  list(APPEND imf_host_cxx_flags -fPIC)
  add_imf_host_cxx_flags_compile_flags_if_supported("-fcf-protection=full")
endif()

set(obj-new-offload_host_compile_opts ${imf_host_cxx_flags} --offload-new-driver
  -foffload-lto=thin)
set(obj_host_compile_opts ${imf_host_cxx_flags} --no-offload-new-driver)

foreach(datatype IN ITEMS fp32 fp64 bf16)
  string(TOUPPER ${datatype} upper_datatype)

  add_custom_command(
    OUTPUT ${imf_${datatype}_fallback_src}
    COMMAND ${CMAKE_COMMAND}
            -D SRC_DIR=${imf_src_dir}
            -D DEST_DIR=${imf_fallback_src_dir}
            -D IMF_TARGET=${upper_datatype}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ImfSrcConcate.cmake
    DEPENDS ${imf_fallback_${datatype}_deps})

  add_custom_target(get_imf_fallback_${datatype}
     DEPENDS ${imf_${datatype}_fallback_src})
endforeach()

# Adds Intel math functions libraries.
#
# Arguments:
# * SRC <string> ...
#    Source code files needed for the compilation.
# * DIR <string>
#    The directory where the output file should be located in.
# * FTYPE <string>
#    Filetype of the output library file (e.g. 'bc').
# * DTYPE <string>
#   The datatype of the library, which determines the input source
#   and dependencies of the compilation command.
# * TGT_NAME <string>
#   Name of the new target that depends on the compilation of the library.
# * EXTRA_OPTS <string> ...
#     List of extra compiler options to use.
#     Note that the ones specified by the compile_opts var are always used.
#
# Depends on the clang target for compiling.
function(add_lib_imf name)
  cmake_parse_arguments(ARG
    ""
    "DIR;FTYPE;DTYPE;TGT_NAME"
    "EXTRA_OPTS"
    ${ARGN})

  add_custom_command(
    OUTPUT ${ARG_DIR}/${name}.${${ARG_FTYPE}-suffix}
    COMMAND ${clang_exe} ${compile_opts} ${ARG_EXTRA_OPTS}
            -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
            ${imf_${ARG_DTYPE}_fallback_src}
            -o
            ${ARG_DIR}/${name}.${${ARG_FTYPE}-suffix}
            DEPENDS ${imf_fallback_${ARG_DTYPE}_deps}
            get_imf_fallback_${ARG_DTYPE} ${sycl-compiler_deps}
    VERBATIM)

  add_custom_target(${ARG_TGT_NAME}
    DEPENDS ${ARG_DIR}/${name}.${${ARG_FTYPE}-suffix})

  add_dependencies(libsycldevice-${ARG_FTYPE} ${ARG_TGT_NAME})
endfunction()

# Add device fallback imf libraries for the SPIRV targets and all filetypes.
foreach(dtype IN ITEMS bf16 fp32 fp64)
  foreach(ftype IN LISTS filetypes)
    set(libsycl_name libsycl-fallback-imf)
    if (NOT (dtype STREQUAL "fp32"))
      set(libsycl_name libsycl-fallback-imf-${dtype})
    endif()
    set(tgt_name imf_fallback_${dtype}_${ftype})

    add_lib_imf(${libsycl_name}
      DIR ${${ftype}_binary_dir}
      FTYPE ${ftype}
      DTYPE ${dtype}
      EXTRA_OPTS ${${ftype}_device_compile_opts}
      TGT_NAME ${tgt_name})
  endforeach()
endforeach()

# Add device fallback imf libraries for single bitcode targets.
# The output files are bitcode.
foreach(arch IN LISTS imf_build_archs)
  foreach(dtype IN ITEMS bf16 fp32 fp64)
    set(tgt_name imf_fallback_${dtype}_bc_${arch})

    add_lib_imf(libsycl-fallback-imf-${arch}-${dtype}
      ARCH ${arch}
      DIR ${bc_binary_dir}
      FTYPE bc
      DTYPE ${dtype}
      EXTRA_OPTS ${bc_device_compile_opts} ${compile_opts_${arch}}
      TGT_NAME ${tgt_name})

    append_to_property(
      ${bc_binary_dir}/libsycl-fallback-imf-${arch}-${dtype}.${bc-suffix}
      PROPERTY_NAME BC_DEVICE_LIBS_${arch})
  endforeach()
endforeach()

# Create one large bitcode file for the NVPTX and AMD targets.
# Use all the files collected in the respective global properties.
foreach(arch IN LISTS full_build_archs)
  get_property(BC_DEVICE_LIBS_${arch} GLOBAL PROPERTY BC_DEVICE_LIBS_${arch})
  # Link the bitcode files together.
  link_bc(TARGET device_lib_device_${arch}
          RSP_DIR ${CMAKE_CURRENT_BINARY_DIR}
          INPUTS ${BC_DEVICE_LIBS_${arch}})
  set( builtins_link_lib_${arch}
    $<TARGET_PROPERTY:device_lib_device_${arch},TARGET_FILE>)
  add_dependencies(libsycldevice-bc device_lib_device_${arch})
  set( builtins_opt_lib_tgt_${arch} builtins_${arch}.opt)

  # Run the optimizer on the resulting bitcode file and call prepare_builtins
  # on it, which strips away debug and arch information.
  process_bc(devicelib-${arch}.bc
    LIB_TGT builtins_${arch}.opt
    IN_FILE ${builtins_link_lib_${arch}}
    OUT_DIR ${bc_binary_dir}
    OPT_FLAGS ${opt_flags_${arch}}
    DEPENDENCIES device_lib_device_${arch})
  add_dependencies(libsycldevice-bc prepare-devicelib-${arch}.bc)
  set(complete_${arch}_libdev
    $<TARGET_PROPERTY:prepare-devicelib-${arch}.bc,TARGET_FILE>)
  install( FILES ${complete_${arch}_libdev}
           DESTINATION ${install_dest_bc}
           COMPONENT libsycldevice)
endforeach()

# Add host device imf libraries for obj and new offload objects.
foreach(dtype IN ITEMS bf16 fp32 fp64)
  foreach(ftype IN ITEMS obj obj-new-offload)
    set(tgt_name imf_fallback_${dtype}_host_${ftype})

    add_lib_imf(fallback-imf-${dtype}-host
      DIR ${${ftype}_binary_dir}
      FTYPE ${ftype}
      DTYPE ${dtype}
      EXTRA_OPTS ${${ftype}_host_compile_opts}
      TGT_NAME ${tgt_name})

    set(wrapper_name imf_wrapper.cpp)
    if (NOT ("${dtype}" STREQUAL "fp32"))
      set(wrapper_name imf_wrapper_${dtype}.cpp)
    endif()
    add_custom_command(
      OUTPUT ${${ftype}_binary_dir}/imf-${dtype}-host.${${ftype}-suffix}
      COMMAND ${clang_exe} ${${ftype}_host_compile_opts}
              ${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_name}
              -o ${${ftype}_binary_dir}/imf-${dtype}-host.${${ftype}-suffix}
      MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_name}
      DEPENDS ${imf_obj_deps}
      VERBATIM)

    add_custom_target(imf_${dtype}_host_${ftype} DEPENDS
      ${obj_binary_dir}/imf-${dtype}-host.${${ftype}-suffix})
  endforeach()
endforeach()

foreach(ftype IN ITEMS obj obj-new-offload)
  add_custom_target(imf_host_${ftype}
    DEPENDS ${${ftype}_binary_dir}/${devicelib_host_static_${ftype}})
  add_custom_command(
    OUTPUT ${${ftype}_binary_dir}/${devicelib_host_static_${ftype}}
    COMMAND ${llvm-ar_exe} rcs
            ${${ftype}_binary_dir}/${devicelib_host_static_${ftype}}
            ${${ftype}_binary_dir}/imf-fp32-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/fallback-imf-fp32-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/imf-fp64-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/fallback-imf-fp64-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/imf-bf16-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/fallback-imf-bf16-host.${${ftype}-suffix}
    DEPENDS imf_fp32_host_${ftype} imf_fallback_fp32_host_${ftype}
    DEPENDS imf_fp64_host_${ftype} imf_fallback_fp64_host_${ftype}
    DEPENDS imf_bf16_host_${ftype} imf_fallback_bf16_host_${ftype}
    DEPENDS ${llvm-ar_target}
    VERBATIM)
  add_dependencies(libsycldevice-obj imf_host_${ftype})

  install( FILES ${obj_binary_dir}/${devicelib_host_static_${ftype}}
           DESTINATION ${install_dest_obj}
           COMPONENT libsycldevice)
endforeach()

foreach(ftype IN LISTS filetypes)
  install(
    FILES ${${ftype}_binary_dir}/libsycl-fallback-imf.${${ftype}-suffix}
          ${${ftype}_binary_dir}/libsycl-fallback-imf-fp64.${${ftype}-suffix}
          ${${ftype}_binary_dir}/libsycl-fallback-imf-bf16.${${ftype}-suffix}
    DESTINATION ${install_dest_${ftype}}
    COMPONENT libsycldevice)
endforeach()

set(libsycldevice_build_targets)
foreach(filetype IN LISTS filetypes)
  list(APPEND libsycldevice_build_targets libsycldevice-${filetype})
endforeach()

add_custom_target(install-libsycldevice
  COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_COMPONENT=libsycldevice -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
  DEPENDS ${libsycldevice_build_targets}
)
add_dependencies(deploy-sycl-toolchain install-libsycldevice)
