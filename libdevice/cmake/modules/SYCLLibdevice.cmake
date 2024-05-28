set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(obj_new_offload_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (MSVC)
  set(lib-suffix obj)
  set(new-offload-lib-suffix new.obj)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(install_dest_spv bin)
  set(devicelib_host_static sycl-devicelib-host.lib)
  set(devicelib_host_static_new_offload sycl-devicelib-host.new.lib)
else()
  set(lib-suffix o)
  set(new-offload-lib-suffix new.o)
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(install_dest_spv lib${LLVM_LIBDIR_SUFFIX})
  set(devicelib_host_static libsycl-devicelib-host.a)
  set(devicelib_host_static_new_offload libsycl-devicelib-host.new.a)
endif()
set(bc_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(install_dest_lib lib${LLVM_LIBDIR_SUFFIX})
set(install_dest_bc lib${LLVM_LIBDIR_SUFFIX})

set(clang $<TARGET_FILE:clang>)
set(llvm-ar $<TARGET_FILE:llvm-ar>)
set(llvm-link $<TARGET_FILE:llvm-link>)
set(llvm-opt $<TARGET_FILE:opt>)

string(CONCAT sycl_targets_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown,"
  "spir64_gen-unknown-unknown,"
  "spir64_fpga-unknown-unknown,"
  "spir64-unknown-unknown,"
  "spirv64-unknown-unknown")

set(compile_opts
  # suppress an error about SYCL_EXTERNAL being used for
  # a function with a raw pointer parameter.
  -Wno-sycl-strict
  # Disable warnings for the host compilation, where
  # we declare all functions as 'static'.
  -Wno-undefined-internal
  -sycl-std=2020
  )

set(SYCL_LIBDEVICE_GCC_TOOLCHAIN "" CACHE PATH "Path to GCC installation")

if (NOT SYCL_LIBDEVICE_GCC_TOOLCHAIN STREQUAL "")
  list(APPEND compile_opts "--gcc-toolchain=${SYCL_LIBDEVICE_GCC_TOOLCHAIN}")
endif()

set(devicelib_arch)

if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  string(APPEND sycl_targets_opt ",nvptx64-nvidia-cuda")
  list(APPEND compile_opts
    "-fno-sycl-libspirv"
    "-fno-bundle-offload-arch"
    "-nocudalib"
    "--cuda-gpu-arch=sm_50")
  set(devicelib_arch "NVPTX")
elseif("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  string(APPEND sycl_targets_opt ",amdgcn-amd-amdhsa")
  list(APPEND compile_opts
    "fno-sycl-libspirv"
    "-fno-bundle-offload-arch"
    # Compile it to a high bc version. The arch info gets removed later.
    # ToDo Do we need it for atomic clang builtin access?
    "--offload-arch=gfx940")
  set(devicelib_arch "AMDGPU")
endif()

if (WIN32)
  list(APPEND compile_opts -D_ALLOW_RUNTIME_LIBRARY_MISMATCH)
  list(APPEND compile_opts -D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

add_custom_target(libsycldevice-obj)
add_custom_target(libsycldevice-obj-new-offload)
add_custom_target(libsycldevice-spv)
add_custom_target(libsycldevice-bc)

add_custom_target(libsycldevice DEPENDS
  libsycldevice-obj
  libsycldevice-bc
  libsycldevice-obj-new-offload
  libsycldevice-spv)

function(add_devicelib_obj obj_filename)
  cmake_parse_arguments(OBJ  "" "" "SRC;DEP;EXTRA_ARGS" ${ARGN})
  set(devicelib-obj-file ${obj_binary_dir}/${obj_filename}.${lib-suffix})
  add_custom_command(OUTPUT ${devicelib-obj-file}
                     COMMAND ${clang} -fsycl -c
                             ${compile_opts} ${sycl_targets_opt} ${OBJ_EXTRA_ARGS}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${OBJ_SRC}
                             -o ${devicelib-obj-file}
                     MAIN_DEPENDENCY ${OBJ_SRC}
                     DEPENDS ${OBJ_DEP}
                     VERBATIM)
  set(devicelib-obj-target ${obj_filename}-obj)
  add_custom_target(${devicelib-obj-target} DEPENDS ${devicelib-obj-file})
  add_dependencies(libsycldevice-obj ${devicelib-obj-target})
  install(FILES ${devicelib-obj-file}
          DESTINATION ${install_dest_lib}
          COMPONENT libsycldevice)

  set(devicelib-obj-file-new-offload ${obj_new_offload_binary_dir}/${obj_filename}.${new-offload-lib-suffix})
  add_custom_command(OUTPUT ${devicelib-obj-file-new-offload}
                     COMMAND ${clang} -fsycl -c --offload-new-driver -foffload-lto=thin
                             ${compile_opts} ${sycl_targets_opt} ${OBJ_EXTRA_ARGS}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${OBJ_SRC}
                             -o ${devicelib-obj-file-new-offload}
                     MAIN_DEPENDENCY ${OBJ_SRC}
                     DEPENDS ${OBJ_DEP}
                     VERBATIM)
  set(devicelib-obj-target-new-offload ${obj_filename}-new-offload-obj)
  add_custom_target(${devicelib-obj-target-new-offload} DEPENDS ${devicelib-obj-file-new-offload})
  add_dependencies(libsycldevice-obj ${devicelib-obj-target-new-offload})
  install(FILES ${devicelib-obj-file-new-offload}
          DESTINATION ${install_dest_lib}
          COMPONENT libsycldevice)
endfunction()

function(add_devicelib_spv spv_filename)
  cmake_parse_arguments(SPV  "" "" "SRC;DEP;EXTRA_ARGS" ${ARGN})
  set(devicelib-spv-file ${spv_binary_dir}/${spv_filename}.spv)
  add_custom_command(OUTPUT ${devicelib-spv-file}
                     COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=spirv
                             ${compile_opts} ${SPV_EXTRA_ARGS}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${SPV_SRC}
                             -o ${devicelib-spv-file}
                     MAIN_DEPENDENCY ${SPV_SRC}
                     DEPENDS ${SPV_DEP}
                     VERBATIM)
  set(devicelib-spv-target ${spv_filename}-spv)
  add_custom_target(${devicelib-spv-target} DEPENDS ${devicelib-spv-file})
  add_dependencies(libsycldevice-spv ${devicelib-spv-target})
  install(FILES ${devicelib-spv-file}
          DESTINATION ${install_dest_spv}
          COMPONENT libsycldevice)
endfunction()

# Links together one or more bytecode files
#
# Arguments:
# * TARGET <string>
#     Custom target to create
# * INPUT <string> ...
#     List of bytecode files to link together
function(link_bc)
  cmake_parse_arguments(ARG
    ""
    "TARGET"
    "INPUTS"
    ${ARGN}
  )

  set( LINK_INPUT_ARG ${ARG_INPUTS} )
  if( WIN32 OR CYGWIN )
    # Create a response file in case the number of inputs exceeds command-line
    # character limits on certain platforms.
    file( TO_CMAKE_PATH ${bc_binary_dir}/${ARG_TARGET}.rsp RSP_FILE )
    # Turn it into a space-separate list of input files
    list( JOIN ARG_INPUTS " " RSP_INPUT )
    file( WRITE ${RSP_FILE} ${RSP_INPUT} )
    # Ensure that if this file is removed, we re-run CMake
    set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${RSP_FILE}
    )
    set( LINK_INPUT_ARG "@${RSP_FILE}" )
  endif()

  add_custom_command(
    #ToDo maybe add bc_bianry_dir to path
    OUTPUT ${ARG_TARGET}.bc
    COMMAND ${llvm-link} -o ${bc_binary_dir}/${ARG_TARGET}.bc ${LINK_INPUT_ARG}
    DEPENDS ${llvm-link} ${ARG_INPUTS} ${RSP_FILE}
  )

  add_custom_target( ${ARG_TARGET} ALL DEPENDS ${ARG_TARGET}.bc )
  set_target_properties( ${ARG_TARGET} PROPERTIES TARGET_FILE ${ARG_TARGET}.bc )
endfunction()

function(append_to_property arg)
  get_property(BC_DEVICE_LIBS GLOBAL PROPERTY BC_DEVICE_LIBS)
  list(APPEND BC_DEVICE_LIBS ${arg})
  set_property(GLOBAL PROPERTY BC_DEVICE_LIBS ${BC_DEVICE_LIBS})
endfunction()

function(add_devicelib_bc bc_filename)
  cmake_parse_arguments(BC  "CUDA;AMD" "" "SRC;DEP;EXTRA_ARGS" ${ARGN})
  list(APPEND compile_opts "-fsycl-device-only" "-fsycl-device-obj=llvmir")

  if(${BC_CUDA})
    list(APPEND compile_opts "-fsycl-targets=nvptx64-nvidia-cuda")
    set (bc_filename ${bc_filename}--cuda)
  elseif(${BC_AMD})
    list(APPEND compile_opts "-Xsycl-target-backend=amdgcn-amd-amdhsa")
    set (bc_filename ${bc_filename}--amd)
  endif()

  set(devicelib-bc-file ${bc_binary_dir}/${bc_filename}.bc)

  add_custom_command(OUTPUT ${devicelib-bc-file}
                     COMMAND ${clang}
                             ${compile_opts}
                             ${BC_EXTRA_ARGS}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${BC_SRC}
                             -o ${devicelib-bc-file}
                     MAIN_DEPENDENCY ${BC_SRC}
                     DEPENDS ${BC_DEP}
                     VERBATIM)

  set(devicelib-bc-target ${bc_filename}-bc)
  add_custom_target(${devicelib-bc-target} DEPENDS ${devicelib-bc-file})
  add_dependencies(libsycldevice-bc ${devicelib-bc-target})
  install(FILES ${devicelib-bc-file}
          DESTINATION ${install_dest_bc}
          COMPONENT libsycldevice)

   if(${BC_CUDA} OR ${BC_AMD})
    append_to_property(${devicelib-bc-file})
  endif()
endfunction()

function(add_devicelib filename)
  cmake_parse_arguments(DL "" "" "SRC;DEP;EXTRA_ARGS" ${ARGN})

  add_devicelib_spv(${filename} SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
  add_devicelib_bc(${filename} dummy SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
  add_devicelib_obj(${filename} SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
  if (${devicelib_arch} STREQUAL "NVPTX")
    add_devicelib_bc(${filename} CUDA SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
  elseif (${devicelib_arch} STREQUAL "AMDGPU")
    add_devicelib_bc(${filename} AMD SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
  endif()
endfunction()

if (${devicelib_arch} STREQUAL "NVPTX")
  add_devicelib_bc(${filename} CUDA SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
elseif (${devicelib_arch} STREQUAL "AMDGPU")
  add_devicelib_bc(${filename} AMD SRC ${DL_SRC} DEP ${DL_DEP} EXTRA_ARGS ${DL_EXTRA_ARGS})
endif()

set(crt_obj_deps wrapper.h device.h spirv_vars.h sycl-compiler)
set(complex_obj_deps device_complex.h device.h sycl-compiler)
set(cmath_obj_deps device_math.h device.h sycl-compiler)
set(imf_obj_deps device_imf.hpp imf_half.hpp imf_bf16.hpp imf_rounding_op.hpp imf_impl_utils.hpp device.h sycl-compiler)
set(itt_obj_deps device_itt.h spirv_vars.h device.h sycl-compiler)
set(bfloat16_obj_deps sycl-headers sycl-compiler)
if (NOT MSVC)
  set(sanitizer_obj_deps
    device.h atomic.hpp spirv_vars.h
    include/asan_libdevice.hpp
    include/sanitizer_utils.hpp
    include/spir_global_var.hpp
    sycl-compiler)
endif()

add_devicelib(libsycl-itt-stubs SRC itt_stubs.cpp DEP ${itt_obj_deps})
add_devicelib(libsycl-itt-compiler-wrappers SRC itt_compiler_wrappers.cpp DEP ${itt_obj_deps})
add_devicelib(libsycl-itt-user-wrappers SRC itt_user_wrappers.cpp DEP ${itt_obj_deps})

get_property(BC_DEVICE_LIBS GLOBAL PROPERTY BC_DEVICE_LIBS)
add_devicelib(libsycl-crt SRC crt_wrapper.cpp DEP ${crt_obj_deps})
add_devicelib(libsycl-complex SRC complex_wrapper.cpp DEP ${complex_obj_deps})
add_devicelib(libsycl-complex-fp64 SRC complex_wrapper_fp64.cpp DEP ${complex_obj_deps} )
add_devicelib(libsycl-cmath SRC cmath_wrapper.cpp DEP ${cmath_obj_deps})
add_devicelib(libsycl-cmath-fp64 SRC cmath_wrapper_fp64.cpp DEP ${cmath_obj_deps} )
add_devicelib(libsycl-imf SRC imf_wrapper.cpp DEP ${imf_obj_deps})
add_devicelib(libsycl-imf-fp64 SRC imf_wrapper_fp64.cpp DEP ${imf_obj_deps})
add_devicelib(libsycl-imf-bf16 SRC imf_wrapper_bf16.cpp DEP ${imf_obj_deps})
add_devicelib(libsycl-bfloat16 SRC bfloat16_wrapper.cpp DEP ${cmath_obj_deps} )
if(MSVC)
  add_devicelib(libsycl-msvc-math SRC msvc_math.cpp DEP ${cmath_obj_deps})
else()
  add_devicelib(libsycl-sanitizer SRC sanitizer_utils.cpp DEP ${sanitizer_obj_deps} EXTRA_ARGS -fno-sycl-instrument-device-code)
endif()

add_devicelib(libsycl-fallback-cassert SRC fallback-cassert.cpp DEP ${crt_obj_deps} EXTRA_ARGS -fno-sycl-instrument-device-code)
add_devicelib(libsycl-fallback-cstring SRC fallback-cstring.cpp DEP ${crt_obj_deps})
add_devicelib(libsycl-fallback-complex SRC fallback-complex.cpp DEP ${complex_obj_deps})
add_devicelib(libsycl-fallback-complex-fp64 SRC fallback-complex-fp64.cpp DEP ${complex_obj_deps} )
add_devicelib(libsycl-fallback-cmath SRC fallback-cmath.cpp DEP ${cmath_obj_deps})
add_devicelib(libsycl-fallback-cmath-fp64 SRC fallback-cmath-fp64.cpp DEP ${cmath_obj_deps})
add_devicelib(libsycl-fallback-bfloat16 SRC fallback-bfloat16.cpp DEP ${bfloat16_obj_deps})
add_devicelib(libsycl-native-bfloat16 SRC bfloat16_wrapper.cpp DEP ${bfloat16_obj_deps})

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
                           imf_utils/double_convert.cpp
                           imf_utils/fp64_round.cpp
                           imf/imf_inline_fp64.cpp
                           imf/imf_fp64_dl.cpp)
set(imf_fallback_bf16_deps device.h device_imf.hpp imf_bf16.hpp
                           imf_utils/bfloat16_convert.cpp
                           imf/imf_inline_bf16.cpp)

set(imf_fp32_fallback_src ${imf_fallback_src_dir}/imf_fp32_fallback.cpp)
set(imf_fp64_fallback_src ${imf_fallback_src_dir}/imf_fp64_fallback.cpp)
set(imf_bf16_fallback_src ${imf_fallback_src_dir}/imf_bf16_fallback.cpp)

set(imf_host_cxx_flags -c
  -D__LIBDEVICE_HOST_IMPL__
)

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

add_custom_command(OUTPUT ${imf_fp32_fallback_src}
                   COMMAND ${CMAKE_COMMAND} -D SRC_DIR=${imf_src_dir}
                                            -D DEST_DIR=${imf_fallback_src_dir}
                                            -D IMF_TARGET=FP32
                                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ImfSrcConcate.cmake
                   DEPENDS ${imf_fallback_fp32_deps})

add_custom_command(OUTPUT ${imf_fp64_fallback_src}
                   COMMAND ${CMAKE_COMMAND} -D SRC_DIR=${imf_src_dir}
                                            -D DEST_DIR=${imf_fallback_src_dir}
                                            -D IMF_TARGET=FP64
                                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ImfSrcConcate.cmake
                   DEPENDS ${imf_fallback_fp64_deps})

add_custom_command(OUTPUT ${imf_bf16_fallback_src}
                   COMMAND ${CMAKE_COMMAND} -D SRC_DIR=${imf_src_dir}
                                            -D DEST_DIR=${imf_fallback_src_dir}
                                            -D IMF_TARGET=BF16
                                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ImfSrcConcate.cmake
                   DEPENDS ${imf_fallback_bf16_deps})

add_custom_target(get_imf_fallback_fp32  DEPENDS ${imf_fp32_fallback_src})
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-imf.spv
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=spirv
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp32_fallback_src}
                           -o ${spv_binary_dir}/libsycl-fallback-imf.spv
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${bc_binary_dir}/libsycl-fallback-imf.bc
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=llvmir
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp32_fallback_src}
                           -o ${bc_binary_dir}/libsycl-fallback-imf.bc
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32
                           sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${bc_binary_dir}/libsycl-fallback-imf--cuda.bc
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=llvmir
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp32_fallback_src} -fsycl-targets=nvptx64-nvidia-cuda
                           -o ${bc_binary_dir}/libsycl-fallback-imf--cuda.bc
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32
                           sycl-compiler
                   VERBATIM)

append_to_property(${bc_binary_dir}/libsycl-fallback-imf--cuda.bc)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-imf.${lib-suffix}
                   COMMAND ${clang} -fsycl -c
                           ${compile_opts} ${sycl_targets_opt}
                           ${imf_fp32_fallback_src} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           -o ${obj_binary_dir}/libsycl-fallback-imf.${lib-suffix}
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-imf.${new-offload-lib-suffix}
                   COMMAND ${clang} -fsycl -c --offload-new-driver -foffload-lto=thin
                           ${compile_opts} ${sycl_targets_opt}
                           ${imf_fp32_fallback_src} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           -o ${obj_binary_dir}/libsycl-fallback-imf.${new-offload-lib-suffix}
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-imf-fp32-host.${lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags}
                           -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp32_fallback_src}
                           -o ${obj_binary_dir}/fallback-imf-fp32-host.${lib-suffix}
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-imf-fp32-host.${new-offload-lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags} --offload-new-driver -foffload-lto=thin
                           -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp32_fallback_src}
                           -o ${obj_binary_dir}/fallback-imf-fp32-host.${new-offload-lib-suffix}
                   DEPENDS ${imf_fallback_fp32_deps} get_imf_fallback_fp32 sycl-compiler
                   VERBATIM)

add_custom_target(get_imf_fallback_fp64  DEPENDS ${imf_fp64_fallback_src})
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-imf-fp64.spv
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=spirv
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp64_fallback_src}
                           -o ${spv_binary_dir}/libsycl-fallback-imf-fp64.spv
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${bc_binary_dir}/libsycl-fallback-imf-fp64.bc
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=llvmir
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp64_fallback_src}
                           -o ${bc_binary_dir}/libsycl-fallback-imf-fp64.bc
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64
                           sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${bc_binary_dir}/libsycl-fallback-imf-fp64--cuda.bc
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=llvmir
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp64_fallback_src} -fsycl-targets=nvptx64-nvidia-cuda
                           -o ${bc_binary_dir}/libsycl-fallback-imf-fp64--cuda.bc
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64
                           sycl-compiler
                   VERBATIM)

append_to_property(${bc_binary_dir}/libsycl-fallback-imf-fp64--cuda.bc)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-imf-fp64.${lib-suffix}
                   COMMAND ${clang} -fsycl -c -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${compile_opts} ${sycl_targets_opt}
                           ${imf_fp64_fallback_src}
                           -o ${obj_binary_dir}/libsycl-fallback-imf-fp64.${lib-suffix}
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-imf-fp64.${new-offload-lib-suffix}
                   COMMAND ${clang} -fsycl -c -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           --offload-new-driver -foffload-lto=thin
                           ${compile_opts} ${sycl_targets_opt}
                           ${imf_fp64_fallback_src}
                           -o ${obj_binary_dir}/libsycl-fallback-imf-fp64.${new-offload-lib-suffix}
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-imf-fp64-host.${lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags}
                           -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp64_fallback_src}
                           -o ${obj_binary_dir}/fallback-imf-fp64-host.${lib-suffix}
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-imf-fp64-host.${new-offload-lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags} --offload-new-driver -foffload-lto=thin
                           -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_fp64_fallback_src}
                           -o ${obj_binary_dir}/fallback-imf-fp64-host.${new-offload-lib-suffix}
                   DEPENDS ${imf_fallback_fp64_deps} get_imf_fallback_fp64 sycl-compiler
                   VERBATIM)

add_custom_target(get_imf_fallback_bf16  DEPENDS ${imf_bf16_fallback_src})
add_custom_command(OUTPUT ${spv_binary_dir}/libsycl-fallback-imf-bf16.spv
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=spirv
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_bf16_fallback_src}
                           -o ${spv_binary_dir}/libsycl-fallback-imf-bf16.spv
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${bc_binary_dir}/libsycl-fallback-imf-bf16.bc
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=llvmir
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_bf16_fallback_src}
                           -o ${bc_binary_dir}/libsycl-fallback-imf-bf16.bc
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16
                           sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${bc_binary_dir}/libsycl-fallback-imf-bf16--cuda.bc
                   COMMAND ${clang} -fsycl-device-only -fsycl-device-obj=llvmir
                           ${compile_opts} -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_bf16_fallback_src} -fsycl-targets=nvptx64-nvidia-cuda
                           -o ${bc_binary_dir}/libsycl-fallback-imf-bf16--cuda.bc
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16
                           sycl-compiler
                   VERBATIM)

append_to_property(${bc_binary_dir}/libsycl-fallback-imf-bf16--cuda.bc)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-imf-bf16.${lib-suffix}
                   COMMAND ${clang} -fsycl -c -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${compile_opts} ${sycl_targets_opt}
                           ${imf_bf16_fallback_src}
                           -o ${obj_binary_dir}/libsycl-fallback-imf-bf16.${lib-suffix}
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/libsycl-fallback-imf-bf16.${new-offload-lib-suffix}
                   COMMAND ${clang} -fsycl -c -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           --offload-new-driver -foffload-lto=thin
                           ${compile_opts} ${sycl_targets_opt}
                           ${imf_bf16_fallback_src}
                           -o ${obj_binary_dir}/libsycl-fallback-imf-bf16.${new-offload-lib-suffix}
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-imf-bf16-host.${lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags}
                           -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_bf16_fallback_src}
                           -o ${obj_binary_dir}/fallback-imf-bf16-host.${lib-suffix}
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16 sycl-compiler
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/fallback-imf-bf16-host.${new-offload-lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags} --offload-new-driver -foffload-lto=thin
                           -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
                           ${imf_bf16_fallback_src}
                           -o ${obj_binary_dir}/fallback-imf-bf16-host.${new-offload-lib-suffix}
                   DEPENDS ${imf_fallback_bf16_deps} get_imf_fallback_bf16 sycl-compiler
                   VERBATIM)

add_custom_target(imf_fallback_fp32_spv DEPENDS ${spv_binary_dir}/libsycl-fallback-imf.spv)
add_custom_target(imf_fallback_fp32_bc DEPENDS ${bc_binary_dir}/libsycl-fallback-imf.bc)
add_custom_target(imf_fallback_fp32_bc_cuda DEPENDS ${bc_binary_dir}/libsycl-fallback-imf--cuda.bc)
add_custom_target(imf_fallback_fp32_obj DEPENDS ${obj_binary_dir}/libsycl-fallback-imf.${lib-suffix})
add_custom_target(imf_fallback_fp32_host_obj DEPENDS ${obj_binary_dir}/fallback-imf-fp32-host.${lib-suffix})
add_custom_target(imf_fallback_fp32_new_offload_obj DEPENDS ${obj_binary_dir}/libsycl-fallback-imf.${new-offload-lib-suffix})
add_custom_target(imf_fallback_fp32_host_new_offload_obj DEPENDS ${obj_binary_dir}/fallback-imf-fp32-host.${new-offload-lib-suffix})
add_dependencies(libsycldevice-spv imf_fallback_fp32_spv)
add_dependencies(libsycldevice-bc imf_fallback_fp32_bc)
add_dependencies(libsycldevice-bc imf_fallback_fp32_bc_cuda)
add_dependencies(libsycldevice-obj imf_fallback_fp32_obj)
add_dependencies(libsycldevice-obj imf_fallback_fp32_new_offload_obj)

add_custom_target(imf_fallback_fp64_spv DEPENDS ${spv_binary_dir}/libsycl-fallback-imf-fp64.spv)
add_custom_target(imf_fallback_fp64_bc DEPENDS ${bc_binary_dir}/libsycl-fallback-imf-fp64.bc)
add_custom_target(imf_fallback_fp64_bc_cuda DEPENDS ${bc_binary_dir}/libsycl-fallback-imf-fp64--cuda.bc)
add_custom_target(imf_fallback_fp64_obj DEPENDS ${obj_binary_dir}/libsycl-fallback-imf-fp64.${lib-suffix})
add_custom_target(imf_fallback_fp64_host_obj DEPENDS ${obj_binary_dir}/fallback-imf-fp64-host.${lib-suffix})
add_custom_target(imf_fallback_fp64_new_offload_obj DEPENDS ${obj_binary_dir}/libsycl-fallback-imf-fp64.${new-offload-lib-suffix})
add_custom_target(imf_fallback_fp64_host_new_offload_obj DEPENDS ${obj_binary_dir}/fallback-imf-fp64-host.${new-offload-lib-suffix})
add_dependencies(libsycldevice-spv imf_fallback_fp64_spv)
add_dependencies(libsycldevice-bc imf_fallback_fp64_bc)
add_dependencies(libsycldevice-bc imf_fallback_fp64_bc_cuda)
add_dependencies(libsycldevice-obj imf_fallback_fp64_obj)
add_dependencies(libsycldevice-obj imf_fallback_fp64_new_offload_obj)

add_custom_target(imf_fallback_bf16_spv DEPENDS ${spv_binary_dir}/libsycl-fallback-imf-bf16.spv)
add_custom_target(imf_fallback_bf16_bc DEPENDS ${bc_binary_dir}/libsycl-fallback-imf-bf16.bc)
add_custom_target(imf_fallback_bf16_bc_cuda DEPENDS  ${bc_binary_dir}/libsycl-fallback-imf-bf16--cuda.bc)
add_custom_target(imf_fallback_bf16_obj DEPENDS ${obj_binary_dir}/libsycl-fallback-imf-bf16.${lib-suffix})
add_custom_target(imf_fallback_bf16_host_obj DEPENDS ${obj_binary_dir}/fallback-imf-bf16-host.${lib-suffix})
add_custom_target(imf_fallback_bf16_new_offload_obj DEPENDS ${obj_binary_dir}/libsycl-fallback-imf-bf16.${new-offload-lib-suffix})
add_custom_target(imf_fallback_bf16_host_new_offload_obj DEPENDS ${obj_binary_dir}/fallback-imf-bf16-host.${new-offload-lib-suffix})
add_dependencies(libsycldevice-spv imf_fallback_bf16_spv)
add_dependencies(libsycldevice-bc imf_fallback_bf16_bc)
add_dependencies(libsycldevice-bc imf_fallback_bf16_bc_cuda)
add_dependencies(libsycldevice-obj imf_fallback_bf16_obj)
add_dependencies(libsycldevice-obj imf_fallback_bf16_new_offload_obj)

get_property(BC_DEVICE_LIBS GLOBAL PROPERTY BC_DEVICE_LIBS)

link_bc(TARGET device_lib_device INPUTS ${BC_DEVICE_LIBS})

# -----------------------------------------------------------------------------------------------------

set( builtins_link_lib $<TARGET_PROPERTY:device_lib_device,TARGET_FILE> )

set( builtins_opt_lib_tgt builtins.opt)

# Add opt target
add_custom_command( OUTPUT ${builtins_opt_lib_tgt}.bc
   COMMAND ${llvm-opt} ${ARG_OPT_FLAGS} -o ${builtins_opt_lib_tgt}.bc
       ${bc_binary_dir}/${builtins_link_lib}
     DEPENDS ${llvm-opt} ${builtins_link_lib} ${device_lib_device}
   )
add_custom_target( ${builtins_opt_lib_tgt}
     ALL DEPENDS ${builtins_opt_lib_tgt}.bc
   )
set_target_properties( ${builtins_opt_lib_tgt}
     PROPERTIES TARGET_FILE ${builtins_opt_lib_tgt}.bc
   )

set( builtins_opt_lib $<TARGET_PROPERTY:${builtins_opt_lib_tgt},TARGET_FILE> )

# Add prepare target
if (${devicelib_arch} STREQUAL "NVPTX")
  set(obj_suffix "devicelib--nvptx.bc")
elseif (${devicelib_arch} STREQUAL "AMDGPU")
  set(obj_suffix "devicelib--amd.bc")
endif()

add_custom_command( OUTPUT ${LLVM_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LLVM_LIBRARY_OUTPUT_INTDIR}
  COMMAND prepare_builtins -o ${LLVM_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
    ${builtins_opt_lib}
  DEPENDS ${builtins_opt_lib} prepare_builtins )
add_custom_target( prepare-${obj_suffix} ALL
  DEPENDS ${LLVM_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
)
set_target_properties( prepare-${obj_suffix}
  PROPERTIES TARGET_FILE ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
)

add_dependencies(libsycldevice-bc prepare-${obj_suffix})
set( builtins_lib $<TARGET_PROPERTY:prepare-${obj_suffix},TARGET_FILE> )

# ----------------------------------------------------------------------------------------------

add_dependencies(libsycldevice-bc device_lib_device)

add_custom_command(OUTPUT ${obj_binary_dir}/imf-fp32-host.${lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags}
                           ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper.cpp
                           -o ${obj_binary_dir}/imf-fp32-host.${lib-suffix}
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper.cpp
                   DEPENDS ${imf_obj_deps}
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/imf-fp32-host.${new-offload-lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags} --offload-new-driver -foffload-lto=thin
                           ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper.cpp
                           -o ${obj_binary_dir}/imf-fp32-host.${new-offload-lib-suffix}
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper.cpp
                   DEPENDS ${imf_obj_deps}
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/imf-fp64-host.${lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags}
                           ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_fp64.cpp
                           -o ${obj_binary_dir}/imf-fp64-host.${lib-suffix}
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_fp64.cpp
                   DEPENDS ${imf_obj_deps}
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/imf-fp64-host.${new-offload-lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags} --offload-new-driver -foffload-lto=thin
                           ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_fp64.cpp
                           -o ${obj_binary_dir}/imf-fp64-host.${new-offload-lib-suffix}
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_fp64.cpp
                   DEPENDS ${imf_obj_deps}
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/imf-bf16-host.${lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags}
                           ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_bf16.cpp
                           -o ${obj_binary_dir}/imf-bf16-host.${lib-suffix}
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_bf16.cpp
                   DEPENDS ${imf_obj_deps}
                   VERBATIM)

add_custom_command(OUTPUT ${obj_binary_dir}/imf-bf16-host.${new-offload-lib-suffix}
                   COMMAND ${clang} ${imf_host_cxx_flags} --offload-new-driver -foffload-lto=thin
                           ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_bf16.cpp
                           -o ${obj_binary_dir}/imf-bf16-host.${new-offload-lib-suffix}
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/imf_wrapper_bf16.cpp
                   DEPENDS ${imf_obj_deps}
                   VERBATIM)

add_custom_target(imf_fp32_host_obj DEPENDS ${obj_binary_dir}/imf-fp32-host.${lib-suffix})
add_custom_target(imf_fp64_host_obj DEPENDS ${obj_binary_dir}/imf-fp64-host.${lib-suffix})
add_custom_target(imf_bf16_host_obj DEPENDS ${obj_binary_dir}/imf-bf16-host.${lib-suffix})

add_custom_target(imf_fp32_host_new_offload_obj DEPENDS ${obj_binary_dir}/imf-fp32-host.${new-offload-lib-suffix})
add_custom_target(imf_fp64_host_new_offload_obj DEPENDS ${obj_binary_dir}/imf-fp64-host.${new-offload-lib-suffix})
add_custom_target(imf_bf16_host_new_offload_obj DEPENDS ${obj_binary_dir}/imf-bf16-host.${new-offload-lib-suffix})

add_custom_target(imf_host_obj DEPENDS ${obj_binary_dir}/${devicelib_host_static})
add_custom_command(OUTPUT ${obj_binary_dir}/${devicelib_host_static}
                  COMMAND ${llvm-ar} rcs ${obj_binary_dir}/${devicelib_host_static}
                          ${obj_binary_dir}/imf-fp32-host.${lib-suffix}
                          ${obj_binary_dir}/fallback-imf-fp32-host.${lib-suffix}
                          ${obj_binary_dir}/imf-fp64-host.${lib-suffix}
                          ${obj_binary_dir}/fallback-imf-fp64-host.${lib-suffix}
                          ${obj_binary_dir}/imf-bf16-host.${lib-suffix}
                          ${obj_binary_dir}/fallback-imf-bf16-host.${lib-suffix}
                  DEPENDS imf_fp32_host_obj imf_fallback_fp32_host_obj
                  DEPENDS imf_fp64_host_obj imf_fallback_fp64_host_obj
                  DEPENDS imf_bf16_host_obj imf_fallback_bf16_host_obj
                  DEPENDS sycl-compiler
                  VERBATIM)
add_custom_target(imf_host_new_offload_obj DEPENDS ${obj_binary_dir}/${devicelib_host_static_new_offload})
add_custom_command(OUTPUT ${obj_binary_dir}/${devicelib_host_static_new_offload}
                  COMMAND ${llvm-ar} rcs ${obj_binary_dir}/${devicelib_host_static_new_offload}
                          ${obj_binary_dir}/imf-fp32-host.${new-offload-lib-suffix}
                          ${obj_binary_dir}/fallback-imf-fp32-host.${new-offload-lib-suffix}
                          ${obj_binary_dir}/imf-fp64-host.${new-offload-lib-suffix}
                          ${obj_binary_dir}/fallback-imf-fp64-host.${new-offload-lib-suffix}
                          ${obj_binary_dir}/imf-bf16-host.${new-offload-lib-suffix}
                          ${obj_binary_dir}/fallback-imf-bf16-host.${new-offload-lib-suffix}
                  DEPENDS imf_fp32_host_new_offload_obj imf_fallback_fp32_host_new_offload_obj
                  DEPENDS imf_fp64_host_new_offload_obj imf_fallback_fp64_host_new_offload_obj
                  DEPENDS imf_bf16_host_new_offload_obj imf_fallback_bf16_host_new_offload_obj
                  DEPENDS sycl-compiler
                  VERBATIM)
add_dependencies(libsycldevice-obj imf_host_obj)
add_dependencies(libsycldevice-obj imf_host_new_offload_obj)
install(FILES ${spv_binary_dir}/libsycl-fallback-imf.spv
              ${spv_binary_dir}/libsycl-fallback-imf-fp64.spv
              ${spv_binary_dir}/libsycl-fallback-imf-bf16.spv
        DESTINATION ${install_dest_spv}
        COMPONENT libsycldevice)

install(FILES ${bc_binary_dir}/libsycl-fallback-imf.bc
              ${bc_binary_dir}/libsycl-fallback-imf-fp64.bc
              ${bc_binary_dir}/libsycl-fallback-imf-bf16.bc
              ${bc_binary_dir}/libsycl-fallback-imf--cuda.bc
              ${bc_binary_dir}/libsycl-fallback-imf-fp64--cuda.bc
              ${bc_binary_dir}/libsycl-fallback-imf-bf16--cuda.bc
        DESTINATION ${install_dest_bc}
        COMPONENT libsycldevice)

install(FILES ${obj_binary_dir}/libsycl-fallback-imf.${lib-suffix}
              ${obj_binary_dir}/libsycl-fallback-imf-fp64.${lib-suffix}
              ${obj_binary_dir}/libsycl-fallback-imf-bf16.${lib-suffix}
              ${obj_binary_dir}/${devicelib_host_static}
        DESTINATION ${install_dest_lib}
        COMPONENT libsycldevice)

install(FILES ${obj_binary_dir}/libsycl-fallback-imf.${new-offload-lib-suffix}
              ${obj_binary_dir}/libsycl-fallback-imf-fp64.${new-offload-lib-suffix}
              ${obj_binary_dir}/libsycl-fallback-imf-bf16.${new-offload-lib-suffix}
              ${obj_binary_dir}/${devicelib_host_static_new_offload}
        DESTINATION ${install_dest_lib}
        COMPONENT libsycldevice)
