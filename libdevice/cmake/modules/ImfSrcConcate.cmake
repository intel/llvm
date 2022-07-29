set(imf_fp32_fallback_src_list imf_utils/integer_misc.cpp
                               imf_utils/half_convert.cpp
                               imf_utils/float_convert.cpp
                               imf_utils/simd_emulate.cpp
                               imf/imf_inline_fp32.cpp)

set(imf_fp64_fallback_src_list imf_utils/double_convert.cpp
                               imf/imf_inline_fp64.cpp)

if (FP64 STREQUAL 0)
  set(imf_fallback_src_list ${imf_fp32_fallback_src_list})
  set(imf_fallback_dest ${DEST_DIR}/imf_fp32_fallback.cpp)
else()
  set(imf_fallback_src_list ${imf_fp64_fallback_src_list})
  set(imf_fallback_dest ${DEST_DIR}/imf_fp64_fallback.cpp)
endif()

set(flag 0)
foreach(src ${imf_fallback_src_list})
  file(READ ${SRC_DIR}/${src} src_contents)
  if(flag STREQUAL 0)
    file(WRITE ${imf_fallback_dest} "${src_contents}")
    set(flag 1)
  else()
    file(APPEND ${imf_fallback_dest} "${src_contents}")
  endif()
endforeach()
