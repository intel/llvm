set(imf_fallback_src_list imf_utils/integer_misc.cpp
                          imf_utils/half_convert.cpp
                          imf_utils/float_convert.cpp
                          imf_utils/simd_emulate.cpp
                          imf_utils/fp32_round.cpp
                          imf/imf_inline_fp32.cpp
                          imf/imf_fp32_dl.cpp
                          imf_utils/double_convert.cpp
                          imf_utils/fp64_round.cpp
                          imf/imf_inline_fp64.cpp
                          imf/imf_fp64_dl.cpp
                          imf_utils/bfloat16_convert.cpp
                          imf/imf_inline_bf16.cpp)

set(imf_fallback_dest ${DEST_DIR}/imf_fallback.cpp)

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
