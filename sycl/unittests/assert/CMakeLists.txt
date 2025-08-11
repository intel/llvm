include(CheckCXXCompilerFlag)

add_sycl_unittest(AssertTests OBJECT
  assert.cpp
  support_native.cpp
)

# Enabling fallback asserts will cause warnings, so we need to suppress them.
check_cxx_compiler_flag(-Wno-error=\#warnings HAS_NO_HASH_WARNINGS_OPT)
check_cxx_compiler_flag(-Wno-error=pedantic HAS_NO_PEDANTIC_WARNINGS_OPT)
check_cxx_compiler_flag(-Wno-error=cpp HAS_NO_CPP_WARNINGS_OPT)
target_compile_options(
  AssertTests_non_preview
  PRIVATE
    $<$<BOOL:${HAS_NO_HASH_WARNINGS_OPT}>:-Wno-error=\#warnings>
    $<$<BOOL:${HAS_NO_PEDANTIC_WARNINGS_OPT}>:-Wno-error=pedantic>
    $<$<BOOL:${HAS_NO_CPP_WARNINGS_OPT}>:-Wno-error=cpp>
)
target_compile_options(
  AssertTests_preview
  PRIVATE
    $<$<BOOL:${HAS_NO_HASH_WARNINGS_OPT}>:-Wno-error=\#warnings>
    $<$<BOOL:${HAS_NO_PEDANTIC_WARNINGS_OPT}>:-Wno-error=pedantic>
    $<$<BOOL:${HAS_NO_CPP_WARNINGS_OPT}>:-Wno-error=cpp>
)
