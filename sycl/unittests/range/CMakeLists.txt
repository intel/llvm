# Test for INT mode - define __SYCL_ID_QUERIES_FIT_IN_INT__
add_sycl_unittest(IdQueriesRangeValidationInt OBJECT
  IdQueriesRangeValidation.cpp
)
target_compile_definitions(IdQueriesRangeValidationInt_Non_Preview_Tests PRIVATE
  __SYCL_ID_QUERIES_FIT_IN_INT__=1
)

# Test for UINT mode - define __SYCL_ID_QUERIES_FIT_IN_UINT__
add_sycl_unittest(IdQueriesRangeValidationUInt OBJECT
  IdQueriesRangeValidation.cpp
)
target_compile_definitions(IdQueriesRangeValidationUInt_Non_Preview_Tests PRIVATE
  __SYCL_ID_QUERIES_FIT_IN_UINT__=1
)

# Test for size_t mode - no macros defined (default case)
add_sycl_unittest(IdQueriesRangeValidationSizeT OBJECT
  IdQueriesRangeValidation.cpp
)
# No compile definitions needed - size_t mode is the default when neither macro is defined
