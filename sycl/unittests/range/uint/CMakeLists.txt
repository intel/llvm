add_sycl_unittest(IdQueriesRangeValidationUInt SHARED
  IdQueriesRangeValidation.cpp
)
target_compile_definitions(IdQueriesRangeValidationUInt_Non_Preview_Tests PRIVATE
  __SYCL_ID_QUERIES_FIT_IN_UINT__=1
)
if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
  target_compile_definitions(IdQueriesRangeValidationUInt_Preview_Tests PRIVATE
    __SYCL_ID_QUERIES_FIT_IN_UINT__=1
  )
endif()
