add_sycl_unittest(IdQueriesRangeValidationInt SHARED
  IdQueriesRangeValidation.cpp
)
target_compile_definitions(IdQueriesRangeValidationInt-Non_Preview_Tests PRIVATE
  __SYCL_ID_QUERIES_FIT_IN_INT__=1
)
if(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB)
  target_compile_definitions(IdQueriesRangeValidationInt-Preview_Tests PRIVATE
    __SYCL_ID_QUERIES_FIT_IN_INT__=1
  )
endif()
