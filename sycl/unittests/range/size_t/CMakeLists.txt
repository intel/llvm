add_sycl_unittest(IdQueriesRangeValidationSizeT SHARED
  IdQueriesRangeValidation.cpp
)
# No compile definitions needed - size_t mode is the default when neither macro is defined
