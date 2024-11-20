
# Grab the date of the latest commit
execute_process(
  COMMAND git log -1 --format=%as # date in YYYY-MM-DD mode
  WORKING_DIRECTORY ${SYCL_ROOT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_DATE_TEMP
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "-" "" __LIBSYCL_TIMESTAMP ${GIT_COMMIT_DATE_TEMP})

# Legacy thing for backwards compatibility. Use of the current date is not
# reliable, because we can always make new build from older commits.
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
configure_file("${sycl_src_dir}/version.hpp.in" "${SYCL_INCLUDE_BUILD_DIR}/sycl/version.hpp")
