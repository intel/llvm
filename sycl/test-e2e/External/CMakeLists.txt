foreach(APP ${SYCL_EXTERNAL_TESTS})
  message("-- Adding external application: ${APP}")
  add_subdirectory(${APP})
endforeach()
