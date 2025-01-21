if (DOXYGEN_FOUND)
if (LLVM_ENABLE_DOXYGEN)
  set(abs_srcdir ${CMAKE_CURRENT_SOURCE_DIR})
  set(abs_builddir ${CMAKE_CURRENT_BINARY_DIR})

  if (HAVE_DOT)
    set(DOT ${LLVM_PATH_DOT})
  endif()

  # TODO add support for external search engine

  if (LLVM_ENABLE_DOXYGEN_QT_HELP)
    set(SYCL_DOXYGEN_QCH_FILENAME "org.llvm.sycl.qch" CACHE STRING
      "Filename of the Qt Compressed help file")
    set(SYCL_DOXYGEN_QHP_NAMESPACE "org.llvm.sycl" CACHE STRING
      "Namespace under which the intermediate Qt Help Project file lives")
    set(SYCL_DOXYGEN_QHP_CUST_FILTER_NAME "SYCL ${SYCL_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#custom-filters")
    set(SYCL_DOXYGEN_QHP_CUST_FILTER_ATTRS "Clang,${SYCL_VERSION}" CACHE STRING
      "See http://qt-project.org/doc/qt-4.8/qthelpproject.html#filter-attributes")
    set(sycl_doxygen_generate_qhp "YES")
    set(sycl_doxygen_qch_filename "${SYCL_DOXYGEN_QCH_FILENAME}")
    set(sycl_doxygen_qhp_namespace "${SYCL_DOXYGEN_QHP_NAMESPACE}")
    set(sycl_doxygen_qhelpgenerator_path "${LLVM_DOXYGEN_QHELPGENERATOR_PATH}")
    set(sycl_doxygen_qhp_cust_filter_name "${SYCL_DOXYGEN_QHP_CUST_FILTER_NAME}")
    set(sycl_doxygen_qhp_cust_filter_attrs "${SYCL_DOXYGEN_QHP_CUST_FILTER_ATTRS}")
  else()
    set(sycl_doxygen_generate_qhp "NO")
    set(sycl_doxygen_qch_filename "")
    set(sycl_doxygen_qhp_namespace "")
    set(sycl_doxygen_qhelpgenerator_path "")
    set(sycl_doxygen_qhp_cust_filter_name "")
    set(sycl_doxygen_qhp_cust_filter_attrs "")
  endif()

  option(LLVM_DOXYGEN_SVG
    "Use svg instead of png files for doxygen graphs." OFF)
  if (LLVM_DOXYGEN_SVG)
    set(DOT_IMAGE_FORMAT "svg")
  else()
    set(DOT_IMAGE_FORMAT "png")
  endif()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  set(abs_top_srcdir)
  set(abs_top_builddir)
  set(DOT)
  set(enable_searchengine)
  set(searchengine_url)
  set(enable_server_based_search)
  set(enable_external_search)
  set(extra_search_mappings)
  set(sycl_doxygen_generate_qhp)
  set(sycl_doxygen_qch_filename)
  set(sycl_doxygen_qhp_namespace)
  set(sycl_doxygen_qhelpgenerator_path)
  set(sycl_doxygen_qhp_cust_filter_name)
  set(sycl_doxygen_qhp_cust_filter_attrs)
  set(DOT_IMAGE_FORMAT)

  add_custom_target(doxygen-sycl
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating sycl doxygen documentation." VERBATIM)

  if (LLVM_BUILD_DOCS)
    add_dependencies(doxygen doxygen-clang)
  endif()

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
      DESTINATION docs/html)
  endif()

endif()
endif()

if (LLVM_ENABLE_SPHINX)
  include(AddSphinxTarget)
  if (SPHINX_FOUND)
    if (${SPHINX_OUTPUT_HTML})
      add_sphinx_target(html sycl)
    endif()
  endif()
endif()
