# Finds or fetches emhash.
if(TARGET emhash::emhash)
  return()
endif()
find_package(emhash QUIET)
if(NOT emhash_FOUND)
  set(EMHASH_REPO https://github.com/ktprime/emhash)
  message(STATUS "Will fetch emhash from ${EMHASH_REPO}")
  include(FetchContent)
  FetchContent_Declare(emhash
    GIT_REPOSITORY    ${EMHASH_REPO}
    GIT_TAG           5e131ba09a5290823fe71099d9c35eb5df5345b6
    SOURCE_SUBDIR  emhash
  )
  FetchContent_MakeAvailable(emhash)

  # The official cmake install target uses the 'emhash' namespace,
  # so emulate that here so client code can use a single target name for both cases.
  add_library(emhash INTERFACE)
  target_include_directories(emhash SYSTEM INTERFACE ${emhash_SOURCE_DIR})
  add_library(emhash::emhash ALIAS emhash)
endif()
