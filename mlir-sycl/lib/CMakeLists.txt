# Enable errors for any global constructors.
add_flag_if_supported("-Werror=global-constructors" WERROR_GLOBAL_CONSTRUCTOR)

add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
