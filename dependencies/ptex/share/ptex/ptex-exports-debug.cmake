#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Ptex::Ptex" for configuration "Debug"
set_property(TARGET Ptex::Ptex APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Ptex::Ptex PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/debug/lib/Ptex.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/bin/Ptex.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS Ptex::Ptex )
list(APPEND _IMPORT_CHECK_FILES_FOR_Ptex::Ptex "${_IMPORT_PREFIX}/debug/lib/Ptex.lib" "${_IMPORT_PREFIX}/debug/bin/Ptex.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
