cmake_minimum_required(VERSION 3.20)
project(CellForgeModSDK LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only interface target for all CellForge mods
add_library(cellforge_mod_sdk INTERFACE)
target_include_directories(cellforge_mod_sdk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Helper function to declare a CellForge mod DLL:
#   cellforge_add_mod(my_mod src/my_mod.cpp)
function(cellforge_add_mod TARGET_NAME)
  add_library(${TARGET_NAME} SHARED ${ARGN})
  target_link_libraries(${TARGET_NAME} PRIVATE cellforge_mod_sdk)
  set_target_properties(${TARGET_NAME} PROPERTIES
    PREFIX ""
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )
  if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W4 /EHsc)
  endif()
endfunction()

option(CELLFORGE_BUILD_SDK_EXAMPLES "Build example mods included with the SDK" ON)
if(CELLFORGE_BUILD_SDK_EXAMPLES)
  cellforge_add_mod(hello_mod examples/hello_mod/hello_mod.cpp)
  cellforge_add_mod(memory_patch_mod examples/memory_patch_mod/memory_patch_mod.cpp)
endif()
