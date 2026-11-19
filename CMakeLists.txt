cmake_minimum_required(VERSION 3.20)
project(ps3recomp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PS3RECOMP_AVX2 "Compile generated code and runtime for AVX2 (fast FMA)" ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /bigobj)
  if(PS3RECOMP_AVX2)
    add_compile_options(/arch:AVX2)
  endif()
else()
  add_compile_options(-Wall -Wextra)
  if(PS3RECOMP_AVX2)
    add_compile_options(-mavx2 -mfma)
  endif()
endif()

find_package(Threads REQUIRED)

# ---- Libraries -------------------------------------------------------------
add_library(ps3_elf STATIC src/elf/elf64.cpp src/elf/ps3_exec.cpp src/elf/nid.cpp)
target_include_directories(ps3_elf PUBLIC src/elf)

add_library(ps3_ppu STATIC src/ppu/ppu_decode.cpp src/ppu/ppu_analysis.cpp src/ppu/ppu_lift.cpp)
target_include_directories(ps3_ppu PUBLIC src/ppu)
target_link_libraries(ps3_ppu PUBLIC ps3_elf)

add_library(ps3_spu STATIC src/spu/spu_interp.cpp src/spu/spu_jit.cpp)
target_include_directories(ps3_spu PUBLIC src/spu)

# Vulkan headers: from an installed SDK if there is one, otherwise the official
# Khronos headers pinned to a release tag. The Vulkan loader itself (vulkan-1.dll)
# comes with the GPU driver and is loaded at runtime.
add_library(ps3_vulkan_headers INTERFACE)
if(DEFINED ENV{VULKAN_SDK})
  target_include_directories(ps3_vulkan_headers INTERFACE "$ENV{VULKAN_SDK}/Include")
else()
  include(FetchContent)
  FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.3.290
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(vulkan_headers)
  target_link_libraries(ps3_vulkan_headers INTERFACE Vulkan::Headers)
endif()

add_library(ps3_gfx STATIC src/gfx/vk_presenter.cpp)
target_include_directories(ps3_gfx PUBLIC src/gfx)
target_link_libraries(ps3_gfx PRIVATE ps3_vulkan_headers)
if(WIN32)
  target_link_libraries(ps3_gfx PUBLIC gdiplus)
endif()

add_library(ps3_runtime STATIC
  src/runtime/guest_memory.cpp src/runtime/mod_loader.cpp
  src/runtime/lv2.cpp src/runtime/hle.cpp src/runtime/hle_modules.cpp src/runtime/boot.cpp
  src/runtime/vfs.cpp src/runtime/hle_fs.cpp src/runtime/input.cpp src/runtime/hle_pad.cpp
  src/runtime/rsx.cpp src/runtime/hle_gcm.cpp src/runtime/rsx_program.cpp src/runtime/rsx_draw.cpp
  src/runtime/spirv_gen.cpp src/runtime/vk_renderer.cpp
  src/runtime/audio.cpp src/runtime/audio_wasapi.cpp src/runtime/hle_audio.cpp
  src/runtime/lv2_spu.cpp src/runtime/hle_misc.cpp src/runtime/hle_spurs.cpp
  src/runtime/hle_savedata.cpp src/runtime/profiler.cpp src/runtime/hid_pad.cpp)
if(WIN32)
  target_link_libraries(ps3_runtime PUBLIC ole32)
endif()
target_include_directories(ps3_runtime PUBLIC src/runtime)
target_link_libraries(ps3_runtime PUBLIC ps3_elf ps3_gfx ps3_spu Threads::Threads PRIVATE ps3_vulkan_headers)

# ---- Builder (GUI + console twin): pick game folder + your decrypted ELF, Build, Play ----
if(WIN32)
  add_library(ps3_builder STATIC src/gui/builder_core.cpp)
  target_include_directories(ps3_builder PUBLIC src/gui)
  target_link_libraries(ps3_builder PUBLIC ps3_runtime)
  add_executable(ps3recomp_builder WIN32 src/gui/builder_gui.cpp)
  set_target_properties(ps3recomp_builder PROPERTIES OUTPUT_NAME "CellForge")
  target_link_libraries(ps3recomp_builder PRIVATE ps3_builder ole32 shell32 uuid gdiplus dwmapi)
  add_executable(ps3build src/gui/builder_cli.cpp)
  target_link_libraries(ps3build PRIVATE ps3_builder)
  add_executable(ps3_builder_tests tests/builder_tests.cpp)
  target_link_libraries(ps3_builder_tests PRIVATE ps3_builder)
endif()

# Prints live controller input (XInput pads, DualSense / DualShock 4 over HID).
add_executable(ps3_pad_test tests/pad_test.cpp)
target_link_libraries(ps3_pad_test PRIVATE ps3_runtime)

# Plays a quiet test tone through the default audio device.
add_executable(ps3_audio_demo tests/audio_demo.cpp)
target_link_libraries(ps3_audio_demo PRIVATE ps3_runtime)

# Opens a window on the GPU and shows an animated test pattern for a few seconds.
add_executable(ps3_gfx_demo tests/gfx_demo.cpp)
target_link_libraries(ps3_gfx_demo PRIVATE ps3_gfx)
if(NOT WIN32)
  target_link_libraries(ps3_runtime PUBLIC ${CMAKE_DL_LIBS})
endif()

# ---- A recompiled game ------------------------------------------------------
# cmake -DPS3RECOMP_GAME_DIR=<dir with ps3recomp --out-dir output> builds `ps3game`.
set(PS3RECOMP_GAME_DIR "" CACHE PATH "Directory containing generated ppu_recomp_*.cpp")
set(PS3RECOMP_GAME_NAME "ps3game" CACHE STRING "Output executable name for the built game")
set(PS3RECOMP_GAME_ICON "" CACHE FILEPATH "Optional .ico file embedded into the game executable")
if(PS3RECOMP_GAME_DIR)
  file(GLOB GAME_SOURCES CONFIGURE_DEPENDS ${PS3RECOMP_GAME_DIR}/*.cpp)
  # The generated code needs only ppu_runtime.h, so it is compiled as its own
  # object library with fixed flags: changes to the runtime's targets, include
  # paths or dependencies never force the (very large) game code to recompile.
  add_library(ps3game_code OBJECT ${GAME_SOURCES})
  target_include_directories(ps3game_code PRIVATE src/runtime)
  if(MSVC)
    # Compile files in parallel, and skip /W4 noise (unused labels etc.).
    target_compile_options(ps3game_code PRIVATE /MP /W1 /wd4102)
  endif()
  set(_game_rc "")
  if(WIN32 AND PS3RECOMP_GAME_ICON AND EXISTS "${PS3RECOMP_GAME_ICON}")
    file(TO_CMAKE_PATH "${PS3RECOMP_GAME_ICON}" _ico_cmake)
    set(_game_rc "${CMAKE_CURRENT_BINARY_DIR}/game_icon.rc")
    file(WRITE "${_game_rc}" "101 ICON \"${_ico_cmake}\"\n")
  endif()
  add_executable(ps3game src/host/main.cpp ${_game_rc} $<TARGET_OBJECTS:ps3game_code>)
  set_target_properties(ps3game PROPERTIES OUTPUT_NAME "${PS3RECOMP_GAME_NAME}")
  target_link_libraries(ps3game PRIVATE ps3_runtime)
  if(MSVC)
    target_link_options(ps3game PRIVATE /MAP)  # ps3game.map: host crash offsets -> fn_XXXXXXXX
  endif()
endif()

# ---- Recompiler CLI --------------------------------------------------------
add_executable(ps3recomp src/tools/recomp_main.cpp)
target_link_libraries(ps3recomp PRIVATE ps3_ppu)

# ---- End-to-end test -------------------------------------------------------
# 1. make_test_elf hand-assembles a tiny PPC64 big-endian ELF (+ test_layout.h).
# 2. ps3recomp lifts it to C++.
# 3. ps3_test_run compiles that C++ with the runtime and checks the results,
#    including hooks and a mod DLL (example_mod).
add_executable(make_test_elf tests/make_test_elf.cpp)
target_link_libraries(make_test_elf PRIVATE ps3_elf)

set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_gen)
set(TEST_MODS_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_mods)
add_custom_command(
  OUTPUT ${TEST_DIR}/test.elf ${TEST_DIR}/test_funcs.txt ${TEST_DIR}/test_layout.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DIR}
  COMMAND make_test_elf ${TEST_DIR}
  DEPENDS make_test_elf)

add_custom_command(
  OUTPUT ${TEST_DIR}/test_gen.cpp
  COMMAND ps3recomp ${TEST_DIR}/test.elf -o ${TEST_DIR}/test_gen.cpp
          --func-file ${TEST_DIR}/test_funcs.txt
  DEPENDS ps3recomp ${TEST_DIR}/test.elf ${TEST_DIR}/test_funcs.txt)

add_executable(ps3_test_run tests/test_run.cpp ${TEST_DIR}/test_gen.cpp ${TEST_DIR}/test_layout.h)
target_include_directories(ps3_test_run PRIVATE ${TEST_DIR} src/ppu)
target_link_libraries(ps3_test_run PRIVATE ps3_runtime ps3_elf)

add_library(example_mod SHARED tests/example_mod.cpp ${TEST_DIR}/test_layout.h)
target_include_directories(example_mod PRIVATE ${TEST_DIR} src/runtime)
set_target_properties(example_mod PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<1:${TEST_MODS_DIR}>
  LIBRARY_OUTPUT_DIRECTORY $<1:${TEST_MODS_DIR}>
  PREFIX "")
add_dependencies(ps3_test_run example_mod)

add_executable(ps3_boot_test tests/boot_test.cpp ${TEST_DIR}/test_gen.cpp ${TEST_DIR}/test_layout.h)
target_include_directories(ps3_boot_test PRIVATE ${TEST_DIR})
target_link_libraries(ps3_boot_test PRIVATE ps3_runtime)

add_executable(ps3_unit_tests tests/unit_tests.cpp)
target_link_libraries(ps3_unit_tests PRIVATE ps3_ppu ps3_runtime)

enable_testing()
add_test(NAME unit_tests COMMAND ps3_unit_tests)
add_executable(ps3_spu_tests tests/spu_tests.cpp)
target_link_libraries(ps3_spu_tests PRIVATE ps3_spu)
add_test(NAME spu COMMAND ps3_spu_tests)
add_test(NAME boot COMMAND ps3_boot_test ${TEST_DIR}/test.elf)
if(WIN32)
  add_test(NAME builder COMMAND ps3_builder_tests ${TEST_DIR}/test.elf)
endif()
add_test(NAME recomp_roundtrip
  COMMAND ps3_test_run ${TEST_DIR}/test.elf ${TEST_MODS_DIR})
