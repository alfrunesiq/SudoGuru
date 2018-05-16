cmake_minimum_required(VERSION 3.5.1)

project(Sudoguru)

# Compiler flags
if (MSVC)
  set(compiler_options /W4)
  add_definitions(/D_USE_MATH_DEFINES)
else ()
  set(compiler_options -Wall -Wextra -Wpedantic)
  #add_definitions(-DDEBUG)
endif (MSVC)

# Add libraries.
find_package(OpenCV 3.3.1 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/sudokusolver)
add_subdirectory(sudokusolver)

add_executable(${PROJECT_NAME}
  main.cpp
  sudoguru.cpp
  extractools.cpp
  homographyestimator.cpp
  )

target_link_libraries(${PROJECT_NAME} SudokuSolver ${OpenCV_LIBS})

#target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${sophus_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${compiler_options})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 11)

