# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Add testing
if(BUILD_TESTING)
  # To get the environment path
  find_package(holoscan 0.5 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  # Add test
  # Configure the yaml file to only play 10 frames
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/orsi_segmentation_ar.yaml" CONFIG_FILE)
  string(REGEX REPLACE "^source:[^\n]*" "^source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/orsi_segmentation_ar_testing.yaml" ${CONFIG_FILE})

  add_test(NAME orsi_segmentation_ar_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/orsi_segmentation_ar.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/orsi_segmentation_ar_testing.yaml
                   --data ${HOLOHUB_DATA_DIR}/orsi
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST orsi_segmentation_ar_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(orsi_segmentation_ar_python_test PROPERTIES
                PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
