/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <getopt.h>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>
#include <vtk_op.hpp>

#include "holoscan/holoscan.hpp"

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;

    const uint32_t width = 854;
    const uint32_t height = 480;
    const uint64_t source_block_size = width * height * 3 * 4;
    const uint64_t source_num_blocks = 2;

    auto source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer", from_config("replayer"), Arg("directory", datapath));

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("format_converter_" + source_),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = datapath + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath + "/engines";

    const uint64_t lstm_inferer_block_size = 107 * 60 * 7 * 4;
    const uint64_t lstm_inferer_num_blocks = 2 + 5 * 2;
    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, lstm_inferer_block_size, lstm_inferer_num_blocks),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const uint64_t tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4;
    const uint64_t tool_tracking_postprocessor_num_blocks = 2;
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator") =
            make_resource<BlockMemoryPool>("device_allocator",
                                           1,
                                           tool_tracking_postprocessor_block_size,
                                           tool_tracking_postprocessor_num_blocks),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    std::shared_ptr<ops::VtkOp> visualizer =
        make_operator<ops::VtkOp>("vtk",
                                  from_config("vtk_op"),
                                  Arg("width") = width,
                                  Arg("height") = height,
                                  Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "annotations"}});

    add_flow(source, format_converter, {{"output", "source_video"}});
    add_flow(format_converter, lstm_inferer);
    add_flow(source, visualizer, {{"output", "videostream"}});
  }

 private:
  std::string datapath = "data/endoscopy";
  std::string source_ = "replayer";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/endoscopy_tool_tracking_vtk.yaml";
    app->config(config_path);
  }

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
