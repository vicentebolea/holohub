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

#ifndef VTK_OPERATOR__HPP
#define VTK_OPERATOR__HPP

#include <memory>
#include <string>
#include <vector>

#include <vtkImageActor.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace holoscan::ops {

class VtkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VtkOp)

  VtkOp() = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override;

 private:
  Parameter<holoscan::IOSpec*> in_annotations;
  Parameter<holoscan::IOSpec*> in_videostream;

  Parameter<uint32_t> width;
  Parameter<uint32_t> height;
  Parameter<std::vector<std::string>> labels;
  CudaStreamHandler cuda_stream_handler_;
  vtkNew<vtkImageActor> imageActor;
  vtkNew<vtkRenderer> backgroundRenderer;
  vtkNew<vtkRenderer> foregroundRenderer;
  vtkNew<vtkRenderWindow> rendererWindow;
  cudaEvent_t cuda_event_ = nullptr;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP */
