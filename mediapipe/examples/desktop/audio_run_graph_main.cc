// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_audio";
constexpr char kOutputStream[] = "output_text";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_audio_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a mic.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.txt). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Check if audio file is provided.";
    const bool audio_file_path = !absl::GetFlag(FLAGS_input_audio_path).empty();
    return absl::OkStatus();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
