/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <iostream>
#include <limits>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "audio_classifier.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "tensorflow_lite_support/examples/task/audio/desktop/audio_classifier_lib.h"

namespace mediapipe {

class TfliteTaskAudioClassifierCalculator : public CalculatorBase {
  public:
    TfliteTaskAudioClassifierCalculator() = default;

    static absl::Status GetContract(CalculatorContract* cc);
    absl::Status Process(CalculatorContext* cc) override;
};

REGISTER_CALCULATOR(TfliteTaskAudioClassifierCalculator);

absl::Status TfliteTaskAudioClassifierCalculator::GetContract(CalculatorContract* cc) {
    RET_CHECK(!cc->InputSidePackets().GetTags().empty());
    RET_CHECK(!cc->Outputs().GetTags().empty());

    cc->InputSidePackets().Tag("MODEL_PATH").Set<std::String>();
    cc->InputSidePackets().Tag("DATA_PATH").Set<std::String>();
    cc->Outputs().Tag("CLASS").Set<std::String>();

    return absl::OkStatus();
}

absl::Status TfliteTaskAudioClassifierCalculator::Process(CalculatorContext* cc) {
    const std::string& input_file_path =
        cc->InputSidePackets().Tag("DATA_PATH").Get<std::string>();
    const std::string& yamnet_model_path =
        cc->InputSidePackets().Tag("MODEL_PATH").Get<std::String>();

    const int min_score_thres = 0.5;

    //Start Classification
    auto result = tflite::task::audio::Classify(
        yamnet_model_path, input_file_path, false);   //False for Coral Edge TPU not connected

    if (result.ok()) {
        const tflite::task::audio::ClassificationResult& result_ = result.value();
        const auto& head = result_.classifications(0);
        const int score = head.classes(0).score();
        const std::string classification = head.classes(0).class_name();

        if (score >= min_score_thres) {
            cc->Outputs().Tag("CLASS").Add(classification, cc->InputTimestamp());
        }

    } else {
        std::cerr << "Classification failed: " << result.status().message() << "\n";
        return 1;
    }
}

}   // namespace mediapipe