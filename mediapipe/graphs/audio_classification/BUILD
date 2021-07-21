# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "desktop_calculators",
    deps = [
        "//mediapipe/calculators/audio:audio_decoder_calculator",
        "//mediapipe/calculators/audio:basic_time_series_calculators",
        "//mediapipe/calculators/audio:mfcc_mel_calculators",
        "//mediapipe/calculators/core:add_header_calculator",
        "//mediapipe/calculators/audio:rational_factor_resample_calculator",
        "//mediapipe/calculators/audio:spectrogram_calculator",
        "//mediapipe/calculators/audio:stabilized_log_calculator",
        "//mediapipe/calculators/audio:time_series_framer_calculator",
        "//mediapipe/calculators/tensorflow:matrix_to_tensor_calculator",
        "//mediapipe/calculators/tensorflow:unpack_media_sequence_calculator",
        "//mediapipe/calculators/tensorflow:string_to_sequence_example_calculator",
        "//mediapipe/calculators/tensorflow:tensorflow_inference_calculator",
        "//mediapipe/calculators/tensorflow:tensor_to_vector_float_calculator",
        "//mediapipe/calculators/tensorflow:tensorflow_session_from_saved_model_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/util:top_k_scores_calculator",       
    ],
)

mediapipe_binary_graph(
    name = "audio_classification_desktop_live_binary_graph",
    graph = "audio_classification_desktop_live.pbtxt",
    output_name = "audio_classification_desktop_live.binarypb",
    deps = [":desktop_calculators"],
)
