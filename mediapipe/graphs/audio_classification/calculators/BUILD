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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_classifier",
    srcs = ["audio_classifier_lib.cc"],
    hdrs = ["audio_classifier_lib.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/audio:audio_classifier",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/audio/core:audio_buffer",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/audio/proto:classifications_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/core:category",
        "@org_tensorflow_lite_support//tensorflow_lite_support/examples/task/audio/desktop/wav:wav_io",
    ] + select({
        "@org_tensorflow_lite_support//tensorflow_lite_support/examples/task:darwinn_portable": [
            "@org_tensorflow_lite_support//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin",
        ],
        "//conditions:default": [
        ],
    }),
)