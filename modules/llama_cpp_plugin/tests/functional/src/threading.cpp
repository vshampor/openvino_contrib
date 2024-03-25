// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <thread>

#include "benchmarking.hpp"
#include "llm_inference.hpp"

const std::string MODEL_FILE = ov::test::utils::getCurrentWorkingDir() + SEP + TEST_FILES_DIR + SEP + "gpt2.gguf";

TEST(LlamaCppThreadingTest, NumThreadSettingDoesntFail) {
    constexpr size_t NUM_THREADS_TO_SET = 2;
    ov::Core core;

    auto model =
        core.compile_model(MODEL_FILE, "LLAMA_CPP", ov::AnyMap{{ov::inference_num_threads.name(), NUM_THREADS_TO_SET}});
    auto lm = model.create_infer_request();

    std::vector<int64_t> mock_input_ids{1337, NUM_THREADS_TO_SET * 10};
    infer_logits_for_tokens_with_positions(lm, mock_input_ids, 0);
}

double measure_inference_speed_for_thread_count(size_t num_threads) {
    ov::Core core;
    auto model =
        core.compile_model(MODEL_FILE, "LLAMA_CPP", ov::AnyMap{{ov::inference_num_threads.name(), num_threads}});
    auto lm = model.create_infer_request();

    constexpr size_t NUM_INFER_REQUESTS = 256;

    auto infer_one_token_fn = [&lm](void) -> void {
        infer_logits_for_tokens_with_positions(lm, {1337}, 0);
    };
    return measure_iterations_per_second(infer_one_token_fn, NUM_INFER_REQUESTS);
}

TEST(LlamaCppThreadingTest, ThreadedExecutionIsFaster) {
    double single_threaded_iters_per_second = measure_inference_speed_for_thread_count(1);

    size_t optimal_num_threads = std::thread::hardware_concurrency();
    ASSERT_GT(optimal_num_threads, 1);

    double multi_threaded_iters_per_second = measure_inference_speed_for_thread_count(optimal_num_threads);
    std::cout << "threaded " << multi_threaded_iters_per_second << ", non-threaded "
              << single_threaded_iters_per_second;
    ASSERT_GE(multi_threaded_iters_per_second, single_threaded_iters_per_second);
}
