// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_fixture.hpp"
#include "openvino/runtime/infer_request.hpp"

const std::vector<int64_t> GPT2_SUN_PROMPT_TOKEN_IDS = {5195, 318, 262, 3825, 7872, 30};
const std::vector<int64_t> GPT2_LENNON_PROMPT_TOKEN_IDS = {8241, 318, 1757, 37470, 30};

std::vector<float> infer_logits_for_tokens_with_positions(ov::InferRequest& lm,
                                                          const std::vector<int64_t>& tokens,
                                                          int64_t position_ids_start_value) {
    auto input_ids_tensor = ov::Tensor(ov::element::Type_t::i64, {1, tokens.size()});
    std::copy(tokens.begin(), tokens.end(), input_ids_tensor.data<int64_t>());
    lm.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids_tensor.get_shape());
    std::iota(position_ids.data<int64_t>(),
              position_ids.data<int64_t>() + position_ids.get_size(),
              position_ids_start_value);

    CompiledModelTest::fill_unused_inputs(lm, input_ids_tensor.get_shape());
    lm.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    float* logits = lm.get_tensor("logits").data<float>() + (input_ids_tensor.get_size() - 1) * vocab_size;
    std::vector<float> logits_vector(vocab_size);
    std::copy(logits, logits + vocab_size, logits_vector.begin());
    return logits_vector;
}

std::vector<int64_t> generate_n_tokens_with_positions(ov::InferRequest& lm,
                                                      int64_t last_token,
                                                      size_t n_tokens,
                                                      int64_t position_ids_start_value) {
    size_t cnt = 0;
    std::vector<int64_t> out_token_ids;
    out_token_ids.push_back(last_token);

    while (cnt < n_tokens) {
        std::vector<float> logits_curr =
            infer_logits_for_tokens_with_positions(lm, {out_token_ids.back()}, cnt + position_ids_start_value);
        int64_t out_token = std::max_element(logits_curr.begin(), logits_curr.end()) - logits_curr.begin();
        out_token_ids.push_back(out_token);
        cnt++;
    }
    return out_token_ids;
}

inline int64_t get_token_from_logits(const std::vector<float>& logits) {
    return std::max_element(logits.cbegin(), logits.cend()) - logits.cbegin();
}

constexpr size_t NUM_TOKENS_TO_GENERATE = 64;

TEST_F(CompiledModelTest, ResetStateGPT2) {
    // collect reference response tokens
    ov::InferRequest lm = model.create_infer_request();
    std::vector<float> logits_sun_ref = infer_logits_for_tokens_with_positions(lm, GPT2_SUN_PROMPT_TOKEN_IDS, 0);
    std::vector<int64_t> out_token_ids_ref = generate_n_tokens_with_positions(lm,
                                                                              get_token_from_logits(logits_sun_ref),
                                                                              NUM_TOKENS_TO_GENERATE,
                                                                              GPT2_SUN_PROMPT_TOKEN_IDS.size());

    // call SetUp to reload the model from scratch, process unrelated prompt, then reset and request generation with the
    // first prompt again
    SetUp();

    ov::InferRequest lm_reset = model.create_infer_request();
    std::vector<float> logits_lennon_reset =
        infer_logits_for_tokens_with_positions(lm, GPT2_LENNON_PROMPT_TOKEN_IDS, 0);

    lm_reset.reset_state();

    std::vector<float> logits_sun_reset =
        infer_logits_for_tokens_with_positions(lm_reset,
                                               GPT2_SUN_PROMPT_TOKEN_IDS,
                                               0);  // GPT2_LENNON_PROMPT_TOKEN_IDS.size());

    std::vector<int64_t> out_token_ids_reset = generate_n_tokens_with_positions(lm_reset,
                                                                                get_token_from_logits(logits_sun_reset),
                                                                                NUM_TOKENS_TO_GENERATE,
                                                                                GPT2_SUN_PROMPT_TOKEN_IDS.size());
    ASSERT_EQ(out_token_ids_reset, out_token_ids_ref);

    // sanity check - without reset the response after the second prompt is different
    SetUp();

    ov::InferRequest lm_bad = model.create_infer_request();
    std::vector<float> logits_lennon_bad = infer_logits_for_tokens_with_positions(lm, GPT2_LENNON_PROMPT_TOKEN_IDS, 0);

    // no reset_state on purpose

    std::vector<float> logits_sun_bad =
        infer_logits_for_tokens_with_positions(lm_reset,
                                               GPT2_SUN_PROMPT_TOKEN_IDS,
                                               0);  // GPT2_LENNON_PROMPT_TOKEN_IDS.size());

    std::vector<int64_t> out_token_ids_bad = generate_n_tokens_with_positions(lm_reset,
                                                                              get_token_from_logits(logits_sun_reset),
                                                                              NUM_TOKENS_TO_GENERATE,
                                                                              GPT2_SUN_PROMPT_TOKEN_IDS.size());
}
