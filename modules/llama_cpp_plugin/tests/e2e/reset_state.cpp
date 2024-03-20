// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_fixture.hpp"

const std::vector<int64_t> GPT2_SUN_PROMPT_TOKEN_IDS = {5195, 318, 262, 3825, 7872, 30};
const std::vector<int64_t> GPT2_LENNON_PROMPT_TOKEN_IDS = {8241, 318, 1757, 37470, 30};

TEST_F(CompiledModelTest, ResetStateGPT2) {}
