import { Features } from './features';

enum ModelId {
  CODE_T5_220M = 'Salesforce/codet5p-220m-py',
  DECICODER_1B_OPENVINO_INT8 = 'chgk13/decicoder-1b-openvino-int8',
  STABLECODE_COMPLETION_ALPHA_3B_4K_OPENVINO_INT8 = 'chgk13/stablecode-completion-alpha-3b-4k-openvino-int8',
  DEEPSEEK_CODER_1_3B = 'kumarijy/deepseek-code-1.3b_base_ov_int8',
}

export enum ModelName {
  CODE_T5_220M = 'code-t5',
  DECICODER_1B_OPENVINO_INT8 = 'decicoder-1b-openvino',
  STABLECODE_COMPLETION_ALPHA_3B_4K_OPENVINO_INT8 = 'stablecode-completion',
  DEEPSEEK_CODER_1_3B = 'deepseek-coder',
}

export const MODEL_NAME_TO_ID_MAP: Record<ModelName, ModelId> = {
  [ModelName.CODE_T5_220M]: ModelId.CODE_T5_220M,
  [ModelName.DECICODER_1B_OPENVINO_INT8]: ModelId.DECICODER_1B_OPENVINO_INT8,
  [ModelName.STABLECODE_COMPLETION_ALPHA_3B_4K_OPENVINO_INT8]: ModelId.STABLECODE_COMPLETION_ALPHA_3B_4K_OPENVINO_INT8,
  [ModelName.DEEPSEEK_CODER_1_3B]: ModelId.DEEPSEEK_CODER_1_3B,
};

export const MODEL_SUPPORTED_FEATURES: Record<ModelName, Features[]> = {
  [ModelName.CODE_T5_220M]: [Features.CODE_COMPLETION],
  [ModelName.DECICODER_1B_OPENVINO_INT8]: [Features.CODE_COMPLETION, Features.SUMMARIZATION],
  [ModelName.STABLECODE_COMPLETION_ALPHA_3B_4K_OPENVINO_INT8]: [Features.CODE_COMPLETION, Features.SUMMARIZATION],
  [ModelName.DEEPSEEK_CODER_1_3B]: [Features.CODE_COMPLETION, Features.SUMMARIZATION, Features.FIM],
};
