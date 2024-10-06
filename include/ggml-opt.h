#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_opt_new_params {
        ggml_backend_t backend;
        struct ggml_tensor * inputs;
        struct ggml_tensor * logits;
        struct ggml_tensor * labels;

        bool forward_only;
        int32_t opt_period;

        // AdamW optimizer parameters
        struct {
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float wd;    // weight decay for AdamW, use 0.0f to disable
        } adamw;
    };

    struct ggml_opt_new_context;

    GGML_API ggml_opt_new_params ggml_opt_new_default_params(
            ggml_backend_t       backend,
            struct ggml_tensor * inputs,
            struct ggml_tensor * logits,
            struct ggml_tensor * labels);

    GGML_API struct ggml_opt_new_context * ggml_opt_new_init(struct ggml_opt_new_params params);

    GGML_API void ggml_opt_new_free(struct ggml_opt_new_context * opt_ctx);

    GGML_API struct ggml_tensor * ggml_opt_new_inputs(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_logits(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_labels(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_loss(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_pred(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_acc_count(struct ggml_opt_new_context * opt_ctx);

    GGML_API void ggml_opt_new_forward(struct ggml_opt_new_context * opt_ctx);

    GGML_API void ggml_opt_new_forward_backward(struct ggml_opt_new_context * opt_ctx);

#ifdef  __cplusplus
}
#endif
