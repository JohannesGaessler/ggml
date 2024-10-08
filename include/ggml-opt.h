#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_opt_new_params {
        ggml_backend_t backend;
        ggml_context * ctx;

        struct ggml_tensor * inputs;
        struct ggml_tensor * logits;

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

    struct ggml_opt_new_result;

    GGML_API ggml_opt_new_params ggml_opt_new_default_params(
            ggml_backend_t       backend,
            struct ggml_tensor * inputs,
            struct ggml_tensor * logits);

    GGML_API struct ggml_opt_new_context * ggml_opt_new_init(struct ggml_opt_new_params params);
    GGML_API void ggml_opt_new_free(struct ggml_opt_new_context * opt_ctx);
    GGML_API void ggml_opt_new_reset(struct ggml_opt_new_context * opt_ctx, bool optimizer);

    GGML_API struct ggml_opt_new_result * ggml_opt_new_result_init();
    GGML_API void ggml_opt_new_result_free(struct ggml_opt_new_result * opt_ctx);
    GGML_API void ggml_opt_new_result_reset(struct ggml_opt_new_result * opt_ctx);

    GGML_API struct ggml_tensor * ggml_opt_new_inputs(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_logits(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_labels(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_loss(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_pred(struct ggml_opt_new_context * opt_ctx);
    GGML_API struct ggml_tensor * ggml_opt_new_acc_count(struct ggml_opt_new_context * opt_ctx);

    GGML_API void ggml_opt_new_forward(struct ggml_opt_new_context * opt_ctx, struct ggml_opt_new_result * result);

    GGML_API void ggml_opt_new_forward_backward(struct ggml_opt_new_context * opt_ctx, struct ggml_opt_new_result * result);

    GGML_API void ggml_opt_new_result_nex(     struct ggml_opt_new_result * result, int64_t * nex);
    GGML_API void ggml_opt_new_result_loss(    struct ggml_opt_new_result * result, double  * loss,     double * unc);
    GGML_API void ggml_opt_new_result_pred(    struct ggml_opt_new_result * result, float   * pred);
    GGML_API void ggml_opt_new_result_accuracy(struct ggml_opt_new_result * result, double  * accuracy, double * unc);

#ifdef  __cplusplus
}
#endif
