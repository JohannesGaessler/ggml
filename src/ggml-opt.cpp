#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

struct ggml_opt_new_context {
    ggml_backend_t backend;
    struct ggml_context * ctx;

    struct ggml_tensor * inputs;
    struct ggml_tensor * logits;
    struct ggml_tensor * labels;

    struct ggml_tensor * loss;
    struct ggml_tensor * pred;
    struct ggml_tensor * acc_count;

    struct ggml_cgraph * gf;
    struct ggml_cgraph * gb_grad;
    struct ggml_cgraph * gb_opt;

    int32_t opt_period;
    int32_t opt_i;
};

struct ggml_opt_new_params ggml_opt_new_default_params(
        ggml_backend_t       backend,
        struct ggml_tensor * inputs,
        struct ggml_tensor * logits,
        struct ggml_tensor * labels) {
    return {
        /*backend    =*/ backend,
        /*inputs     =*/ inputs,
        /*logits     =*/ logits,
        /*labels     =*/ labels,
        /*forward_only =*/ false,
        /*opt_period =*/ 1,
        /*adamw      =*/ {
            /*alpha      =*/ 0.001f,
            /*beta1      =*/ 0.9f,
            /*beta2      =*/ 0.999f,
            /*eps        =*/ 1e-8f,
            /*wd         =*/ 0.0f,
        },
    };
}

struct ggml_opt_new_context * ggml_opt_new_init(struct ggml_opt_new_params params) {
    struct ggml_opt_new_context * result = new struct ggml_opt_new_context;
    result->backend    = params.backend;
    result->inputs     = params.inputs;
    result->logits     = params.logits;
    result->labels     = params.labels;
    result->opt_period = params.opt_period;
    result->opt_i      = 0;

    {
        // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
        const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
        struct ggml_init_params ctx_params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(ctx_params);
    }

    result->gf = ggml_new_graph_custom(result->ctx, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.

    ggml_set_input(result->inputs);
    ggml_set_input(result->labels);

    ggml_set_output(result->logits);
    ggml_build_forward_expand(result->gf, result->logits);

    result->loss = ggml_cross_entropy_loss(result->ctx, result->logits, result->labels);
    ggml_set_output(result->loss);
    ggml_set_loss(result->loss);
    ggml_build_forward_expand(result->gf, result->loss);

    result->pred = ggml_argmax(result->ctx, result->logits);
    ggml_set_output(result->pred);
    ggml_build_forward_expand(result->gf, result->pred);

    result->acc_count = ggml_count_equal(result->ctx, result->pred, ggml_argmax(result->ctx, result->labels));
    ggml_set_output(result->acc_count);
    ggml_build_forward_expand(result->gf, result->acc_count);

    if (params.forward_only) {
        result->gb_grad = nullptr;
        result->gb_opt  = nullptr;

        ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);

        return result;
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    result->gb_grad = ggml_graph_dup(result->ctx, result->gf);
    ggml_build_backward_expand(result->ctx, result->gf, result->gb_grad, params.opt_period > 1);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    result->gb_opt = ggml_graph_dup(result->ctx, result->gb_grad);

    for (int i = result->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = result->gf->nodes[i];

        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            struct ggml_tensor * opt_step = ggml_opt_step_adamw(
                result->ctx, node, node->grad, params.adamw.alpha, params.adamw.beta1, params.adamw.beta2, params.adamw.eps, params.adamw.wd);
            ggml_build_forward_expand(result->gb_opt, opt_step);
        }
    }

    ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);

    ggml_graph_reset(result->gb_opt); // Set gradients to zero, reset optimizer.

    return result;
}

void ggml_opt_new_free(struct ggml_opt_new_context * opt_ctx) {
    ggml_free(opt_ctx->ctx);
    delete opt_ctx;
}

struct ggml_tensor * ggml_opt_new_inputs(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_new_logits(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->logits;
}

struct ggml_tensor * ggml_opt_new_loss(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_new_pred(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_new_acc_count(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->acc_count;
}

void ggml_opt_new_forward(struct ggml_opt_new_context * opt_ctx) {
    ggml_backend_graph_compute(opt_ctx->backend, opt_ctx->gf);
}

void ggml_opt_new_forward_backward(struct ggml_opt_new_context * opt_ctx) {
    GGML_ASSERT(opt_ctx->gb_grad);
    GGML_ASSERT(opt_ctx->gb_opt);
    if (opt_ctx->opt_period == 1) {
        ggml_backend_graph_compute(opt_ctx->backend, opt_ctx->gb_opt);
        return;
    }

    const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    if (opt_i_next == 0) {
        ggml_backend_graph_compute(opt_ctx->backend, opt_ctx->gb_opt);
        ggml_graph_reset(opt_ctx->gb_grad);
    } else {
        ggml_backend_graph_compute(opt_ctx->backend, opt_ctx->gb_grad);
    }
    opt_ctx->opt_i = opt_i_next;
}
