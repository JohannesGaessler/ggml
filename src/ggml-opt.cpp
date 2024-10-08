#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdint>
#include <vector>

struct ggml_opt_new_context {
    ggml_backend_t backend;
    ggml_backend_buffer_t buf;
    struct ggml_context * ctx;
    bool ctx_owned;

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

struct ggml_opt_new_result {
    int64_t              nex      = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;
};

struct ggml_opt_new_params ggml_opt_new_default_params(
        ggml_backend_t       backend,
        struct ggml_tensor * inputs,
        struct ggml_tensor * logits) {
    return {
        /*backend    =*/ backend,
        /*ctx        =*/ nullptr,
        /*inputs     =*/ inputs,
        /*logits     =*/ logits,
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
    result->opt_period = params.opt_period;
    result->opt_i      = 0;

    if (params.ctx) {
        result->ctx = params.ctx;
        result->ctx_owned = false;
    } else {
        // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
        const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
        struct ggml_init_params ctx_params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(ctx_params);
        result->ctx_owned = true;
    }

    result->gf = ggml_new_graph_custom(result->ctx, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.

    ggml_set_input(result->inputs);

    ggml_set_output(result->logits);
    ggml_build_forward_expand(result->gf, result->logits);

    result->labels = ggml_dup_tensor(result->ctx, result->logits);
    ggml_set_input(result->labels);

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

        result->buf = ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);

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

    if (result->ctx_owned) {
        result->buf = ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);
        ggml_opt_new_reset(result, /*optimizer =*/ true);
    } else {
        result->buf = nullptr;
    }

    return result;
}

void ggml_opt_new_free(struct ggml_opt_new_context * opt_ctx) {
    if (opt_ctx->ctx_owned) {
        ggml_backend_buffer_free(opt_ctx->buf);
        ggml_free(opt_ctx->ctx);
    }
    delete opt_ctx;
}

void ggml_opt_new_reset(struct ggml_opt_new_context * opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

struct ggml_opt_new_result * ggml_opt_new_result_init() {
    return new ggml_opt_new_result;
}

void ggml_opt_new_result_free(struct ggml_opt_new_result * result) {
    delete result;
}

void ggml_opt_new_result_reset(struct ggml_opt_new_result * result) {
    result->nex = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

struct ggml_tensor * ggml_opt_new_inputs(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_new_logits(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->logits;
}

struct ggml_tensor * ggml_opt_new_labels(struct ggml_opt_new_context * opt_ctx) {
    return opt_ctx->labels;
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

static void ggml_opt_new_eval_graph(struct ggml_opt_new_context * opt_ctx, ggml_cgraph * graph, ggml_opt_new_result * result) {
    GGML_ASSERT(graph);
    ggml_backend_graph_compute(opt_ctx->backend, graph);

    if (!result) {
        return;
    }

    const int64_t nex = opt_ctx->logits->ne[1];
    GGML_ASSERT(result->nex == nex*int64_t(result->loss.size()) && "varying batch size not supported");
    result->nex += nex;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
    std::vector<int32_t> pred(nex);
    ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
    result->pred.insert(result->pred.end(), pred.begin(), pred.end());

    GGML_ASSERT(ggml_is_scalar(opt_ctx->acc_count));
    GGML_ASSERT(opt_ctx->acc_count->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->acc_count, &ncorrect, 0, ggml_nbytes(opt_ctx->acc_count));
    result->ncorrect += ncorrect;
}

void ggml_opt_new_forward(struct ggml_opt_new_context * opt_ctx, ggml_opt_new_result * result) {
    ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gf, result);
}

void ggml_opt_new_forward_backward(struct ggml_opt_new_context * opt_ctx, ggml_opt_new_result * result) {
    if (opt_ctx->opt_period == 1) {
        ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gf, result);
        return;
    }

    const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    if (opt_i_next == 0) {
        ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gb_opt, result);
        ggml_opt_new_reset(opt_ctx, /*optimizer =*/ false);
    } else {
        ggml_opt_new_eval_graph(opt_ctx, opt_ctx->gb_grad, result);
    }
    opt_ctx->opt_i = opt_i_next;
}

void ggml_opt_new_result_nex(struct ggml_opt_new_result * result, int64_t * nex) {
    *nex = result->nex;
}

void ggml_opt_new_result_loss(struct ggml_opt_new_result * result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size();

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        sum         += loss;
        sum_squared += loss*loss;
    }

    *loss = sum/nbatches;

    if (!unc) {
        return;
    }

    *unc = nbatches >= 2 ? sqrt((sum_squared/nbatches - (*loss)*(*loss)) / (nbatches - 1)) : NAN;
}

void ggml_opt_new_result_pred(struct ggml_opt_new_result * result, float * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_new_result_accuracy(struct ggml_opt_new_result * result, double * accuracy, double * unc) {
    *accuracy = double(result->ncorrect) / double(result->nex);

    if (!unc) {
        return;
    }

    *unc = sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->nex - 1));
}
