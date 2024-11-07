#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <inttypes.h>
#include <map>
#include <random>
#include <vector>

struct ggml_opt_dataset {
    struct ggml_context   * ctx;
    ggml_backend_t          backend;
    ggml_backend_buffer_t   buf;
    struct ggml_tensor    * data;
    struct ggml_tensor    * labels;

    int64_t ndata;
    int64_t ndata_shard;
    size_t  nbs_data;
    size_t  nbs_labels;

    std::vector<int64_t> permutation;
};

struct ggml_opt_context {
    ggml_backend_sched_t    backend_sched;
    ggml_cgraph           * allocated_graph;
    ggml_cgraph           * allocated_graph_copy;
    struct ggml_context   * ctx_static;
    struct ggml_context   * ctx_compute;
    ggml_backend_buffer_t   buf_static;
    std::mt19937            rng;

    struct ggml_tensor * inputs;
    struct ggml_tensor * outputs;
    struct ggml_tensor * labels;

    struct ggml_tensor * loss;
    struct ggml_tensor * pred;
    struct ggml_tensor * ncorrect;

    struct ggml_cgraph * gf;
    struct ggml_cgraph * gb_grad;
    struct ggml_cgraph * gb_opt;

    bool    forward_only;
    int32_t opt_period;
    int32_t opt_i;
    bool    loss_per_datapoint;
};

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    bool loss_per_datapoint = false;
    int64_t opt_period = -1;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(int64_t ne_datapoint, int64_t ne_label, int64_t ndata, int64_t ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, GGML_TYPE_F32, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, GGML_TYPE_F32, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->backend = ggml_backend_cpu_init();
    GGML_ASSERT(result->backend);
    result->buf = ggml_backend_alloc_ctx_tensors(result->ctx, result->backend);

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_backend_free(dataset->backend);
    ggml_free(dataset->ctx);
    delete dataset;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_default_optimizer_params(){
    return {
        /*adamw =*/ {
            /*alpha =*/ 0.001f,
            /*beta1 =*/ 0.9f,
            /*beta2 =*/ 0.999f,
            /*eps   =*/ 1e-8f,
            /*wd    =*/ 0.0f,
        },
    };
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t backend_sched,
        struct ggml_context * ctx_compute,
        struct ggml_tensor * inputs,
        struct ggml_tensor * outputs,
        enum ggml_opt_loss_type loss_type) {
    return {
        /*backend_sched    =*/ backend_sched,
        /*ctx_compute      =*/ ctx_compute,
        /*inputs           =*/ inputs,
        /*logits           =*/ outputs,
        /*loss_type        =*/ loss_type,
        /*forward_only     =*/ false,
        /*opt_period       =*/ 1,
        /*optimizer_params =*/ ggml_opt_default_optimizer_params(),
    };
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_cgraph * graph) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_cgraph * new_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true);

    for (int i = 0; i < graph->n_leafs; i++) {
        ggml_build_forward_expand(new_graph, map_tensor(tensor_map, ctx, graph->leafs[i]));
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_build_forward_expand(new_graph, map_tensor(tensor_map, ctx, graph->nodes[i]));
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor * node = ggml_graph_node(new_graph, i);
        node->grad = map_tensor(tensor_map, ctx, node->grad);
    }

    return new_graph;
}

static void ggml_opt_alloc_graph(ggml_opt_context_t opt_ctx, ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    if (opt_ctx->allocated_graph == graph) {
       return;
    }

    opt_ctx->allocated_graph_copy = dup_graph(graph);

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph
    // for (struct ggml_tensor * t = ggml_get_first_tensor(opt_ctx->ctx_compute); t != nullptr; t = ggml_get_next_tensor(opt_ctx->ctx_compute, t)) {
    //     ggml_backend_tensor_reset(t); // clear dangling pointers from tensors, making them reusable for allocation
    // }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched   = params.backend_sched;
    result->allocated_graph = nullptr;
    result->allocated_graph_copy = nullptr;
    result->ctx_compute     = params.ctx_compute;
    result->inputs          = params.inputs;
    result->outputs         = params.outputs;
    result->forward_only    = params.forward_only;
    result->opt_period      = params.opt_period;
    result->opt_i           = 0;

    GGML_ASSERT(result->inputs->data && "the inputs must be allocated statically");
    GGML_ASSERT(result->opt_period >= 1);

    ggml_set_input(result->inputs);
    ggml_set_output(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    int n_param = 0;
    for (int i = 0; i < result->gf->n_nodes; ++i) {
        if (result->gf->nodes[i]->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
    }

    {
        // The static context is used for:
        //   - gradients (1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels
        //   - loss + its gradient (up to 5 tensors)
        //   - pred
        //   - ncorrect (2 tensors).
        const size_t tensors_per_param = result->forward_only ? 0 : (result->opt_period > 1 ? 3 : 2);
        const size_t size_meta = (tensors_per_param*n_param + 9) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx_static = ggml_init(params);
    }


    switch (params.loss_type) {
        case GGML_OPT_LOSS_TYPE_MEAN: {
            result->labels = nullptr;
            result->loss = ggml_sum(result->ctx_static, result->outputs);
            ggml_set_name(result->loss, "loss_sum");
            const float scale = 1.0f / (result->opt_period * ggml_nelements(result->outputs));
            result->loss = ggml_scale(result->ctx_static, result->loss, scale);
            ggml_set_name(result->loss, "loss_mean");
            result->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_SUM: {
            result->labels = nullptr;
            result->loss = ggml_sum(result->ctx_static, result->outputs);
            ggml_set_name(result->loss, "loss_sum");
            result->loss_per_datapoint = false;
            break;
        }
        case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            result->labels = ggml_dup_tensor(result->ctx_static, result->outputs);
            ggml_set_input(result->labels);
            result->loss = ggml_cross_entropy_loss(result->ctx_static, result->outputs, result->labels);
            ggml_set_name(result->loss, "loss_cross_entropy");
            if (result->opt_period > 1) {
                result->loss = ggml_scale(result->ctx_static, result->loss, 1.0f / result->opt_period);
                ggml_set_name(result->loss, "loss_cross_entropy_scaled");
            }
            result->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            result->labels = ggml_dup_tensor(result->ctx_static, result->outputs);
            ggml_set_input(result->labels);
            result->loss = ggml_sub(result->ctx_static, result->outputs, result->labels);
            ggml_set_name(result->loss, "loss_error");
            result->loss = ggml_sqr(result->ctx_static, result->loss);
            ggml_set_name(result->loss, "loss_squared_error");
            result->loss = ggml_sum(result->ctx_static, result->loss);
            ggml_set_name(result->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (result->opt_period * ggml_nelements(result->outputs));
            result->loss = ggml_scale(result->ctx_static, result->loss, scale);
            ggml_set_name(result->loss, "loss_mean_squared_error");
            result->loss_per_datapoint = true;
            break;
        }
    }
    ggml_set_output(result->loss);
    ggml_set_loss(result->loss);
    ggml_build_forward_expand(result->gf, result->loss);

    result->pred = ggml_argmax(result->ctx_static, result->outputs);
    ggml_set_name(result->pred, "pred");
    ggml_set_output(result->pred);
    ggml_build_forward_expand(result->gf, result->pred);

    if (result->labels) {
        result->ncorrect = ggml_count_equal(result->ctx_static, result->pred, ggml_argmax(result->ctx_static, result->labels));
        ggml_set_name(result->ncorrect, "ncorrect");
        ggml_set_output(result->ncorrect);
        ggml_build_forward_expand(result->gf, result->ncorrect);
    } else {
        result->ncorrect = nullptr;
    }

    if (result->forward_only) {
        result->gb_grad = nullptr;
        result->gb_opt  = nullptr;

        result->buf_static = ggml_backend_alloc_ctx_tensors(result->ctx_static, ggml_backend_sched_get_backend(result->backend_sched, 0));

        ggml_opt_alloc_graph(result, result->gf);

        return result;
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    result->gb_grad = ggml_graph_dup(result->ctx_compute, result->gf);
    ggml_build_backward_expand(result->ctx_static, result->ctx_compute, result->gf, result->gb_grad, params.opt_period > 1);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    result->gb_opt = ggml_graph_dup(result->ctx_compute, result->gb_grad);

    const ggml_opt_optimizer_params op = params.optimizer_params;
    for (int i = result->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = result->gf->nodes[i];

        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            struct ggml_tensor * m = ggml_dup_tensor(result->ctx_static, node);
            struct ggml_tensor * v = ggml_dup_tensor(result->ctx_static, node);
            struct ggml_tensor * opt_step = ggml_opt_step_adamw(
                result->ctx_compute, node, node->grad, m, v,
                op.adamw.alpha, op.adamw.beta1, op.adamw.beta2, op.adamw.eps, op.adamw.wd);
            ggml_build_forward_expand(result->gb_opt, opt_step);
        }
    }

    result->buf_static = ggml_backend_alloc_ctx_tensors(result->ctx_static, ggml_backend_sched_get_backend(result->backend_sched, 0));

    ggml_opt_alloc_graph(result, result->gb_opt);
    ggml_graph_reset(result->gb_opt);

    return result;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_free(opt_ctx->ctx_static);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

static void ggml_opt_eval_graph(ggml_opt_context_t opt_ctx, ggml_cgraph * graph, ggml_opt_result * result) {
    ggml_opt_alloc_graph(opt_ctx, graph);
    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
    std::vector<int32_t> pred(ndata);
    ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
    result->pred.insert(result->pred.end(), pred.begin(), pred.end());

    if (!opt_ctx->labels || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

void ggml_opt_forward(ggml_opt_context_t opt_ctx, ggml_opt_result * result) {
    ggml_opt_eval_graph(opt_ctx, opt_ctx->gf, result);
}

void ggml_opt_forward_backward(ggml_opt_context_t opt_ctx, ggml_opt_result * result) {
    if (opt_ctx->opt_period == 1) {
        ggml_opt_eval_graph(opt_ctx, opt_ctx->gb_opt, result);
        return;
    }

    const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    if (opt_i_next == 0) {
        ggml_opt_eval_graph(opt_ctx, opt_ctx->gb_opt, result);
        ggml_opt_reset(opt_ctx, /*optimizer =*/ false);
    } else {
        ggml_opt_eval_graph(opt_ctx, opt_ctx->gb_grad, result);
    }
    opt_ctx->opt_i = opt_i_next;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    struct ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_forward_backward(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_forward(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    constexpr int64_t bar_length = 25;
    for (int64_t j = 0; j < bar_length; ++j) {
        const int64_t ibatch_j = ibatch_max * j/bar_length;
        if (ibatch_j < ibatch) {
            fprintf(stderr, "=");
        } else if (ibatch_max * (j - 1)/bar_length < ibatch) {
            fprintf(stderr, ">");
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "| data=%06" PRId64 "/%06" PRId64 ", loss=%.6lf+-%.6lf, accuracy=%.2lf+-%.2lf%%, "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 ", ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 "]\r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t        backend_sched,
        ggml_context              * ctx_compute,
        ggml_tensor               * inputs,
        ggml_tensor               * outputs,
        ggml_opt_dataset_t          dataset,
        enum ggml_opt_loss_type     loss_type,
        ggml_opt_optimizer_params   optimizer_params,
        int64_t                     nepoch,
        int64_t                     nbatch_logical,
        float                       val_split,
        bool                        silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, ctx_compute, inputs, outputs, loss_type);
    params.optimizer_params = optimizer_params;
    params.opt_period = opt_period;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (int64_t epoch = 1; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}
