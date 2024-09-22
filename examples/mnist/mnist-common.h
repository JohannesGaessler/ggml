#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#define MNIST_NTRAIN          60000
#define MNIST_NTEST           10000
#define MNIST_NBATCH_LOGICAL   1000
#define MNIST_NBATCH_PHYSICAL   500

static_assert(MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL == 0, "MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL != 0");
static_assert(MNIST_NTRAIN % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");
static_assert(MNIST_NTEST  % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");

#define MNIST_HW       28
#define MNIST_NINPUT   (MNIST_HW*MNIST_HW)
#define MNIST_NCLASSES 10

#define MNIST_NHIDDEN  500

// NCB = number of channels base
#define MNIST_CNN_NCB 8

struct mnist_model {
    std::string arch;
    std::vector<ggml_backend_t> backends;
    int nbatch_logical;
    int nbatch_physical;

    struct ggml_tensor  * images = nullptr;
    struct ggml_tensor  * labels = nullptr;
    struct ggml_tensor  * logits = nullptr;
    struct ggml_tensor  * probs  = nullptr;
    struct ggml_tensor  * loss   = nullptr;

    struct ggml_tensor * fc1_weight = nullptr;
    struct ggml_tensor * fc1_bias   = nullptr;
    struct ggml_tensor * fc2_weight = nullptr;
    struct ggml_tensor * fc2_bias   = nullptr;

    struct ggml_tensor * conv1_kernel = nullptr;
    struct ggml_tensor * conv1_bias   = nullptr;
    struct ggml_tensor * conv2_kernel = nullptr;
    struct ggml_tensor * conv2_bias   = nullptr;
    struct ggml_tensor * dense_weight = nullptr;
    struct ggml_tensor * dense_bias   = nullptr;

    struct ggml_context * ctx_weight  = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_weight  = nullptr;
    ggml_backend_buffer_t buf_compute = nullptr;

    mnist_model(const std::string & backend_name) {
        std::vector<std::string> backend_names = {backend_name};
        if (backend_name != "CPU") {
            backend_names.push_back("CPU");
        }
        for (const std::string & bn : backend_names) {
            const size_t backend_index = ggml_backend_reg_find_by_name(bn.c_str());
            if (backend_index == SIZE_MAX) {
                fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, bn.c_str());
                for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
                    fprintf(stderr, "  - %s\n", ggml_backend_reg_get_name(i));
                }
                exit(1);
            }

            ggml_backend_t be = ggml_backend_reg_init_backend(backend_index, nullptr);
            if (ggml_backend_is_cpu(be)) {
                const int ncores_logical = std::thread::hardware_concurrency();
                ggml_backend_cpu_set_n_threads(be, std::min(ncores_logical, (ncores_logical + 4)/2));
            }
            backends.push_back(be);
        }

        if (backends.size() == 1) {
            fprintf(stderr, "%s: using %s backend\n", __func__, ggml_backend_name(backends[0]));
        } else if (backends.size() == 2) {
            fprintf(stderr, "%s: using %s as primary backend with %s as fallback\n",
                    __func__, ggml_backend_name(backends[0]), ggml_backend_name(backends[1]));
        } else {
            GGML_ASSERT(false);
        }

        {
            const size_t size_meta = 1024*ggml_tensor_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_weight = ggml_init(params);
        }

        {
            // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
            const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~mnist_model() {
        ggml_free(ctx_weight);
        ggml_free(ctx_compute);

        ggml_backend_buffer_free(buf_weight);
        ggml_backend_buffer_free(buf_compute);
        for (ggml_backend_t be : backends) {
            ggml_backend_free(be);
        }
    }
};

struct mnist_eval_result {
    bool success = false;

    std::vector<float>   loss;
    std::vector<int32_t> pred;
};

bool mnist_image_load(const std::string & fname, float * buf, const int nex);
void mnist_image_print(FILE * f, const float * image);
bool mnist_label_load(const std::string & fname, float * buf, const int nex);

mnist_eval_result mnist_graph_eval(const std::string & fname, const float * images, const float * labels, const int nex, const int nthreads);

mnist_model       mnist_model_init_from_file(const std::string & fname, const std::string & backend);
mnist_model       mnist_model_init_random(const std::string & arch, const std::string & backend);
void              mnist_model_build(mnist_model & model, const int nbatch_logical, const int nbatch_physical);
mnist_eval_result mnist_model_eval(mnist_model & model, const float * images, const float * labels, const int nex);
void              mnist_model_train(mnist_model & model, const float * images, const float * labels, const int nex, const int nepoch, const float val_split);
void              mnist_model_save(mnist_model & model, const std::string & fname);

std::pair<double, double> mnist_loss(const mnist_eval_result & result);
std::pair<double, double> mnist_accuracy(const mnist_eval_result & result, const float * labels);
