#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <inttypes.h>
#include <thread>
#include <vector>

static bool test_dataset(ggml_backend_t backend) {
    bool ok = true;

    constexpr int64_t ne_datapoint = 2;
    constexpr int64_t ne_label     = 1;
    constexpr int64_t ndata        = 6;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ndata*2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<struct ggml_tensor *>   data_batch(ndata);
    std::vector<struct ggml_tensor *> labels_batch(ndata);
    for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
        data_batch[ndata_batch-1]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ndata_batch*ne_datapoint);
        labels_batch[ndata_batch-1] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ndata_batch*ne_label);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        struct ggml_opt_new_dataset * dataset = ggml_opt_new_dataset_init(ne_datapoint, ne_label, ndata, ndata_shard);

        {
            float * data   = ggml_get_data_f32(ggml_opt_new_dataset_data(  dataset));
            float * labels = ggml_get_data_f32(ggml_opt_new_dataset_labels(dataset));

            for (int64_t idata = 0; idata < ndata; ++idata) {
                for (int64_t id = 0; id < ne_datapoint; ++id) {
                    data[  idata*ne_datapoint + id] =     16*idata + id;
                }
                for (int64_t il = 0; il < ne_label;     ++il) {
                    labels[idata*ne_label     + il] = 16*(16*idata + il);
                }
            }
        }

        for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
            if (ndata_batch % ndata_shard != 0) {
                continue;
            }
            bool subtest_ok = true;

            std::vector<float>   data(ggml_nelements(  data_batch[ndata_batch-1]));
            std::vector<float> labels(ggml_nelements(labels_batch[ndata_batch-1]));

            const int64_t nbatches = ndata / ndata_batch;
            for (int64_t ibatch = 0; subtest_ok && ibatch < nbatches; ++ibatch) {
                ggml_opt_new_dataset_get_batch(dataset, data_batch[ndata_batch-1], labels_batch[ndata_batch-1], ibatch);

                ggml_backend_tensor_get(  data_batch[ndata_batch-1],   data.data(), 0, ggml_nbytes(  data_batch[ndata_batch-1]));
                ggml_backend_tensor_get(labels_batch[ndata_batch-1], labels.data(), 0, ggml_nbytes(labels_batch[ndata_batch-1]));

                for (int64_t idata_batch = 0; subtest_ok && idata_batch < ndata_batch; ++idata_batch) {
                    const int64_t idata = ibatch*ndata_batch + idata_batch;

                    for (int64_t id = 0; subtest_ok && id < ne_datapoint; ++id) {
                        if (data[  idata_batch*ne_datapoint + id] != 16*idata + id) {
                            subtest_ok = false;
                        }
                    }
                    for (int64_t il = 0; subtest_ok && il < ne_label;     ++il) {
                        if (labels[idata_batch*ne_label     + il] != 16*(16*idata + il)) {
                            subtest_ok = false;
                        }
                    }
                }
            }

            printf("  test_dataset(shuffle=0, ndata_shard=%" PRId64 ", ndata_batch=%" PRId64 "): ", ndata_shard, ndata_batch);
            if (subtest_ok) {
                printf("\033[1;32mOK\033[0m\n");
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
                ok = false;
            }
        }

        ggml_opt_new_dataset_free(dataset);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

static bool test_backend(ggml_backend_t backend) {
    bool ok = true;

    ok = ok && test_dataset(backend);

    return ok;
}

int main(void) {
    printf("Testing %zu devices\n\n", ggml_backend_dev_count());
    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        printf("Backend %zu/%zu: %s\n", i + 1, ggml_backend_dev_count(), ggml_backend_dev_name(dev));

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, std::thread::hardware_concurrency() / 2);
        }

        printf("  Device description: %s\n", ggml_backend_dev_description(dev));
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        const bool ok = test_backend(backend);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_dev_count());
    if (n_ok != ggml_backend_dev_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }
    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
