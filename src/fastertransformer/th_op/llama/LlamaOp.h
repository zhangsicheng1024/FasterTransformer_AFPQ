#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/word_list.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"

#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFLlama {
public:
    virtual ~IFLlama() {};
    virtual std::vector<th::Tensor> forward(th::Tensor               input_ids,
                                            th::Tensor               input_lengths,
                                            const int64_t            request_output_len,
                                            th::optional<th::Tensor> bad_words_list_opt,
                                            th::optional<th::Tensor> stop_words_list_opt) = 0;
};

template<typename T>
class FTLlama: public IFLlama {
public:
    FTLlama(const INIReader reader)
    {
        init(reader);
    }

    ~FTLlama() override
    {
        ftNcclParamDestroy(tensor_para_);
        ftNcclParamDestroy(pipeline_para_);

        cudaStreamDestroy(stream_);
        cublasDestroy(cublas_handle_);
        cublasLtDestroy(cublaslt_handle_);

        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
        delete cublas_wrapper_;

        delete gpt_weights_;
        delete gpt_;
    }

    std::vector<th::Tensor> forward(th::Tensor               input_ids,
                                    th::Tensor               input_lengths,
                                    const int64_t            request_output_len,
                                    th::optional<th::Tensor> bad_words_list_opt,
                                    th::optional<th::Tensor> stop_words_list_opt) override
    {
        CHECK_TH_CUDA(input_ids);
        CHECK_CONTIGUOUS(input_ids);
        TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
        CHECK_TH_CUDA(input_lengths);
        CHECK_CONTIGUOUS(input_lengths);
        TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");

        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_len      = (size_t)input_ids.size(1);

        const int total_output_len = max_input_len + request_output_len;
        std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);
        std::vector<int> start_ids(request_batch_size, start_id_);
        std::vector<int> end_ids(request_batch_size, end_id_);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size, max_input_len}, get_ptr<int>(input_ids)}},
            {"input_lengths", ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            // NOTE: if you need prefix prompts, remember to add prefix_prompt_task_ids here
            // {"prompt_learning_task_name_ids", Tensor{ft::MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size},
            // prefix_prompt_task_ids.data()}},
            {"output_seq_len",
            ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}},
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature_}},
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty_}},
            {"min_length", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &min_length_}},
            {"start_id", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, start_ids.data()}},
            {"end_id", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, end_ids.data()}}};

        if (bad_words_list_opt.has_value()) {
            CHECK_INPUT(bad_words_list_opt.value(), torch::kInt32);
            input_tensors.insert({"bad_words_list", convert_tensor<int>(bad_words_list_opt.value())});
        }

        if (stop_words_list_opt.has_value()) {
            CHECK_INPUT(bad_words_list_opt.value(), torch::kInt32);
            input_tensors.insert({"stop_words_list", convert_tensor<int>(stop_words_list_opt.value())});
        }

        if (repetition_penalty_ != 1.0f) {
            input_tensors.insert(
                {"repetition_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty_}});
        }
        if (presence_penalty_ != 0.0f) {
            input_tensors.insert(
                {"presence_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &presence_penalty_}});
        }

        // NOTE: task_name_ids for each sequence in one batch
        // Each sequence can have different prompt learning task ids
        std::vector<int> prefix_prompt_task_ids(request_batch_size, 0);

        // Set different task ids
        for (int i = 0; i < request_batch_size; i++) {
            prefix_prompt_task_ids[i] = (num_tasks_ > 0) ? i % num_tasks_ : 0;
        }

        if (num_tasks_ > 0) {
            // Prefix Prompt Task Name Ids here
            input_tensors.insert(
                {"prompt_learning_task_name_ids",
                ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, prefix_prompt_task_ids.data()}});
        }

        if (top_k_ == 0 && top_p_ == 0.0f) {
            ft::FT_CHECK(beam_width_ > 1);
            input_tensors.insert({"beam_search_diversity_rate",
                                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate_}});
        }
        else {
            input_tensors.insert({"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed_}});
            if (top_p_ != 0.0f) {
                input_tensors.insert({"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p_}});
            }
            if (top_k_ != 0) {
                input_tensors.insert({"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{1}, &top_k_}});
            }
        }

        th::Tensor output_ids = torch::empty({request_batch_size, beam_width_, (size_t)total_output_len},
                                            torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        th::Tensor sequence_lengths =
            torch::empty({request_batch_size, beam_width_}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        th::Tensor output_log_probs =
            torch::empty({(size_t)request_output_len, request_batch_size, beam_width_}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       std::vector<size_t>{request_batch_size, beam_width_, (size_t)total_output_len},
                       get_ptr<int>(output_ids)}},
            {"sequence_length",
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width_}, get_ptr<int>(sequence_lengths)}},
            {"output_log_probs",
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_FP32,
                       std::vector<size_t>{(size_t)request_output_len, request_batch_size, beam_width_},
                       get_ptr<float>(output_log_probs)}}};

        gpt_->forward(&output_tensors, &input_tensors, gpt_weights_);

        return std::vector<th::Tensor>{output_ids, sequence_lengths, output_log_probs};
    }

private:
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    ft::cublasAlgoMap* cublas_algo_map_;
    ft::Allocator<ft::AllocatorType::CUDA>* allocator_;
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasMMWrapper* cublas_wrapper_;
    struct cudaDeviceProp prop_;
    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    ft::LlamaWeight<T>* gpt_weights_;
    ft::AttentionType attention_type_;
    ft::Llama<T>* gpt_;

    int start_id_;
    int end_id_;
    float temperature_;
    float len_penalty_;
    int min_length_;
    float repetition_penalty_;
    float presence_penalty_;
    int num_tasks_;
    uint top_k_;
    float top_p_;
    float beam_search_diversity_rate_;
    size_t beam_width_;
    unsigned long long random_seed_;

    void init(const INIReader reader)
    {
        const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
        std::string       model_dir  = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));

        int tensor_para_size   = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
        int pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");
        int int8_mode  = reader.GetInteger("ft_instance_hyperparameter", "int8_mode", 0);

        const size_t head_num             = reader.GetInteger(model_name, "head_num");
        const size_t size_per_head        = reader.GetInteger(model_name, "size_per_head");
        const size_t vocab_size           = reader.GetInteger(model_name, "vocab_size");
        const size_t decoder_layers       = reader.GetInteger(model_name, "num_layer");
        const size_t rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
        const float  layernorm_eps        = reader.GetFloat(model_name, "layernorm_eps");
        const int    start_id             = reader.GetInteger(model_name, "start_id");
        const int    end_id               = reader.GetInteger(model_name, "end_id");

        const size_t hidden_units = head_num * size_per_head;
        const size_t inter_size   = reader.GetInteger(model_name, "inter_size");

        const size_t beam_width                 = reader.GetInteger("request", "beam_width");
        const uint   top_k                      = (uint)reader.GetInteger("request", "top_k");
        const float  top_p                      = reader.GetFloat("request", "top_p");
        const float  temperature                = reader.GetFloat("request", "temperature");
        const float  repetition_penalty         = reader.GetFloat("request", "repetition_penalty", 1.0f);
        const float  presence_penalty           = reader.GetFloat("request", "presence_penalty", 0.0f);
        const float  len_penalty                = reader.GetFloat("request", "len_penalty");
        const float  beam_search_diversity_rate = reader.GetFloat("request", "beam_search_diversity_rate");
        const int    min_length                 = reader.GetInteger("request", "min_length", 0);
        const size_t request_batch_size         = (size_t)reader.GetInteger("request", "request_batch_size");
        // The length of tokens we hope this model to generate
        const int request_output_len = reader.GetInteger("request", "request_output_len");
        const int warmup_ite = reader.GetInteger("request", "warmup_ite", 1);
        const int run_ite = reader.GetInteger("request", "run_ite", 1);

        start_id_ = start_id;
        end_id_ = end_id;
        temperature_ = temperature;
        len_penalty_ = len_penalty;
        min_length_ = min_length;
        repetition_penalty_ = repetition_penalty;
        presence_penalty_ = presence_penalty;
        beam_search_diversity_rate_ = beam_search_diversity_rate;
        top_k_ = top_k;
        top_p_ = top_p;
        beam_width_ = beam_width;

        ft::FT_CHECK(head_num % tensor_para_size == 0);
        ft::FT_CHECK(decoder_layers % pipeline_para_size == 0);
        FT_CHECK_WITH_INFO(
            repetition_penalty == 1.0f || presence_penalty == 0.0f,
            ft::fmtstr("Found ambiguous parameters repetition_penalty (%f) and presence_penalty (%f) "
                "which are mutually exclusive. Please remove one of repetition_penalty or presence_penalty "
                "or set to a default value.",
                repetition_penalty,
                presence_penalty));

        // Prepare the parallelism parameters
        int rank       = ft::mpi::getCommWorldRank();
        int world_size = ft::mpi::getCommWorldSize();
        if (rank == 0) {
            printf("Total ranks: %d.\n", world_size);
        }
        int device, device_count;
        ft::check_cuda_error(cudaGetDeviceCount(&device_count));
        ft::check_cuda_error(cudaSetDevice(rank % device_count));
        ft::check_cuda_error(cudaGetDevice(&device));

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device));
        printf("Device %s\n", prop_.name);

        printf("P%d is running with GPU #%d.\n", rank, device);
        if (tensor_para_size * pipeline_para_size != world_size) {
            if (world_size % pipeline_para_size) {
                printf("[ERROR] tensor_para_size * pipeline_para_size should equal to world_size \n");
                exit(-1);
            }
            tensor_para_size = world_size / pipeline_para_size;
            printf("[INFO] Setting tensor_para_size to %d \n", tensor_para_size);
        }

        const int layers_per_group = decoder_layers / pipeline_para_size;
        if (layers_per_group * pipeline_para_size != (int)decoder_layers) {
            printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
                layers_per_group,
                pipeline_para_size,
                decoder_layers);
            exit(-1);
        }

        // assume gpu_num = k * n,
        // tensor parallelism group size is n
        // pipeline parallelism group size is k
        ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

        // Prompt Learning Configurations
        // NOTE: if you don't need prefix prompts, remember to set max_prefix_len to 0 and others to nullptr
        int prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
        fastertransformer::PromptLearningType prompt_learning_type =
            static_cast<fastertransformer::PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));

        // NOTE: specify task names, take name id, prompt length in order to load those prompt learning tables.
        // NOTE: Please make sure task ids are continuous and start from 0
        // for example:
        // std::map<std::string, std::pair<int, int>> prefix_prompt_table_pair{{"no_prompt", {0, 0}},
        //                                                                     {"prompt_1", {1, 1}},
        //                                                                     {"prompt_2", {2, 2}},
        //                                                                     {"prompt_3", {3, 3}},
        //                                                                     {"prompt_4", {4, 4}},
        //                                                                     {"prompt_5", {5, 5}}};

        std::map<std::string, std::pair<int, int>> prefix_prompt_table_pair;

        // NOTE: get prompt table pairs from configuration files
        const int num_tasks = reader.GetInteger(model_name, "num_tasks", 0);
        for (int task_name_id = 0; task_name_id < num_tasks; task_name_id++) {
            std::string config_task_name = model_name + "_task_" + std::to_string(task_name_id);
            std::string task_name        = reader.Get(config_task_name, "task_name");
            const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
            prefix_prompt_table_pair.insert({task_name, {task_name_id, prompt_length}});
        }

        num_tasks_ = num_tasks;

        cudaStreamCreate(&stream_);
        cublasCreate(&cublas_handle_);
        cublasLtCreate(&cublaslt_handle_);
        cublasSetStream(cublas_handle_, stream_);
        cublas_algo_map_ = new ft::cublasAlgoMap("gemm_config.in");

        allocator_ = new ft::Allocator<ft::AllocatorType::CUDA>(ft::getDevice());

        cublas_wrapper_mutex_ = new std::mutex();
        cublas_wrapper_ = new ft::cublasMMWrapper(cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
        if (std::is_same<T, half>::value) {
            cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
    #ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper_->setBF16GemmConfig();
        }
    #endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper_->setFP32GemmConfig();
        }

        const bool use_gptj_residual = false;
        gpt_weights_ = new fastertransformer::LlamaWeight<T>(hidden_units,
                                                            inter_size,
                                                            vocab_size,
                                                            decoder_layers,
                                                            0,  // max_seq_len, deprecated
                                                            tensor_para_.world_size_,
                                                            tensor_para_.rank_,
                                                            pipeline_para_.world_size_,
                                                            pipeline_para_.rank_,
                                                            use_gptj_residual,
                                                            int8_mode,
                                                            prompt_learning_type,
                                                            prefix_prompt_table_pair);

        gpt_weights_->loadModel(model_dir);
        if (rank == 0) {
            random_seed_ = (unsigned long long)(0);
        }
        if (world_size > 1) {
            ft::mpi::bcast(&random_seed_, 1, ft::mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, ft::mpi::COMM_WORLD);
        }

        attention_type_ = ft::getAttentionType<T>(size_per_head,
                                                  ft::getSMVersion(),
                                                  true,   // remove_padding
                                                  0,      // gpt supports any-seq-length fmha
                                                  true,   // is_fuse
                                                  false,  // with_relative_position_bias
                                                  true);  // causal_mask

        gpt_ = new ft::Llama<T>(head_num,
                                size_per_head,
                                inter_size,
                                decoder_layers,
                                vocab_size,
                                rotary_embedding_dim,
                                layernorm_eps,
                                start_id,
                                end_id,
                                prompt_learning_start_id,
                                prompt_learning_type,
                                use_gptj_residual,
                                0.0f,
                                top_k,
                                top_p,
                                random_seed_,
                                temperature,
                                len_penalty,
                                repetition_penalty,
                                tensor_para_,
                                pipeline_para_,
                                stream_,
                                cublas_wrapper_,
                                allocator_,
                                false,
                                &prop_,
                                attention_type_,
                                int8_mode,
                                nullptr,
                                0,
                                1.0f);
    }
};

class LlamaOp: public th::jit::CustomClassHolder {
public:
    LlamaOp(const std::string ini_name);
    std::vector<th::Tensor> forward(th::Tensor               input_ids,
                                    th::Tensor               input_lengths,
                                    const int64_t            request_output_len,
                                    th::optional<th::Tensor> bad_words_list_opt,
                                    th::optional<th::Tensor> stop_words_list_opt);
    ~LlamaOp();

private:
    IFLlama* ftllama_;
};

}
