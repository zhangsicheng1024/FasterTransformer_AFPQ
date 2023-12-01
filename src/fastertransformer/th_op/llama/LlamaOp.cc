#include "src/fastertransformer/th_op/llama/LlamaOp.h"

namespace th = torch;
namespace ft = fastertransformer;
namespace torch_ext {

LlamaOp::LlamaOp(const std::string ini_name)
{
    srand(0);

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        FT_LOG_ERROR("[ERROR] Can't load '" + ini_name);
        std::exit(EXIT_FAILURE);
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    // TODO:
    // To support more data types,
    // need a interface IFLlama
    if (data_type == "fp32") {
        ftllama_ = new FTLlama<float>(reader);
    }
    else if (data_type == "fp16") {
        ftllama_ = new FTLlama<half>(reader);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        ftllama_ = new FTLlama<__nv_bfloat16>(reader);
    }
#endif
    else {
        FT_LOG_ERROR("is_fp16 should be 0 (use float) or 1 (use half).");
        std::exit(EXIT_FAILURE);
    }
}

std::vector<th::Tensor> LlamaOp::forward(th::Tensor               input_ids,
                                         th::Tensor               input_lengths,
                                         const int64_t            request_output_len,
                                         th::optional<th::Tensor> bad_words_list_opt,
                                         th::optional<th::Tensor> stop_words_list_opt)
{
    return ftllama_->forward(input_ids,
                             input_lengths,
                             request_output_len,
                             bad_words_list_opt,
                             stop_words_list_opt);
}

LlamaOp::~LlamaOp()
{
    delete ftllama_;
}

}

static auto fasterTransformerLlamaTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::LlamaOp>("FasterTransformerLlamaOp")
#else
    torch::jit::class_<torch_ext::LlamaOp>("FasterTransformer", "LlamaOp")
#endif
        .def(torch::jit::init<std::string>())
        .def("forward", &torch_ext::LlamaOp::forward);

