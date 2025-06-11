import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import find_module, InsertLayer
from safetensors.torch import save_file


with torch.no_grad():
    layer = 20
    alpha = -0.1

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # Load the tokenizer        

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    mlp_keywords = ["mlp", "feedforward", "ffn"]
    w_wait = (
        torch.load(
            f"./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/DeepSeek-R1-Distill-Qwen-1.5B_hs/before_wait_DeepSeek-R1-Distill-Qwen-1.5B_{args.injection_layer}_-1.pt"
        )
        .cpu()
        .to(torch.float32)
    )
    w_wo_wait = (
        torch.load(
            f"./asset/response/DeepSeek-R1-Distill-Qwen-1.5B_MATH-500_140/DeepSeek-R1-Distill-Qwen-1.5B_hs/before_wo_wait_DeepSeek-R1-Distill-Qwen-1.5B_{args.injection_layer}_-1.pt"
        )
        .cpu()
        .to(torch.float32)
    )
    insert_vector = w_wait.mean(dim=0) - w_wo_wait.mean(dim=0)

    original_mlp = find_module(model.model.layers[layer], mlp_keywords)
    model.model.layers[19].mlp = torch.nn.Sequential(
        original_mlp,
        InsertLayer(insert_vector.to("cuda").to(torch.bfloat16), alpha=alpha),
    )
    state_dict = model.state_dict()
    save_file(state_dict, "model.safetensors")