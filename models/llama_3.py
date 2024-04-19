import transformers
from transformers import AutoModel
import torch

# Versucht das neue Meta-Llama-3-8B zu run als potentieller gpt_eval anstatt ChatGPT4, denke mal dass ich aber leider nicht genug RAM habe,
# da process "Killed" wird nach "Loading checkpoint shards:   0%| "

access_token="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"
model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModel.from_pretrained(model_id, token=access_token)

pipeline = transformers.pipeline(
    "text-generation", model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
pipeline("Hey how are you doing today?")