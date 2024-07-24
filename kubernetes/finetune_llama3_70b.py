from optimum.tpu import fsdp_v2
from datasets import load_dataset
from transformers import AutoTokenizer
# from optimum.tpu import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
# import torch_xla.core.xla_model as xm


def preprocess_function(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    prompt += tokenizer.eos_token
    sample["prompt"] = prompt
    return sample


model_id = "meta-llama/Meta-Llama-3-70B"

fsdp_v2.use_fsdp_v2()

print("----0----")

dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:05%]")

tokenizer = AutoTokenizer.from_pretrained(model_id)

data = dataset.map(preprocess_function, remove_columns=list(dataset.features))

print("----1----")

model = AutoModelForCausalLM.from_pretrained("/usr/share/storage/llama3_70b/", use_cache=True, device_map="auto")

print("----2----")

# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

print("----3----")

# Set up the FSDP arguments
# fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
cls_to_wrap = "LlamaDecoderLayer"
fsdp_training_args = {
    "fsdp": "full_shard",
    "fsdp_config": fsdp_v2.get_fsdp_config(cls_to_wrap),
}
tokenizer.pad_token = tokenizer.eos_token

# print("wenxin: get_memory info")
# print(xm.get_memory_info())

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=2,
        save_steps=1,
        output_dir="./output_llama3_70b",
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last = True,  # Required for FSDPv2.
        **fsdp_training_args,
    ),
    peft_config=lora_config,
    dataset_text_field="prompt",
    max_seq_length=1024,
    packing=True,
)

print("----4----")

trainer.train()

print("----5----")
