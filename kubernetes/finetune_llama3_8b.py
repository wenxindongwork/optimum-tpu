from optimum.tpu import fsdp_v2
from datasets import load_from_disk
from datasets import Dataset

from transformers import AutoTokenizer
# from optimum.tpu import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from trl import ORPOTrainer, ORPOConfig
from transformers import TrainingArguments


#model_id = "google/gemma-2b"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

fsdp_v2.use_fsdp_v2()



print("----0----")

orpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ] * 100,
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ] * 100,
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ] * 100,
}

dataset = Dataset.from_dict(orpo_dataset_dict)
dataset = dataset.train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("----1----")

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)

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


orpo_config = ORPOConfig(
    max_length=1024,
    max_prompt_length=512,
    is_encoder_decoder=False,
    dataset_num_proc=32,
    per_device_train_batch_size=64,
    num_train_epochs=32,
    max_steps=-1,
    output_dir="./output",
    optim="adafactor",
    logging_steps=1,
    dataloader_drop_last=True,  # Required for FSDPv2.
    **fsdp_training_args,
)

# Set up the trainer
trainer = ORPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=orpo_config,
)


print("----4----")

trainer.train()