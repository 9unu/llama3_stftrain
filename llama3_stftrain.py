# import os
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     pipeline,
#     logging,
# )
# from peft import LoraConfig
# from trl import SFTTrainer

import torch
  
if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
else:
    print("CUDA is not available. Using CPU.")


# # Hugging Face Basic Model 한국어 모델
# base_model = "beomi/Llama-3-Open-Ko-8B"  # beomi님의 Llama3 한국어 파인튜닝 모델

# # baemin_dataset_simple.json
# dataset = "train_data.json"

# # 새로운 모델 이름
# new_model = "llama3-sft"

# dataset = load_dataset(dataset, split="train")

# # 데이터 확인
# print(len(dataset))
# print(dataset[0])

# if torch.cuda.get_device_capability()[0] >= 8:
#     !pip install -qqq flash-attn
#     attn_implementation = "flash_attention_2"
#     torch_dtype = torch.bfloat16
# else:
#     attn_implementation = "eager"
#     torch_dtype = torch.float16

# # QLoRA config
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=False,
# )

# # llama 데이터 로드
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     quantization_config=quant_config,
#     device_map={"": 0}
# )
# model.config.use_cache = False
# model.config.pretraining_tp = 1

# # 토크나이저 로드
# tokenizer = AutoTokenizer.from_pretrained(
#     base_model,
#     trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# peft_params = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# training_params = TrainingArguments(
#     output_dir="/results",
#     num_train_epochs=1,  # epoch는 1로 설정
#     max_steps=5000,  # max_steps을 5000으로 설정
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     optim="paged_adamw_8bit",
#     warmup_steps=0.03,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=100,
#     push_to_hub=False,
#     report_to='none',
# )

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_params,
#     dataset_text_field="text",
#     max_seq_length=256,
#     tokenizer=tokenizer,
#     args=training_params,
#     packing=False,
# )

# trainer.train()

# trainer.save_model(new_model)

# prompt = "알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?"
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])