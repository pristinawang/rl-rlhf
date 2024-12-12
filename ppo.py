import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForSeq2SeqLMWithValueHead
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import wandb

wandb.login(key="2f2f31c163a20a9c2a01e8b1fdcb071b4e066060")
lr=1e-5
wandb.init(
    # set the wandb project where this run will be logged
    project="rlhf-ppo",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "flan-t5-base as policy",
    "dataset": "psyche/anthropic-hh-rlhf",
    }
)
# Load dataset
dataset = load_dataset("psyche/anthropic-hh-rlhf")
train_dataset=dataset["train"]

# Reward model
reward_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")
#model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2")

#reward_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
reward_model = pipeline(
    "sentiment-analysis",
    model="OpenAssistant/reward-model-deberta-v3-large-v2",
    device="cuda",
    tokenizer=reward_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Initialize policy and value models
policy_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(policy_model_name)

# Use flan-t5-base for both policy and value models
policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(policy_model_name)


# Tokenize dataset
# def preprocess(batch):
#     tokenized = tokenizer(batch["prompt"], padding="max_length", truncation=True, return_tensors="pt")
#     tokenized["chosen"] = tokenizer(batch["chosen"], padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
#     tokenized["rejected"] = tokenizer(batch["rejected"], padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
#     return tokenized

# train_dataset = train_dataset.map(preprocess, batched=False)


def tokenize(sample):
    #sample["input_ids"] = tokenizer.encode(sample["prompt"])
    return tokenizer(sample["prompt"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize, batched=False)
train_dataset = train_dataset.remove_columns(["chosen", "rejected"])

train_dataset = train_dataset.rename_column("prompt", "query")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "query"])
# PPO Configuration
ppo_config = PPOConfig(
    model_name=policy_model_name,
    learning_rate=lr,
    batch_size=8,
    mini_batch_size=4,
    cliprange=0.2,  # Clipping parameter for PPO,
    log_with='wandb'
)
# print(train_dataset)
# print('len', len(train_dataset))
# print('vvvvvv---Ensure---vvvvvvvv')
# print(train_dataset[0])
# print('vvvvvv---Ensure---vvvvvvvv')
# print(type(train_dataset[0]['input_ids']))
# print(len(train_dataset[0]['input_ids']))
# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    model=policy_model,
    config=ppo_config,
    dataset=train_dataset,
    tokenizer=tokenizer
)

generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": 32, # specify how many tokens you want to generate at most
}

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1
}
# Training loop
def custom_collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "query": [item["query"] for item in batch],  # Keep raw text as a list
    }
    
dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=custom_collate_fn
    )

# for epoch, batch in tqdm(enumerate(dataloader)):
#     print('-------------------')
#     print(batch.keys())
#     print(type(batch["input_ids"]))
#     print(batch["input_ids"].shape)
#     print('-------------------')
#     break

for epoch, batch in tqdm(enumerate(dataloader)):
    # print('batch type', type(batch))
    # print(batch.keys())
    try:
        query_tensors = list(batch["input_ids"])
        # print('-------------------')

        # print(type(query_tensors))
        # print(query_tensors)
        # print('-------------------')
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # pipe_outputs = reward_model(texts)
        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        pipe_outputs = reward_model(texts, **pipe_kwargs)
        #print('pip output',len(pipe_outputs))
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        # print("Query tensors shape:", len(query_tensors))
        # print("Response tensors shape:", len(response_tensors))
        # print("Rewards shape:", len(rewards))


        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    except IndexError as e:
        print(f"Skipping batch due to error: {str(e)}")
        continue

#### Save model
ppo_trainer.save_model("my_ppo_modelflan-t5-ppo-finetuned")

