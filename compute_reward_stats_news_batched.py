import argparse
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm  # 진행상황 바 표시

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 모델/데이터 경로
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--rm_adapter", type=str, default="./exp/final_model")
    parser.add_argument("--dataset", type=str, default="RyunHo/news_prompt")
    parser.add_argument("--split", type=str, default="train[:]")
    parser.add_argument("--input_key", type=str, default="prompt")

    # 샘플링 / 생성 관련
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    
    # [NEW] 배치 사이즈 옵션 추가
    parser.add_argument("--batch_size", type=int, default=16, help="한 번에 처리할 배치 크기")

    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_policy_and_rm(base_model: str, rm_adapter: str):
    print(f"[+] Load tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # 배치 처리를 위해 pad_token 필수 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[+] Load base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False # RM forward시 메모리 절약
    
    print(f"[+] Load RM LoRA adapter: {rm_adapter}")
    rm_model = PeftModel.from_pretrained(
        base,
        rm_adapter,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    rm_model.eval()
    
    return tokenizer, rm_model

def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[+] Load dataset: {args.dataset} / split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)

    total = len(ds)
    n = min(args.num_samples, total)
    print(f"[+] Processing {n} samples with batch_size={args.batch_size}...")

    # 모델 로드
    tokenizer, model = load_policy_and_rm(args.base_model, args.rm_adapter)
    
    all_prompts = [ds[i][args.input_key] for i in range(n)]
    rewards = []

    # 배치 단위 루프
    for i in tqdm(range(0, n, args.batch_size), desc="Generating & Scoring"):
        batch_prompts = all_prompts[i : i + args.batch_size]
        
        # 1. Generation (Policy Mode)
        model.disable_adapter_layers() 
        tokenizer.padding_side = "left"
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                # [FIX] 에러 해결을 위해 use_cache=False 설정
                use_cache=False,
            )
        
        # 프롬프트 부분 제외하고 Response만 추출
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, input_len:]
        batch_responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # 2. Reward Calculation (RM Mode)
        model.enable_adapter_layers()
        model.config.use_cache = False
        tokenizer.padding_side = "right"
        
        full_texts = [p + "\n" + r for p, r in zip(batch_prompts, batch_responses)]
        
        rm_inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        with torch.no_grad():
            out = model(**rm_inputs)
            
            if hasattr(out, "rewards"):
                r_tensor = out.rewards
            elif hasattr(out, "logits"):
                r_tensor = out.logits
            else:
                r_tensor = out.logits

            if r_tensor.ndim == 3:
                r_tensor = r_tensor.mean(dim=-1)
            
            # 마지막 유효 토큰 인덱스 찾기
            last_token_indices = rm_inputs.attention_mask.sum(dim=1) - 1
            batch_rewards = r_tensor[torch.arange(r_tensor.size(0)), last_token_indices]
            rewards.extend(batch_rewards.float().cpu().tolist())

    # 통계 출력
    rewards_np = np.array(rewards, dtype=float)
    mean = rewards_np.mean()
    std = rewards_np.std()
    
    print("\n================ Reward Stats ================")
    print(f"#samples      : {len(rewards_np)}")
    print(f"mean          : {mean:.6f}")
    print(f"std           : {std:.6f}")
    print(f"min / max     : {rewards_np.min():.6f} / {rewards_np.max():.6f}")
    print("=============================================\n")
    print(f"  --reward_mean {mean:.6f} \\")
    print(f"  --reward_std  {std:.6f} \\")

if __name__ == "__main__":
    main()