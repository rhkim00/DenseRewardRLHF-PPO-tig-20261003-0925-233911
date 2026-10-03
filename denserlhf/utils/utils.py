import os

from datasets import interleave_datasets, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        # -----------------------------
        # 🔹 DatasetDict / Dataset 처리
        # -----------------------------
        if isinstance(data, DatasetDict):
            # train split 이름 파싱 (예: 'train[:256]' → 'train')
            split_for_train = train_split or "train"
            if isinstance(split_for_train, str) and ":" in split_for_train:
                split_for_train = split_for_train.split("[")[0].split(":")[0]

            if split_for_train not in data:
                # 기본 fallback
                if "train" in data:
                    split_for_train = "train"
                else:
                    split_for_train = list(data.keys())[0]

            base_train = data[split_for_train]
            train_data = base_train.select(range(min(max_count, len(base_train))))

            if return_eval:
                split_for_eval = eval_split or "test"
                if isinstance(split_for_eval, str) and ":" in split_for_eval:
                    split_for_eval = split_for_eval.split("[")[0].split(":")[0]

                if split_for_eval not in data:
                    # eval split이 없으면 train split 재사용
                    if "test" in data:
                        split_for_eval = "test"
                    else:
                        split_for_eval = split_for_train

                base_eval = data[split_for_eval]
                eval_data = base_eval.select(range(min(max_count, len(base_eval))))
        else:
            # Dataset인 경우: 기존 로직 유지하되, train_split은 무시하고 전체에서 max_count만큼 사용
            if train_split and isinstance(data, dict) and train_split in data:
                # (거의 안 쓰이는 케이스지만 안전하게 유지)
                train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
            else:
                train_data = data.select(range(min(max_count, len(data))))

            if return_eval:
                if eval_split and isinstance(data, dict) and eval_split in data:
                    eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
                else:
                    eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))

        train_data_list.append(train_data)

        if return_eval:
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
