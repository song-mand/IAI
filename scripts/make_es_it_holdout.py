import argparse
import os
from datasets import load_dataset, concatenate_datasets


def load_source_train(args):
    if args.source_train_file:
        ds = load_dataset(
            "parquet",
            data_files={"train": args.source_train_file},
        )["train"]
    else:
        ds = load_dataset(
            args.dataset_name,
            split="train",
            token=True if args.token else None,
        )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_train_file", type=str, default="train-00000-of-00001.parquet")
    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--token", action="store_true", help="Use Hugging Face auth token when loading from Hub.")
    parser.add_argument("--langs", nargs="+", default=["es", "it"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./eval_splits")
    parser.add_argument("--prefix", type=str, default="es_it")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_source_train(args)

    train_parts = []
    valid_parts = []

    print("Source rows:", len(dataset))

    for lang in args.langs:
        lang_data = dataset.filter(lambda x, lang=lang: x["lang"] == lang)
        print(f"[{lang}] rows:", len(lang_data))

        if len(lang_data) == 0:
            print(f"[{lang}] skipped: no rows")
            continue

        split = lang_data.train_test_split(test_size=args.test_size, seed=args.seed)
        print(f"[{lang}] train rows:", len(split["train"]))
        print(f"[{lang}] valid rows:", len(split["test"]))

        train_parts.append(split["train"])
        valid_parts.append(split["test"])

    if not train_parts:
        raise ValueError("No data found for selected languages.")

    train_data = concatenate_datasets(train_parts)
    valid_data = concatenate_datasets(valid_parts)

    train_path = os.path.join(args.out_dir, f"{args.prefix}_train.parquet")
    valid_path = os.path.join(args.out_dir, f"{args.prefix}_valid.parquet")

    train_data.to_parquet(train_path)
    valid_data.to_parquet(valid_path)

    print("\nSaved:")
    print("train:", train_path, "rows:", len(train_data))
    print("valid:", valid_path, "rows:", len(valid_data))


if __name__ == "__main__":
    main()
