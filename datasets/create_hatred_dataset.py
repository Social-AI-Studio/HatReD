import os
import argparse
import pandas as pd

def construct_dataset(
        annotations_fp: str,
        reasonings_fp: str,
        race_fp: str,
        entity_fp: str,
):
    annotations_df = pd.read_json(annotations_fp, lines=True)
    annotations_df['img'] = annotations_df['img'].apply(lambda x: os.path.basename(x))
    reasonings_df = pd.read_json(reasonings_fp, lines=True)
    race_df = pd.read_json(race_fp, lines=True)
    entity_df = pd.read_json(entity_fp, lines=True)

    hatred_df = reasonings_df
    hatred_df = pd.merge(hatred_df, race_df, on=["id", "img"], how="inner")
    hatred_df = pd.merge(hatred_df, entity_df, on=["id", "img"], how="inner")
    hatred_df = pd.merge(hatred_df, annotations_df, on=["id", "img"], how="inner")

    return hatred_df
    

def main(github_dir):
    train_annotations_fp = os.path.join(github_dir, "data", "annotations", "train.json")
    test_annotations_fp = os.path.join(github_dir, "data", "annotations", "dev_seen.json")

    train_df = construct_dataset(
        train_annotations_fp,
        "fhm/annotations/fhm_train_reasonings.jsonl",
        "fhm/auxiliary/fhm_train_race.jsonl",
        "fhm/auxiliary/fhm_train_entity.jsonl"
    )

    test_df = construct_dataset(
        test_annotations_fp,
        "fhm/annotations/fhm_test_reasonings.jsonl",
        "fhm/auxiliary/fhm_test_race.jsonl",
        "fhm/auxiliary/fhm_test_entity.jsonl"
    )

    train_df.to_json("fhm/annotations/fhm_train.jsonl", orient="records", lines=True)
    test_df.to_json("fhm/annotations/fhm_test.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Construct HatReD's train and test dataset")
    parser.add_argument("--github-dir", type=str, required=True, help="system path to Facebook's Hateful Memes Finegrained Dataset")
    args = parser.parse_args()

    main(args.github_dir)