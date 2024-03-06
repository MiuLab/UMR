import argparse
import csv
import json


def convert_nq_to_jsonl(input_file, output_file, split):
    with open(input_file, "r") as f:
        # The NQ dataset is tab separated without a header row.
        reader = csv.DictReader(f, delimiter="\t", fieldnames=["text", "answer"])
        with open(output_file, "w") as out:
            for i, row in enumerate(reader):
                answers = eval(row["answer"])
                d = {
                    "id": split + "-" + str(i),
                    "question": row["text"],
                    "answers": answers,
                    "lang": "en",
                    "split": split,
                }
                out.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("split", type=str)
    args = parser.parse_args()

    convert_nq_to_jsonl(args.input_file, args.output_file, args.split)
