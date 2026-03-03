import argparse
import os
import pandas as pd
from concept_recognition import *
from entity_linking import entity_linking_from_term
from openai import OpenAI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run concept recognition + entity linking from an input text.")
    p.add_argument("--api-key", dest="api_key", help="OpenAI API key")
    p.add_argument("--text", dest="text", required=True, help="Input text to analyze (e.g., 'A proband with high temperature').")
    p.add_argument("--process-json-path", dest="json_path", default="json_path", help="Path to JSON resources (if required by concept_recognition_from_text).")
    p.add_argument("--output-json-path", dest="output_json_path", default="output_json_path", help="Where to write output JSON (if used by concept_recognition_from_text).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key as OPENAI_API_KEY")

    client_open_ai = OpenAI(api_key=args.api_key)

    # ------------------ launch concept recognition ------------------
    df = concept_recognition_from_text(args.text, client_open_ai, args.json_path, args.output_json_path)
    print('CONCEPT RECOGNITION RESULTS')
    print(df)

    # ------------------ launch entity linking -----------------------
    for i, row in df.iterrows():
        (top_1, top_1_hpo, top_1_rag, top_1_hpo_rag, matching_keys_top_1, matching_keys_top_1_rag) = entity_linking_from_term(row["phenotype"], client_open_ai)

        df.at[i, "mapped_HPO"] = top_1_rag
        df.at[i, "mapped_HPO_ID"] = top_1_hpo_rag

    print('ENTITY LINKING RESULTS')
    print(df)


if __name__ == "__main__":
    main()
