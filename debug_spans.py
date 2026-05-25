import json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract spans from a JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("--line", type=int, default=0, help="Line number to extract spans from (default: 0).")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        for i, line in enumerate(f):
            if i == args.line:
                data = json.loads(line)
                spans = data.get("annotations", {}).get("token_edits", [])
                print("spans: ", " | ".join([span["text"] for span in spans]))
                break