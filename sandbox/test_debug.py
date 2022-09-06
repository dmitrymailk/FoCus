import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description="Optional app description")
parser.add_argument(
    "--test",
    type=str,
)

print(parser.parse_args())
