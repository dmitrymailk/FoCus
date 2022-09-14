from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="",
    help="{GPT2, BART, transformer-decoder, transformer-encdec}",
)

args = parser.parse_args()
print(vars(args))
