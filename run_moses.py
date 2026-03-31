from pathlib import Path


def main() -> None:
    direct_path = Path("/weka/oe-adapt-default/michaeln/moses/molecules")
    novomolgen_path = Path("/weka/oe-adapt-default/michaeln/moses/NovoMolGen/molecules")

    if direct_path.exists():
        print(f"Found molecules directory: {direct_path}")
    elif novomolgen_path.exists():
        print(f"Found molecules directory in NovoMolGen: {novomolgen_path}")
    else:
        print("Did not find molecules directory in either expected location.")

    import moses  # noqa: F401

    print("Imported moses successfully.")


if __name__ == "__main__":
    main()
