from collections.abc import MutableSet
import regex as re
from datasets import load_dataset, Dataset


# Im not sure this was the best approach for doc comments
# Clean inline comments and block comments
def clean_comments(code: str) -> str:
    # remove comments with #
    no_comment = re.sub(r"#[^\n]*", "", code)

    string_pre = r"(\S+\s*=\s*)"
    between = r'(""")((?:(?!""")[\s\S])*)(""")'

    # this should not hit
    holder = "!|<1multiline1>|!"

    # initially save mutliline strings with flag value
    save_strings = re.sub(
        string_pre + between,
        rf"\1{holder}\3{holder}",
        no_comment,
    )

    # removing docs without caring about mutliline
    remove_docs = re.sub(between, "", save_strings)

    # fix flags back to original
    fix_strings = re.sub(holder, '"""', remove_docs)

    # get rid of trailing
    no_trailing = re.sub(r"\s*\n", "\n", fix_strings)
    return no_trailing


def include(content: str) -> bool:
    libraries = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "tensorflow",
        "torch",
        "scipy",
    ]
    for library in libraries:
        a = "import " + library
        b = "from " + library + " import"
        if a in content or b in content:
            return True
    return False


def get_data():
    ds = load_dataset(
        "codeparrot/codeparrot-clean",
        streaming=True,
        trust_remote_code=True,
        split="train",
    )

    ds = ds.filter(lambda x: include(x["content"]))
    ds = ds.map(lambda x: {"content": clean_comments(x["content"])})
    return ds


def preview():
    ds = get_data()
    i = 0
    for x in ds:
        print(x["content"])
        i += 1
        if i == 100:
            break


if __name__ == "__main__":
    pass
