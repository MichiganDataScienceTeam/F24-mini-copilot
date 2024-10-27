import regex as re
from datasets import load_dataset, Dataset, IterableDataset


# Clean inline comments and block comments
def clean_comments(code: str) -> str:
    no_comment = re.sub(r"#[^\n]*", "", code)  # remove comments '# comment'

    # remove doc strings (limited to be following def or class)
    # https://stackoverflow.com/questions/1687620/regex-match-everything-but-a-specific-pattern
    no_docstring = re.sub(
        r'(class|def)(.+)\s+"""((?:(?!""")[\s\S])*)"""', "", no_comment
    )
    return re.sub(r"\s*\n", "\n", no_docstring)  # remove trailing whitespace


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
        b = "from " + library
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
    ds = ds.map(lambda x: {"content": clean_comments(x["content"])})
    ds = ds.filter(lambda x: include(x["content"]))
    return ds


if __name__ == "__main__":
    ds = get_data()
    i = 0
    for x in ds:
        print(x["content"])
        i += 1
        if i == 100:
            break
