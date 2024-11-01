import regex as re
from datasets import load_dataset, Dataset


# Clean inline comments and block comments
def clean_comments(code: str) -> str:
    # Remove inline comments using #
    no_comment = re.sub(
        pattern=r"#[^\n]*",
        repl="",
        string=code)

    # Remove """ docstrings at the start of a file
    no_leading_docstring = re.sub(
        pattern=r'\A\s*"""((?:(?!""")[\s\S])*)"""',
        repl="",
        string=no_comment
    )

    # Remove doc strings (limited to be following def or class)
    # https://stackoverflow.com/questions/1687620/regex-match-everything-but-a-specific-pattern
    no_docstring = re.sub(
        pattern=r'(class|def)(.+)\s+"""((?:(?!""")[\s\S])*)"""',
        repl="",
        string=no_leading_docstring
    )

    # Remove extra whitespace
    return re.sub(
        pattern=r"\s*\n",
        repl="\n",
        string=no_docstring)


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

def keep_only_content(sample: dict) -> dict:
    # TODO: Is there a way to remove other keys instead of forcing []?
    for key in sample.keys():
        if key == "content":
            continue

        sample[key] = []
    
    return sample


if __name__ == "__main__":
    ds = get_data()
    i = 0
    for x in ds:
        print(x["content"])
        i += 1
        if i == 100:
            break
