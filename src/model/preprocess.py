import regex as re

# -----------------------  Implementation ---------------------
def clean_doc(code: str, delim: str):
    string_pre = r"(\S+\s*=\s*)"
    between = rf"({delim})((?:(?!{delim})[\s\S])*)({delim})"

    # this should not hit
    holder = "!|<1multiline1>|!"

    # initially save mutliline strings with flag value
    code = re.sub(
        string_pre + between,
        rf"\1{holder}\3{holder}",
        code,
    )

    # removing docs without caring about mutliline
    code = re.sub(between, "", code)

    # fix flags back to original
    code = re.sub(re.escape(holder), delim, code)
    return code


# Im not sure this was the best approach for doc comments
# Clean inline comments and block comments
def clean_comments(code: str) -> str:
    # remove comments with #
    code = re.sub(r"#[^\n]*", "", code)

    # remove docs with """
    code = clean_doc(code, '"""')

    # remove docs with '''
    code = clean_doc(code, "'''")

    # get rid of trailing
    code = re.sub(r"\s*\n", "\n", code)
    return code


# for lambda deciding what files to include as training data
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

    return bool(re.search("|".join(libraries), content))


# Filters samples to return only code
def keep_only_content(sample: dict) -> dict:
    return {"content": sample["content"]}


# preview cleaning and check
def preview():
    from dataset import CleanDataset

    ds = CleanDataset(
        train_split=False,
        max_size=100
    )

    for x in ds:
        print(x["content"])

def tests():
    input1 = """
        string1 = '''
            keep me
        '''

        '''
        docstring1
        '''

        '''docstring2'''

        '''docstring3
        docstring3'''

        string2 = '''keep me'''
    """
    _ = clean_comments(input1)

    input2 = '''
        string = """
            "hello there" 'hi there'
        """

        """
        1234567890qwertyuiop[!@#$%^&*()].
        """
    '''
    print(clean_comments(input1))
    print(clean_comments(input2))


if __name__ == "__main__":
    tests()

