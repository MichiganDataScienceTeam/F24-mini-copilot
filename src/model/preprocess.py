import regex as re
from datasets import load_dataset, Dataset, IterableDataset

        
# Clean inline comments and block comments
def clean_comments(code: str) -> str:
    no_comment = re.sub(r'#[^\n]*', '', code)                       # remove comments '# comment'
    no_docstring = re.sub(r'\n\s*"""[^(""")]+"""', '', no_comment)  # remove docstrings '"""docstring"""'
    return re.sub(r'\s*\n', '\n', no_docstring)                     # remove trailing whitespace
