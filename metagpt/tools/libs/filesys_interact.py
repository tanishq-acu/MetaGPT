from metagpt.tools.tool_registry import register_tool
import os
from metagpt.config2 import config
from metagpt.llm import LLM
from metagpt.utils.text import generate_prompt_chunk

CHUNK_SIZE = 2500

CONSTRAINTS = """
1. Choose descriptive and meaningful names for variables, functions, and classes unless they are in loops.
2. Use variable typing and type hints where needed.
3. Document your code with docstrings and comments if its functionality is not already obvious.
"""

INFER_PROMPT = """
##### PREV SUMMARY ABOVE 
Given the above summary of a python program up to this point and the below python code snippet, create a summary that infers the purpose of the program so far. 

Constraints:
1. The summary must be a maximum of 2 sentences.
2. Ensure that your summary incorporates both the content of the python code snippet and the previous summary. 
3. Ensure that you clearly and concisely state what the program/its functions likely do.
4. Reference important functions or lines if needed in your explanation. 
4. Respond only with your summary. 

###### CODE SNIPPET:
{content}
######
"""

COMMENT_PROMPT = """
Below, you are given a summary of the overall purpose of a python file, a snippet from that python file, and a list of programming conventions defined by the user.
If the code in the snippet seems to generally follow the provided conventions (be lenient), your response should simply be: 
"LGTM". Ignore lines or code that is cutoff at the start or end, as this snippet is chosen randomly. 

Otherwise, if a part of the code directly violates one of the given conventions, generate ONLY 1 - 2 short concise bullet-point comments each containing:
<The not satisfied convention>
<specific old line(s) of code> 
<suggested new code>

Remember:
Deliberately ignore all issues or errors if they do not fall under one of the given conventions. 
##### RULES:
{constraints}
#####

##### PURPOSE:
{purpose}
#####
##### SNIPPET:
{content}
#####
"""
ERRORS_PROMPT = """
# Important Instructions: 
Below, you are given a summary of the overall purpose of a python file and a incomplete code fragment from the middle that python file.
Please verify that the code snippet contains no fatal logical errors with regards to the basics of the language.
Deliberately ignore any possible issues with libraries, functions, or function calls that are not in core python or standard lib and assume they are correctly defined and used.
Also deliberately ignore any functions or function signatures you do not recognize.

Since the snippet is from the middle of the python file, ignore any formatting, missing return statements/lines of code, or misspelled/incorrect method names.

Examples: 
- if you have a line with "if" and there is no "else", ignore it as it may be contained in the later lines. 
- if you have a for loop or while loop and you cannot see the entirety of the loop to see if it will execute, ignore it, as it may be just outside of the current fragment. 
- if you are unsure what type of method you are within, assume you are in the method that makes the code work correctly.

If the individual lines of code have no fatal issues, or you have been instructed to ignored the existing fatal issues, respond only with "LGTM".

Otherwise, generate ONLY short concise bullet-point comments each containing:
<Describe the error>
<specific old line(s) of code> 
<suggested new code>

Once again, ignore any issues with spelling, naming, or code conventions.

##### PURPOSE:
{purpose}
#####
##### SNIPPET:
{content}
#####
"""


@register_tool(tags=["list python files", "directory", "folder"])
def ListPythonFiles(directory: str):
    """
    Given a directory, extracts all of the files within that directory that end in .py (python files).

    Args:
        str: A path to a directory. 
    Returns: 
        str: A list of paths to python files in that directory.
    """
    dirs = set()
    if os.path.isdir(directory):
        for dirpath, _, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.endswith(".py"):
                        dirs.add(os.path.join(dirpath, filename))
    else:
        if os.path.exists(directory):
            dirs.add(directory)
    return list(dirs)
@register_tool(tags=["program purpose", "summarize code"])
async def InferProgramPurpose(path: str):
    """
    Given a path to a python file, infers the purpose of the program and what it does. 

    Args:
        str: A path to a python file. 
    Returns: 
        str: A summary/explanation of what that python program does. 
    """
    llm = LLM(llm_config=config.get_openai_llm())
    llm.model = "gpt-4o-mini"
    if not os.path.exists(path):
        return "Given path does not exist!"
    if os.path.isdir(path):
        return "Given path refers to a directory, not a file."
    prompt_template = INFER_PROMPT.format(content="{}")
    sys_text = "You are an AI that is an expert at deducing the purpose of a program. Your sole purpose is to deduce the purpose of a program given some supporting information."
    with open(path) as file:
        contents = file.read()
    chunks = generate_prompt_chunk(contents, prompt_template, "gpt-3.5-turbo-0613", system_text = sys_text)
    prev_summary = "# This is the start of the program."
    for prompt in chunks:
        prompt = prev_summary + prompt
        prev_summary = await llm.aask(msg=prompt, system_msgs=[sys_text])
    return prev_summary
@register_tool(tags=["generate comments", "comment on code"])
async def GenerateComments(path: str, purpose: str):
    """
    Given a path to a python file and a description of the purpose of that python file, generate some comments about what could be improved or fixed. 
    """
    llm = LLM(llm_config=config.get_openai_llm())
    llm.model="gpt-4o-mini"
    if not os.path.exists(path):
        return "Given path does not exist!"
    if os.path.isdir(path):
        return "Given path refers to a directory, not a file!"
    sys_text = "You are an helpful AI agent who follows user instructions precisely. You are a novice to python and therefore tend to assume that code contains no issues. You have no knowledge of programming convention or best practices."
    with open(path) as file:
        contents = file.read()
    if not os.path.exists("/Users/tanishq/Downloads/MetaGPT/metagpt/tools/libs/.github/workflows/constraints.txt"):
        constraints = CONSTRAINTS
    else:
        with open("/Users/tanishq/Downloads/MetaGPT/metagpt/tools/libs/.github/workflows/constraints.txt") as file:
            constraints = file.read()
    lines = contents.splitlines(keepends=True)
    chunks = []
    curr_chunk = ""
    for line in lines:
        while len(line) > CHUNK_SIZE:
            chunks.append(line[:CHUNK_SIZE])
            line = line[CHUNK_SIZE:]
        if len(curr_chunk) + len(line) > CHUNK_SIZE:
            chunks.append(curr_chunk)
            curr_chunk = ""
        curr_chunk += line
    if len(curr_chunk) > 0:
        chunks.append(curr_chunk)
    comments = ""
    for item in chunks:
        prompt = ERRORS_PROMPT.format(purpose=purpose, content=item)
        item = await llm.aask(msg=prompt, system_msgs=[sys_text])
        if item == "LGTM":
            pass
        else:
            comments = f"{comments} {item} \n"
    if comments != "":
        return comments
    else:
        sys_text_v2 = "You are an helpful AI agent who follows user instructions precisely."
        ## run with comment code style/extra constraints prompt
        for item in chunks:
            prompt = COMMENT_PROMPT.format(purpose = purpose, constraints = constraints, content = item)
            item = await llm.aask(msg = prompt, system_msgs = [sys_text_v2])
            if item == "LGTM":
                pass
            else:
                comments = f"{comments} {item} \n"
    if comments == "":
        return "LGTM"
    else:
        return comments