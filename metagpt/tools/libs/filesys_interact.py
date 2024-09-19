from metagpt.tools.tool_registry import register_tool
import os
from metagpt.config2 import config
from metagpt.llm import LLM
from metagpt.utils.text import generate_prompt_chunk
CONSTRAINTS = """
1. Choose descriptive and meaningful names for variables, functions, and classes unless they are in loops.
2. Ensure variable typing is made clear with type hints.
3. Properly document your code with docstrings and comments if its functionality is not already obvious.
4. Avoid excessively using global variables in your code.
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
Below, you are given a summary of the overall purpose of a python file, a snippet from that python file, and a list of programming rules defined by the user.
If the code in the snippet seems to generally follow the provided rules (be lenient), your response should simply be: 
"LGTM".

Otherwise, if a part of the code directly violates one of the given rules, generate ONLY short concise bullet-point comments each structured as:
"
- <description of rule violation>:
`
<specific line(s) that violate rule> 
# suggestion:
<suggested change>
`
"

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
    sys_text = "You are an helpful AI agent who follows user instructions perfectly and precisely."
    with open(path) as file:
        contents = file.read()
    if not os.path.exists(".github/workflows/constraints.txt"):
        constraints = CONSTRAINTS
    else:
        with open(path) as file:
            constraints = file.read()
    prompt_template = COMMENT_PROMPT.format(purpose=purpose, constraints=constraints, content="{}")
    chunks = generate_prompt_chunk(text=contents, prompt_template=prompt_template, model_name="gpt-3.5-turbo-0613", system_text=sys_text)
    comments = ""
    for prompt in chunks:
        item = await llm.aask(msg=prompt, system_msgs=[sys_text])
        comments = f"{comments} {item} \n"
    return comments

    

