from langchain_core.tools import StructuredTool
from pydantic import BaseModel , Field


class MultipleInput(BaseModel) : 
    a : int = Field( description = 'the first number to add')
    b : int = Field( description = 'the second number to add')

def multiple( a, b) :
    """
    multiply the given two numbers 
    """
    return a * b

multiple_tool = StructuredTool.from_function(

    func = multiple,
    name = "multiply",
    description = "multiply two numbers",
    args_schema = MultipleInput
)

# Add this line at the end of your script to test it:
print(multiple_tool.invoke({"a": 5, "b": 4}))