from langchain_core.tools import tool


@tool
def multiple( a, b ) : 
    """
    multiply two numbers 
    """
    return a * b

@tool
def add( a, b ) : 
    """
    add two numbers 
    """
    return a + b

class MathToolKit : 
    name = "math_tool_kit"
    description = "use this to do math operations"
    def get_tools(self) :
        return [multiple, add]


toolkit = MathToolKit()
tools = toolkit.get_tools()

for tool in tools : 
    print(tool.name, "  ", tool.description)