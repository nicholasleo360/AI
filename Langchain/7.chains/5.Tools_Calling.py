from langchain_core.tools import tool

@tool
def multiple( a, b ) :
    """ multiple two numbers"""
    return a*b


print(multiple.invoke({'a' : 2, 'b' : 2}))