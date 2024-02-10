from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key


import os
os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature = 0.6)
name = llm("Suggest top 5 popular names of American baby girls.")
print(name)

def generate_baby_names(gender: str,nationality:str) -> list[str]:
    """
    Generate a list of 5 baby names

    Parameters:
    gender (str): gender of baby
    nationailty (str) : nationailty of baby

    Returns:
    list: list of baby names
    """
    prompt_template_name = PromptTemplate(
        input_variables=['gender', 'nationality'],
        template="""I want to find a name for a {nationality} {gender} baby. Suggest top 5 popular names for the baby.
                   Return it as a comma separated list """)
    name_chain = LLMChain(llm=llm,
                          prompt=prompt_template_name,
                          output_key='baby_names')

    chain = SequentialChain(
        chains=[name_chain],
        input_variables=['gender', 'nationality'],
        output_variables=['baby_names']
    )

    response = chain({'gender': gender,
                      'nationality': nationality})
    return response

if __name__ == "__main__":
    print(generate_baby_names("Indian"))
