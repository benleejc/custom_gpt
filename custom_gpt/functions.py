from config import AppConfig
import datetime as dt
import argparse
import string
# from dotenv import load_dotenv # if using .env files; pip install python-dotenv

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI, human
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.schema.messages import HumanMessage, SystemMessage


def basic_llm(template=None, params={}, **kwargs):
    """Basic llm call
    """
    llm = OpenAI(**kwargs)
    # prompt
    if not template or not params:
        raise Exception('Must declare template and/or params')
    prompt = PromptTemplate.from_template(template)

    # chaining inputs & prompt templates
    chain = LLMChain(llm=llm, prompt=prompt)
    with get_openai_callback() as cb:
        res = chain.run(params)
    return res, cb


def basic_chat_model(system_template, human_template, params):
    chat = ChatOpenAI()
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt,
                                                    human_prompt])

    with get_openai_callback() as cb:
        res = chat(chat_prompt.format_prompt(**params).to_messages())
    return res, cb


# example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", help="use llm or chat model")
    parser.add_argument("-t", "--template", help="template")
    parser.add_argument("-i", "--inputs", action='store_true',
                        help="inputs for template")
    parser.add_argument("-st", "--sys_template",
                        help="system template for chat model")

    args = parser.parse_args()

    print("running basic llm demo")
    conf = AppConfig()
    params = {}

    if args.template:
        template = args.template
        template_args = [i[1] for i in string.Formatter().parse(args.template) 
                         if i[1] is not None]
        for i in template_args:
            params[i] = input(f"input {i}: ")
    else:
        params = dict(x='declare a variable',y='python')
        template = "How to {x} in {y}"
    if not args.sys_template and args.model_type == 'chat':
        system_template = "You're a helpful programming assistant"
    else:
        system_template = args.sys_template
    if args.model_type == 'llm':
        print(template.format(**params))
        res, cb = basic_llm(template, params, temperature=0.3)
    elif args.model_type == 'chat':
        res, cb = basic_chat_model(system_template=system_template,
                                   human_template=template, params=params)
    print(res)
    print()
    print(cb)
