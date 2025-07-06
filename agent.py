from agents import Agent, Runner,run,set_default_openai_api,set_tracing_disabled,set_default_openai_client,function_tool,FunctionTool,enable_verbose_stdout_logging
from openai import AsyncOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_google_genai import GoogleGenerativeAI , GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

enable_verbose_stdout_logging()

load_dotenv()
set_tracing_disabled(True)

Google_API_Key = os.getenv("Google_API")

extenal_client = AsyncOpenAI(
    api_key=Google_API_Key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

set_default_openai_client(extenal_client)
set_default_openai_api("chat_completions")

Devops_agent = Agent(
    name="Devops_Assistant",
    instructions=
    """
    You are an devops agent of panacloud company which provides services of
    development of every type and you will give information to user about what is 
    devops and every other query user asks and also remember that you are also the
    part of agentic AI field""",
    model= "gemini-2.0-flash"
)

Openai_agent = Agent(
    name="Openai_Assistant",
    instructions=
    """
    You are an openai info agent of panacloud company which provides services of 
    development of every type and you will give information to user about openai 
    on agentic ai purposes . remember that you are the part of agentic ai field.
    """,
    model="gemini-2.0-flash")

agentic_ai_agent = Agent(
    name="agentic ai assistant",
    instructions=
    """
    you will provide the data on the agentic ai field to user as user asks , 
    remember that you will use your tools as i will provide to you. when user 
    will ask for devops you will use the tool devops assistant, and when user 
    asks about openai you will use tool openai assistant. Do not tell the user 
    that you will use that tool just use the tool without seeking permission of user.
    """,
    model= "gemini-2.0-flash",
    tools= [Devops_agent.as_tool(tool_name="Devops_Assistant", tool_description=
                                 """
    You are an devops agent of panacloud company which provides services of
    development of every type and you will give information to user about what is 
    devops and every other query user asks and also remember that you are also the
    part of agentic AI field"""),
    Openai_agent.as_tool(tool_name="Openai_Assistant", tool_description=
                         """
    You are an openai info agent of panacloud company which provides services of 
    development of every type and you will give information to user about openai 
    on agentic ai purposes . remember that you are the part of agentic ai field.
    """)],
    handoff_description=
    """
    Purpose: This agent is designed to autonomously reason, plan, and execute tasks 
    in AI-driven workflows, often coordinating other agents or tools in the process.

    Handoff Responsibilities:

    Provide execution logs, task chains, or plans followed during autonomous operation.

    Share model parameters, prompts used, memory/context state, and decision criteria.

    Deliver results generated, including structured data, user-facing outputs, or 
    triggered downstream actions.

    Identify any limitations encountered (e.g., tool access, ambiguity in task instructions).

    Suggest follow-up actions or agents to continue/extend the workflow.

    Expected Recipients: Human supervisors, downstream agents, or orchestration frameworks 
    (e.g., LangChain, AutoGen) for review, oversight, or continuation.
    """
)

web_development_agent = Agent(
    name= "Web_developemnt_Assistant",
    instructions=
    """
    You are the agent which will provide the information about the web development 
    to the user as the user asks the question or any type of information.
    """,
    model="gemini-2.0-flash",
    handoff_description=
    """
    Purpose: This agent handles the creation of responsive, SEO-optimized, and scalable 
    websites, including front-end and back-end development.

    Handoff Responsibilities:

    Deliver complete website source code with organized file structure and configuration 
    files.

    Include hosting/deployment guide and DNS configuration details.

    Provide CMS credentials (if applicable), database schemas, and admin access.

    Share responsive and cross-browser compatibility test results.

    Highlight pending content, plugins, or 3rd-party integrations still in progress.

    Expected Recipients: Content team, deployment/hosting engineer, or marketing team for 
    go-live coordination and optimization.
    """
)

Mobile_app_development = Agent(
    name="Mobile_app_development_Assistant",
    instructions=
    """
    You are the Agent which will provide information to the user about the mobile 
    app development as the user asks the question or has any queries about the 
    mobile app development.
    """,
    model="gemini-2.0-flash",
    handoff_description=
    """
    Purpose: This agent specializes in designing, developing, testing, and deploying 
    mobile applications for iOS and Android platforms.

    Handoff Responsibilities:

    Deliver fully functional mobile app source code (React Native / Swift / Kotlin / Flutter, etc.).

    Provide technical documentation including API integrations, architecture overview, and deployment steps.

    Include test cases, QA reports, and user acceptance testing results.

    Submit app store deployment artifacts and metadata (screenshots, app descriptions, etc.).

    Outline remaining issues, known bugs, or pending app store approvals if any.

    Expected Recipients: QA team, DevOps engineer, or product owner for deployment and post-launch support.
    """
)

Panacloud_Agent = Agent(
    name="Panacloud_Agent",
    instructions=
    """
    You are the triage agent which will identify the user's query and handoff that 
    to another agent which you think is made for that specific query and you will 
    also answer general questions. you will transfer the user's query to handoff agent and wont ask user.
    """,
    model="gemini-2.0-flash",
    handoffs=[Mobile_app_development,web_development_agent,agentic_ai_agent,Devops_agent,Openai_agent],
)

query = input("what do you wanna ask\n")

result = Runner.run_sync(
    Panacloud_Agent,
    query,
)

print(result.final_output)
print(result.last_agent)