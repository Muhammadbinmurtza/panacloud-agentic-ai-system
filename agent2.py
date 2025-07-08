from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled, enable_verbose_stdout_logging
from openai import AsyncOpenAI
import chainlit as cl
import os
from dotenv import load_dotenv

# Enable detailed logging (optional for debugging)
enable_verbose_stdout_logging()

# Load API keys from .env
load_dotenv()

# Disable OpenAI SDK tracing
set_tracing_disabled(True)

# Set up Google Gemini API via OpenAI-compatible endpoint
google_api_key = os.getenv("Google_API")

external_client = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini-compatible OpenAI proxy endpoint
)

# Set OpenAI-compatible client and chat completion usage
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")

# === Define Agents ===
Devops_agent = Agent(
    name="Devops_Assistant",
    instructions="""
    You are a DevOps agent at Panacloud. Answer all user queries about DevOps and agentic AI clearly and concisely.
    """,
    model="gemini-2.0-flash"
)

Openai_agent = Agent(
    name="Openai_Assistant",
    instructions="""
    You are an OpenAI info agent at Panacloud. Provide information related to OpenAI and agentic AI.
    """,
    model="gemini-2.0-flash"
)

agentic_ai_agent = Agent(
    name="Agentic_AI_Assistant",
    instructions="""
    You answer questions about the Agentic AI field. When a question is about DevOps or OpenAI, silently use the appropriate sub-agent.
    """,
    model="gemini-2.0-flash",
    tools=[
        Devops_agent.as_tool(
            tool_name="Devops_Assistant",
            tool_description="Handles DevOps-related queries."
        ),
        Openai_agent.as_tool(
            tool_name="Openai_Assistant",
            tool_description="Handles OpenAI-related queries."
        )
    ]
)

web_dev_agent = Agent(
    name="Web_Development_Assistant",
    instructions="""
    Answer questions related to web development, both frontend and backend.
    """,
    model="gemini-2.0-flash"
)

mobile_dev_agent = Agent(
    name="Mobile_App_Development_Assistant",
    instructions="""
    Answer mobile app development queries (iOS/Android).
    """,
    model="gemini-2.0-flash"
)

Panacloud_Agent = Agent(
    name="Panacloud_Agent",
    instructions="""
    You are a triage agent. Route the user query to the most appropriate assistant. Never ask the user for permission to delegate.
    """,
    model="gemini-2.0-flash",
    handoffs=[
        mobile_dev_agent,
        web_dev_agent,
        agentic_ai_agent,
        Devops_agent,
        Openai_agent
    ]
)

# === Chainlit Handler (No History) ===
@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(Panacloud_Agent, input=message.content)
    await cl.Message(content=result.final_output).send()
