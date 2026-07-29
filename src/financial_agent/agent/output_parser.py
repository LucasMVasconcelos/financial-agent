"""Output parsers used by the agent's LangChain Runnables.

Two distinct output-parsing responsibilities exist in this project:

  1. The main conversational agent (`agent/agent_executor.py`) is built
     with `create_openai_tools_agent`, which already wires up LangChain's
     `OpenAIToolsAgentOutputParser` internally — it has to, in order to
     distinguish "call this tool" turns from "final answer" turns. We do
     not reimplement that; `AgentExecutor` owns it.
  2. The "filler" chain (`agent/filler_agent.py`) is a plain LCEL
     `prompt | llm | parser` pipeline with *no* tools, used to keep the
     customer engaged while the main agent's tool-calling turn is still in
     flight. `FillerReplyParser` guarantees its output is a short,
     structured `FillerReply` instead of unbounded free text.
"""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class FillerReply(BaseModel):
    """A short holding message sent while the real recommendation loads."""

    message: str = Field(
        description="A single short, friendly sentence acknowledging the "
        "customer while the recommendation is being prepared. No more than "
        "160 characters. Never mention tools, models, or internal system details."
    )


filler_reply_parser: PydanticOutputParser[FillerReply] = PydanticOutputParser(
    pydantic_object=FillerReply
)
