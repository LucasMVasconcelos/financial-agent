"""Output parsers used by the agent's LangChain Runnables.

Two distinct output-parsing responsibilities exist in this project:

  1. The main conversational agent (`agent/main_graph.py`) calls
     `llm.bind_tools(tools)` directly and reads the native
     `AIMessage.tool_calls` field the `reason` node's routing function
     checks — no separate parser needed, since tool-calling models already
     return that distinction structurally rather than as text to parse.
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
