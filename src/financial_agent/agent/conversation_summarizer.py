"""Conversation summarizer — folds old turns into a running summary.

A plain, tool-less LCEL chain (`prompt | llm | StrOutputParser`), the same
shape as `agent/filler_agent.py`. Deliberately built on the Model Router's
utility tier (see `agent/model_router.py`): summarization is mechanical
text compression, not judgment — it doesn't need the reasoning tier's cost
or latency.

Called by `ConversationService._maybe_compact` once the raw message window
grows past its cap; never called directly by a route handler.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from financial_agent.domain.models.conversation import ConversationMessage

_SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você resume conversas entre um cliente e um assistente financeiro. "
            "Produza um resumo curto (no máximo 5 frases) em português, preservando "
            "fatos relevantes sobre o cliente e o que já foi discutido ou recomendado "
            "— produtos mencionados, recomendações dadas, dúvidas já respondidas. "
            "Não invente nada que não esteja no texto abaixo. Responda apenas com o "
            "resumo, sem introduções.",
        ),
        (
            "human",
            "Resumo existente (pode estar vazio):\n{existing_summary}\n\n"
            "Novas mensagens a incorporar:\n{new_turns}\n\n"
            "Escreva o resumo atualizado, combinando o que já existia com as novas "
            "mensagens.",
        ),
    ]
)


class ConversationSummarizer:
    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = _SUMMARIZER_PROMPT | llm | StrOutputParser()

    async def summarize(
        self, *, existing_summary: str | None, new_turns: list[ConversationMessage]
    ) -> str:
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in new_turns)
        result: str = await self._chain.ainvoke(
            {
                "existing_summary": existing_summary or "(nenhum ainda)",
                "new_turns": transcript,
            }
        )
        return result.strip()
