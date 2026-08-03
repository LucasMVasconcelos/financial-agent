"""Domain models."""

from financial_agent.domain.models.conversation import (
    ConversationHistory,
    ConversationMessage,
    MessageRole,
)
from financial_agent.domain.models.customer import CustomerProfile, Product, RiskProfile
from financial_agent.domain.models.knowledge import KnowledgeSnippet
from financial_agent.domain.models.loan import LoanApplication, LoanStatus
from financial_agent.domain.models.nba import NextBestActionCandidate, NextBestActionType

__all__ = [
    "ConversationHistory",
    "ConversationMessage",
    "CustomerProfile",
    "KnowledgeSnippet",
    "LoanApplication",
    "LoanStatus",
    "MessageRole",
    "NextBestActionCandidate",
    "NextBestActionType",
    "Product",
    "RiskProfile",
]
