"""GetProductsTool — the catalog, cross-referenced with what the customer owns.

Unlike the other two tools, this one has a genuinely useful LLM-controlled
input: an optional `category` filter. That field is safe to expose because
it carries no identity information and only narrows a read — the
`user_id` used to compute `owned_by_customer` is still bound via closure,
never supplied by the model.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from financial_agent.agent.tools.base import run_tool
from financial_agent.services.products_service import ProductsService

GET_PRODUCTS_TOOL_NAME = "get_products"


class GetProductsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description=(
            "Optional category filter, e.g. 'renda_fixa', 'seguro', 'cartao', "
            "'credito', 'conta'. Omit to list all products."
        ),
    )


class ProductItem(BaseModel):
    code: str
    name: str
    category: str
    owned_by_customer: bool


class GetProductsOutput(BaseModel):
    products: list[ProductItem]


def build_get_products_tool(*, user_id: int, products_service: ProductsService) -> StructuredTool:
    async def _handler(category: str | None = None) -> str:
        async def _call() -> GetProductsOutput:
            products = await products_service.list_products(user_id)
            if category:
                products = [p for p in products if p.category == category]
            return GetProductsOutput(
                products=[
                    ProductItem(
                        code=p.code,
                        name=p.name,
                        category=p.category,
                        owned_by_customer=p.owned_by_customer,
                    )
                    for p in products
                ]
            )

        return await run_tool(tool_name=GET_PRODUCTS_TOOL_NAME, user_id=user_id, handler=_call)

    return StructuredTool.from_function(
        name=GET_PRODUCTS_TOOL_NAME,
        description=(
            "Lists the bank's products, optionally filtered by category, showing "
            "which ones the current authenticated customer already owns. Use this "
            "to check whether a recommended product is already owned before "
            "suggesting it."
        ),
        args_schema=GetProductsInput,
        coroutine=_handler,
    )
