"""Product catalog service.

Merges a static product catalog with the customer's owned products so
`GetProductsTool` can answer "what does the bank offer" and "what do I
already have" in a single call. The catalog would realistically come from a
product/CRM microservice; kept static here since it is out of scope for the
NBA demo.
"""

from __future__ import annotations

from financial_agent.domain.errors import ToolError, ToolErrorCode
from financial_agent.domain.models.customer import Product
from financial_agent.repositories.customer_repository import CustomerRepository

_CATALOG: tuple[Product, ...] = (
    Product(code="conta_corrente", name="Conta Corrente", category="conta"),
    Product(code="cartao_black", name="Cartão Black", category="cartao"),
    Product(code="cartao_gold", name="Cartão Gold", category="cartao"),
    Product(code="cdb_liquidez_diaria", name="CDB Liquidez Diária", category="renda_fixa"),
    Product(code="tesouro_selic", name="Tesouro Selic", category="renda_fixa"),
    Product(code="seguro_vida", name="Seguro de Vida", category="seguro"),
    Product(code="seguro_residencial", name="Seguro Residencial", category="seguro"),
    Product(code="emprestimo_pessoal", name="Empréstimo Pessoal", category="credito"),
)


class ProductsService:
    def __init__(self, customer_repository: CustomerRepository) -> None:
        self._customer_repository = customer_repository

    async def list_products(self, user_id: int) -> list[Product]:
        profile = await self._customer_repository.get_by_id(user_id)
        if profile is None:
            raise ToolError(
                ToolErrorCode.NOT_FOUND,
                "Cannot list products: customer not found.",
                details={"user_id": user_id},
            )
        owned_codes = {p.code for p in profile.products}
        return [
            product.model_copy(update={"owned_by_customer": product.code in owned_codes})
            for product in _CATALOG
        ]
