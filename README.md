# Financial AI Agent

Um agente de IA financeiro que conversa pelo **Telegram** e recomenda a próxima
melhor ação financeira (**Next Best Action — NBA**), construído com **FastAPI**,
**LangChain** e **Pydantic v2**.

## Arquitetura

```
Telegram → FastAPI webhook → validação (secret token + rate limit)
         → AgentExecutor (LangChain, tool-calling)
              ├─ get_customer_profile   → CustomerService      → CustomerRepository
              ├─ get_next_best_action   → NBAService           → NBAModelGateway (mock | SageMaker)
              ├─ get_products           → ProductsService      → CustomerRepository
              └─ search_knowledge_base  → KnowledgeBaseService → KnowledgeBaseGateway (RAG, vector store)
         → resposta em linguagem natural → Telegram
```

Camadas (`src/financial_agent/`):

| Camada | Pasta | Responsabilidade |
|---|---|---|
| Domain | `domain/` | Modelos Pydantic e contrato de erros estruturados. Sem I/O. |
| Repository | `repositories/` | Abstrai onde os dados vivem (hoje: fakes em memória). |
| Gateway | `gateways/` | Integrações externas (Telegram Bot API, modelo NBA, vector store/RAG). |
| Service | `services/` | Orquestração de casos de uso; traduz falhas em `ToolError`. |
| Agent | `agent/` | Prompt, Tools, Output Parser, AgentExecutor (LangChain). |
| Security | `security/` | Validação do webhook Telegram, auth de serviço, rate limit. |
| Observability | `observability/` | Logs estruturados (structlog), correlation id, tracing. |
| API | `api/` | FastAPI: routers, middleware, injeção de dependências. |
| Infra | `infra/aws/` | Adaptador para AWS Lambda (Mangum). |

Cada seta acima é uma dependência em uma única direção (routers → services →
repositories/gateways), nunca o contrário — é isso que permite trocar a
implementação de qualquer camada (ex.: repositório em memória → Postgres, ou
o modelo NBA mockado → SageMaker) sem tocar nas demais.

### Decisões arquiteturais

- **Repository Pattern** para `CustomerRepository`/`ConversationRepository`: hoje
  fakes em memória seedadas; a troca por Postgres/DynamoDB implica apenas uma
  nova classe que satisfaça o mesmo `Protocol`.
- **Gateway Layer** para tudo que é externo (Telegram, modelo NBA): isola
  latência, timeouts e formatos de payload de terceiros do resto do código.
- **Service Layer**: único lugar que decide "isso é um `NOT_FOUND`" ou "isso é
  um `UPSTREAM_ERROR`" — Tools e futuros endpoints REST reaproveitam a mesma
  tradução de erro.
- **Composition Root** (`api/app_state.py`): o único módulo que conhece
  implementações concretas; tudo mais depende de `Protocol`s.
- **Identity-required dispatch**: `user_id` nunca é um campo de Input Schema
  de Tool — é injetado via closure a partir do payload autenticado do
  Telegram (`message.from.id`), nunca do texto da conversa nem de um
  parâmetro que o LLM possa preencher.

## LangChain — componentes utilizados

| Componente | Onde | Papel |
|---|---|---|
| Chat Model | `agent/llm_factory.py` | `ChatOpenAI`, construído uma vez por processo. |
| System Prompt / Human Prompt | `agent/prompts/system_prompt_v2.py` (registrado em `prompt_registry.py`), `agent/agent_executor.py` | Persona, guardrails e compliance versionados como código; `ChatPromptTemplate` compõe system + histórico + mensagem humana. |
| Output Parser | `agent/output_parser.py` | `PydanticOutputParser` estrutura a resposta do "filler agent"; o agente principal usa o parser interno do `create_openai_tools_agent`. |
| Tool Calling | `agent/tools/*.py` | Quatro `StructuredTool`s com schema estreito e identidade vinculada por closure. |
| Runnable | em toda parte | Prompt, LLM, parser e Tools são todos `Runnable`s componíveis com `|`. |
| Agent Executor | `agent/agent_executor.py` | Laço que decide "chamar uma Tool" vs. "responder", com `handle_parsing_errors=True`. |

## Tools — contrato

Cada Tool (`agent/tools/get_*.py`) segue: input schema estreito → identidade
via closure (nunca via LLM) → handler → `run_tool` (`agent/tools/base.py`)
captura qualquer exceção e devolve sempre um `ToolEnvelope` serializado:

```json
{"success": true, "data": {"...": "..."}, "error": null}
{"success": false, "data": null, "error": {"code": "NOT_FOUND", "message": "...", "details": {}}}
```

Códigos de erro: `RATE_LIMITED`, `UPSTREAM_ERROR`, `NOT_FOUND`, `UNAUTHORIZED`,
`VALIDATION_ERROR`, `UNKNOWN_ERROR` — nunca uma exceção crua chega ao laço do
agente.

## RAG — busca na base de conhecimento

Das quatro Tools, `search_knowledge_base` é a única que faz Retrieval-Augmented
Generation de verdade — as outras três fazem *lookup* estruturado (chamadas de
função com schemas fixos), não busca semântica.

```
search_knowledge_base(query)
    → KnowledgeBaseService (valida a query)
        → KnowledgeBaseGateway (Protocol)
            → InMemoryKnowledgeBaseGateway
                → InMemoryVectorStore (langchain_core) + OpenAIEmbeddings
                    → corpus estático de ~8 artigos (gateways/knowledge_base_gateway.py)
```

- O corpus (políticas/como-funciona de CDB, Tesouro Selic, seguros,
  portabilidade de crédito, antecipação de parcelas etc.) é **embedado uma
  única vez no startup** (`InMemoryKnowledgeBaseGateway.build`, chamado em
  `api/app_state.py`), não a cada request.
- `query` é o único campo do Input Schema desta Tool, e é o único caso em que
  deixamos o LLM controlar livremente um parâmetro — porque ele não carrega
  identidade nem filtra dados de outro cliente, só *o que* é buscado.
- O System Prompt v2 instrui o agente a responder **apenas** com base no que
  a ferramenta retornou, nunca preenchendo lacunas com conhecimento do
  próprio modelo — e a nunca usar esta ferramenta no lugar de
  `get_next_best_action` para recomendações personalizadas.
- Troca de backend: `InMemoryVectorStore` é adequado para uma dúzia de
  artigos estáticos; para produção/escala, implemente uma nova classe do
  `KnowledgeBaseGateway` Protocol sobre um vector store real (pgvector,
  Pinecone, OpenSearch, ...) — nenhuma outra camada muda.
- Testes nunca chamam a API de embeddings real: usam
  `DeterministicFakeEmbedding` (`langchain_core`), determinístico e sem rede
  (ver `tests/conftest.py` e `tests/unit/test_knowledge_base_gateway.py`).

## Segurança

- **Webhook Telegram**: autenticado via header `X-Telegram-Bot-Api-Secret-Token`
  (configurado no `setWebhook`), comparado em tempo constante — Telegram não
  assina o payload por padrão, então este é o mecanismo recomendado.
- **user_id**: sempre lido de `message.from.id` (campo autenticado do próprio
  Telegram), nunca do texto da mensagem nem de parâmetros de Tool.
- **Rate limiting**: por `user_id`, antes de qualquer chamada ao LLM (fixed
  window, em memória ou Redis).
- **Auth de serviço**: endpoints internos (fora do webhook) exigem
  `X-Service-Api-Key`.

## Observabilidade

- Logs estruturados (`structlog`), JSON em produção/staging, console legível
  em dev.
- `correlation_id`/`request_id` por requisição (`CorrelationIdMiddleware`),
  mesclados automaticamente em todo log.
- `traced_span` (`observability/tracing.py`) mede e loga a duração de cada
  chamada de Tool; emite spans OpenTelemetry reais se o SDK opcional estiver
  instalado e configurado.
- Tracing de ponta a ponta do LLM via LangSmith (opcional, `LANGCHAIN_TRACING_V2=true`).

## Rodando localmente com Docker Compose

1. Copie o arquivo de exemplo e preencha os segredos:

   ```bash
   cp .env.example .env
   ```

   Preencha `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` (uma string
   aleatória sua escolha), `OPENAI_API_KEY` e `SERVICE_API_KEY`.

2. Suba os serviços:

   ```bash
   docker compose up --build
   ```

   A API sobe em `http://localhost:8000`; `GET /health` deve responder
   `{"status": "ok"}`.

3. Exponha a porta 8000 publicamente (Telegram precisa alcançar um HTTPS
   público) — em desenvolvimento, use um túnel como `ngrok http 8000`.

### Configurar o bot do Telegram

1. Crie o bot com [@BotFather](https://t.me/BotFather) e obtenha o token
   (`TELEGRAM_BOT_TOKEN`).
2. Escolha um segredo aleatório para `TELEGRAM_WEBHOOK_SECRET` (ex.:
   `openssl rand -hex 32`).
3. Registre o webhook, apontando para a URL pública (do túnel/AWS/etc.):

   ```bash
   curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
     -H "Content-Type: application/json" \
     -d '{
           "url": "https://<sua-url-publica>/webhook/telegram",
           "secret_token": "<TELEGRAM_WEBHOOK_SECRET>"
         }'
   ```

4. Converse com o bot no Telegram. Os usuários de exemplo seedados
   (`repositories/customer_repository.py`) são `123` (Ana Souza) e `456`
   (Bruno Lima) — para testar com um `user_id` real do Telegram, adicione seu
   próprio id ao repositório em memória.

## Rodando sem Docker (desenvolvimento)

```bash
poetry install
poetry run pre-commit install
cp .env.example .env  # preencha os valores
poetry run uvicorn financial_agent.main:app --reload
```

## Testes, lint e type-check

```bash
poetry run pytest              # unit + contract + api tests, com cobertura
poetry run ruff check .        # lint
poetry run ruff format .       # formatação
poetry run mypy src            # type-check estrito
poetry run pre-commit run --all-files
```

## Substituindo o modelo mockado por um modelo real

Toda a troca acontece em um único ponto de extensão:
`gateways/nba_model_gateway.py`, atrás do Protocol `NBAModelGateway`
(`async def predict(customer: CustomerProfile) -> NextBestActionCandidate`).

1. Implemente uma nova classe satisfazendo esse Protocol (ex.: chamando um
   endpoint REST, um modelo local, ou o `SageMakerNBAModelGateway` já incluso
   como referência).
2. Registre-a em `api/app_state.py::_build_nba_model_gateway`, condicionada a
   uma nova opção de `Settings.nba_model_provider`.
3. Nenhuma outra camada muda: `NBAService`, a Tool `get_next_best_action`, o
   prompt e os testes de contrato continuam válidos, pois dependem apenas do
   Protocol.

## AWS (SageMaker / Lambda)

- **Modelo real via SageMaker**: implemente/ajuste
  `gateways/nba_model_gateway.SageMakerNBAModelGateway` (já incluso) para o
  formato de payload do seu endpoint, defina
  `NBA_MODEL_PROVIDER=sagemaker`, `SAGEMAKER_ENDPOINT_NAME` e `AWS_REGION`, e
  instale o grupo opcional de dependências: `poetry install --with aws`.
- **API via AWS Lambda**: `infra/aws/lambda_handler.py` expõe `handler`,
  adaptando a mesma app FastAPI via [Mangum](https://mangum.io/) — nenhuma
  rota ou serviço precisa mudar. Empacote como imagem de container (reaproveita
  o `Dockerfile` trocando o `CMD`) ou como zip de dependências, publique atrás
  de API Gateway, e aponte o `setWebhook` do Telegram para a URL do API
  Gateway. Configure as mesmas variáveis de ambiente do `.env.example` como
  variáveis de ambiente da função Lambda.
- Cold starts: mantenha a função aquecida (concorrência provisionada) se a
  latência do webhook for crítica — o Telegram espera resposta em poucos
  segundos.
