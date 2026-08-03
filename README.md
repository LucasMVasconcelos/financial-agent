# Financial AI Agent

Um agente de IA financeiro que conversa pelo **Telegram** e recomenda a próxima
melhor ação financeira (**Next Best Action — NBA**), construído com **FastAPI**,
**LangChain** e **Pydantic v2**.

## Arquitetura

> **Nota sobre esta branch (`feature/langgraph`)**: o diagrama e a tabela
> abaixo já refletem o estado desta branch, onde o agente principal também é
> um grafo LangGraph (`agent/main_graph.py`), não mais um `AgentExecutor`.
> Ver "Grafo principal (`feature/langgraph`)" logo abaixo para o porquê e
> para o que muda em relação à `master`.

```
Telegram → FastAPI webhook → validação (secret token + rate limit)
         → classificação de complexidade → Model Router (tier reasoning | utility)
         → MainGraph (LangGraph, FSM cíclica com loop ReAct — agent/main_graph.py)
              reason ⇄ validate_tool_calls → execute_tool ⇄ self_correct
              ├─ get_customer_profile   → CustomerService      → CustomerRepository
              ├─ get_next_best_action   → NBAService           → NBAModelGateway (mock | SageMaker)
              ├─ get_products           → ProductsService      → CustomerRepository
              ├─ search_knowledge_base  → KnowledgeBaseService → KnowledgeBaseGateway (RAG, vector store)
              └─ request_loan           → LoanService          → LoanGraph (LangGraph, checkpointed)
                                                                      ├─ valor ≤ limite → aprova automaticamente
                                                                      └─ valor > limite → pausa (PENDING_APPROVAL)
                                                                            → admin aprova/rejeita → retoma → notifica cliente
         → resposta em linguagem natural → Telegram

  MainGraph e LoanGraph compartilham o mesmo checkpointer de processo
  (MemorySaver local | RedisSaver com USE_REDIS=true — ver seção dedicada).
```

Camadas (`src/financial_agent/`):

| Camada | Pasta | Responsabilidade |
|---|---|---|
| Domain | `domain/` | Modelos Pydantic e contrato de erros estruturados. Sem I/O. |
| Repository | `repositories/` | Abstrai onde os dados vivem (hoje: fakes em memória). |
| Gateway | `gateways/` | Integrações externas (Telegram Bot API, modelo NBA, vector store/RAG). |
| Service | `services/` | Orquestração de casos de uso; traduz falhas em `ToolError`. |
| Agent | `agent/` | Prompt, Tools, Output Parser, o grafo principal (`main_graph.py`, LangGraph) e o fluxo de empréstimo (`loan_graph.py`, LangGraph). |
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

## Grafo principal (`feature/langgraph`) — todo o projeto em LangGraph

Na `master`, só o fluxo de empréstimo era um grafo — o agente conversacional
principal era um `AgentExecutor` do LangChain (loop ReAct opaco: decide, age,
observa, repete, sem estados nem transições visíveis de fora). Esta branch
substitui isso por `agent/main_graph.py`: um `StateGraph` explícito, cíclico,
com cada etapa do loop ReAct como um nó individual, checkpointado e
retomável — atendendo ao pedido de migrar o projeto inteiro para LangGraph,
com FSM, validação determinística entre etapas, retentativa e checkpoint em
Redis.

```
        START
          │
          ▼
    ┌─────────┐   sem tool_calls, ou            ┌──────────┐
 ┌─▶│  reason │──────cap de iterações──────────▶│ finalize │──▶ END
 │  └────┬────┘                                  └──────────┘
 │       │ tool_calls
 │       ▼
 │  ┌──────────────────┐   inválida     ┌───────────────┐
 │  │ validate_tool_    │───────────────▶│ self_correct  │
 │  │ calls             │                └───────┬───────┘
 │  └────────┬──────────┘                        │
 │           │ válida                             │
 │           ▼                                    │
 │     ┌─────────────┐   tool falhou (ToolEnvelope│
 │     │ execute_tool│───success=false)───────────┘
 │     └──────┬──────┘
 │            │ sucesso
 └────────────┘
```

- **`reason`** — o passo ReAct de fato: o LLM (já com `bind_tools`) vê a
  lista corrente de mensagens (prompt de sistema, histórico, resultados de
  tools anteriores) e ou chama uma tool ou produz a resposta final. É
  visitado a cada iteração do loop — é isso que faz o grafo ser cíclico, e
  não um pipeline fixo de passo único.
- **`validate_tool_calls`** — validação determinística *entre* etapas,
  antes de qualquer tool rodar: nome de tool desconhecido, um campo de
  identidade (`user_id`) injetado nos argumentos (defesa em profundidade —
  os Input Schemas já excluem esse campo; isso é uma segunda barreira caso
  um schema futuro regrida), ou um valor de empréstimo fora do intervalo
  permitido. Uma rejeição aqui nunca chega a tocar um service — vai direto
  para `self_correct` com um `ToolEnvelope` de erro sintético, no mesmo
  formato exato de uma falha real de tool, para o LLM não conseguir
  distinguir as duas.
- **`execute_tool`** — roda a(s) tool call(s) já validada(s). Nenhuma
  exceção escapa daqui (contrato de `run_tool`, ver "Tools — contrato"
  abaixo); todo resultado é uma string JSON `ToolEnvelope`.
- **`self_correct`** — **Tool Self-Correction**: ao ver um erro estruturado
  (validação ou `ToolEnvelope.success=False`), injeta uma instrução
  corretiva na conversa e volta para `reason` em vez de travar o turno ou
  desistir. Limitado por `_MAX_TOOL_RETRIES` (2) — depois disso,
  `route_after_execute` para de mandar de volta para `self_correct`, o LLM
  vê a falha mais uma vez e decide como encerrar (normalmente admitindo que
  não conseguiu completar a ação).
- **`finalize`** — extrai o texto final quando `reason` produz uma resposta
  sem tool calls, ou quando o cap de iterações (`_MAX_ITERATIONS = 6`, igual
  ao `max_iterations` do `AgentExecutor` na master) é atingido.

### Human-in-the-loop: por que continua no `LoanGraph`, não no grafo principal

O pedido original passa a leitura de "esta branch usa LangGraph em tudo,
com capacidade de pausar para validação humana" como se fosse uma única
propriedade de um único grafo. Na prática isso vira **dois grafos LangGraph
coordenados**, cada um resolvendo o tipo de pausa que faz sentido para si:

- `main_graph.py` **nunca** pausa um turno de conversa esperando um humano.
  Pausar um turno inteiro por horas seria péssima UX (o cliente ficaria
  vendo "digitando..." indefinidamente) e desnecessário — o checkpoint desse
  grafo só precisa sobreviver a um round-trip do Telegram, não a uma espera
  indeterminada.
- `agent/loan_graph.py` é o lugar certo para essa pausa: `request_loan`
  continua sendo uma Tool comum que retorna na hora (com status
  `pending_approval` quando for o caso), e é o `LoanGraph`, por trás dela,
  que efetivamente pausa em `await_decision` e retoma depois via
  `POST /admin/loans/{id}/decide` — ver "Empréstimos" abaixo.

Os dois grafos agora compartilham o **mesmo checkpointer de processo**
(injetado em `api/app_state.py`, ver próxima seção) — thread_ids nunca
colidem entre eles (`telegram:{update_id}` vs. `application_id`), então
compartilhar a mesma instância é seguro e evita segurar duas conexões Redis
redundantes.

### Persistência de estado no Redis

`Settings.use_redis` (env `USE_REDIS`, mesma flag que já seleciona o backend
do rate limiter) decide o checkpointer:

- `USE_REDIS=false` (padrão, e o que a suíte de testes usa) — `MemorySaver`,
  em processo, perdido ao reiniciar. Suficiente para dev local sem Docker.
- `USE_REDIS=true` — `AsyncRedisSaver` (`langgraph-checkpoint-redis`),
  conectado em `REDIS_URL`, com `asetup()` chamado uma vez no startup
  (`api/app_state.py::_build_checkpointer`). **Requer Redis Stack**
  (`redis/redis-stack-server`, não `redis:7-alpine`) — o checkpointer indexa
  os checkpoints via RediSearch (comandos `FT.*`), que o Redis "puro" não
  tem; `docker-compose.yml` já usa a imagem certa nesta branch. Isso foi
  confirmado tentando `AsyncRedisSaver` contra um Redis comum: falha
  imediatamente com `unknown command 'FT._LIST'`.

Com o checkpoint em Redis, um restart do processo no meio de um turno (entre
`execute_tool` e `reason`, por exemplo) retoma do último nó concluído em vez
de perder o turno — o mesmo raciocínio de recuperação a frio que já valia
para o `LoanGraph` na master, agora estendido ao grafo principal.

## LangChain — componentes utilizados

| Componente | Onde | Papel |
|---|---|---|
| Chat Model | `agent/llm_factory.py` + `agent/model_router.py` | `ChatOpenAI`, construído em duas variantes (reasoning/utility) por processo — ver "Model Router" abaixo. |
| System Prompt / Human Prompt | `agent/prompts/system_prompt_v4.py` (registrado em `prompt_registry.py`), `agent/main_graph.py` | Persona, guardrails e compliance versionados como código; `ChatPromptTemplate` compõe system + histórico + mensagem humana (usado só para *formatar* as mensagens iniciais do grafo, não para rodar um loop). |
| Output Parser | `agent/output_parser.py` | `PydanticOutputParser` estrutura a resposta do "filler agent"; o grafo principal lê `AIMessage.tool_calls` nativamente (`llm.bind_tools`), sem parser separado. |
| Tool Calling | `agent/tools/*.py` | Cinco `StructuredTool`s com schema estreito e identidade vinculada por closure — quatro somente-leitura, uma (`request_loan`) com efeito real. |
| Runnable | em toda parte | Prompt, LLM, parser, Tools e os próprios grafos compilados são todos `Runnable`s componíveis com `|`. |
| LangGraph (`StateGraph` + checkpointer) | `agent/main_graph.py` | Grafo principal: FSM cíclica (`reason ⇄ validate_tool_calls ⇄ execute_tool ⇄ self_correct`) que substitui o `AgentExecutor` — ver "Grafo principal" acima. |
| LangGraph (`StateGraph` + checkpointer) | `agent/loan_graph.py` | Segundo grafo, à parte, estilo Plan-and-Execute, só para originação de empréstimo — ver seção "Empréstimos" abaixo para o porquê de ser um grafo separado do principal. |

## Tools — contrato

Cada Tool (`agent/tools/*.py`) segue: input schema estreito → identidade
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

## Model Router — modelo menor vs. modelo maior, por atividade e por complexidade

`agent/model_router.py` mantém dois clientes `ChatOpenAI` já construídos no
startup e roteia em **dois eixos independentes**:

**Por atividade** (`for_activity`, fixo por call site):

| Atividade | Tier | Por quê |
|---|---|---|
| `AGENT_REASONING` | **reasoning** (`OPENAI_REASONING_MODEL`, padrão `gpt-4o`) | Decide qual tool chamar, segue os guardrails de compliance, escreve a resposta final. |
| `FILLER_REPLY` (mensagem de espera) | **utility** (`OPENAI_UTILITY_MODEL`, padrão `gpt-4o-mini`) | Frase curta, sem tools, sem guardrails de compliance — latência importa mais que qualidade aqui. |
| `SUMMARIZATION` (compressão de histórico) | **utility** | Compressão de texto é tarefa mecânica, não julgamento. |

O filler **só é enviado se o agente principal ainda não respondeu depois de
`FILLER_DELAY_SECONDS`** (padrão 2,5s) — `telegram_webhook.py` roda o agente
como task em background e só dispara o filler se ela não terminou dentro do
prazo. Enviar o filler incondicionalmente faria a mensagem de espera chegar
colada na resposta real em qualquer turno rápido (a maioria), parecendo
duas respostas/spam em vez de uma UX de latência percebida.

**Por complexidade da mensagem** (`for_complexity`, decidido a cada turno pelo
webhook): `agent/query_complexity.py::classify_query_complexity` classifica
cada mensagem como `SIMPLE` ou `COMPLEX` com uma heurística **determinística
e sem chamada de LLM** (tamanho da mensagem, palavras-sinal de comparação/
justificativa, menção a empréstimo, múltiplas perguntas no mesmo texto) —
gastar uma chamada de modelo só para decidir qual modelo usar anularia o
ganho que o roteamento existe para dar. `SIMPLE` usa o tier utility,
`COMPLEX` usa o tier reasoning — é essa escolha, não `AGENT_REASONING` fixo,
que decide o modelo do grafo principal a cada turno (`telegram_webhook.py`).

### ReAct no modelo pequeno, Plan-and-Execute (LangGraph) no fluxo complexo

A combinação intuitiva ("modelo grande = técnica mais sofisticada") não é o
que está implementado, de propósito. Perguntas simples e complexas passam
pelo **mesmo** grafo/loop ReAct (`reason ⇄ validate_tool_calls ⇄
execute_tool`, ver "Grafo principal" acima) — só troca o tamanho do modelo
por baixo, porque decidir "chamar 1 tool ou responder" não muda de forma com
o tamanho da pergunta. Onde a técnica realmente muda é no pedido de
empréstimo: por ser a única ação com efeito real, sempre tratada com o tier
reasoning, e por precisar de um plano revisável com um ponto de pausa para
aprovação humana — é aí que o `LoanGraph` (LangGraph, estilo
Plan-and-Execute, grafo separado do principal) entra, em vez de mais uma
Tool comum dentro do loop ReAct. Ver "Empréstimos" abaixo para o porquê
completo.

## Empréstimos — a única ação com efeito real (LangGraph + human-in-the-loop)

Das cinco Tools, `request_loan` é a única que muda estado de verdade — as
outras quatro são consulta. Por isso ela não é apenas "mais uma tool" que o
`execute_tool` do grafo principal chama e pronto: por trás dela existe um
segundo grafo `LangGraph` dedicado (`agent/loan_graph.py`), porque o
problema que ela resolve — "talvez seja preciso pausar por um tempo
indeterminado esperando um humano decidir, e retomar exatamente de onde
parou, mesmo bem depois da requisição do Telegram já ter terminado" — não é
algo que uma tool comum, executada dentro de um turno de conversa, consegue
expressar (ver "Human-in-the-loop" acima para o porquê de ser um segundo
grafo em vez do mesmo grafo principal).

```
request_loan(amount)
  → LoanService.request_loan → LoanGraph.ainvoke(estado_inicial, thread_id=application_id)

        assess (consulta CustomerService)
            │
        route_by_amount
            ├─ valor ≤ LOAN_HUMAN_APPROVAL_THRESHOLD (padrão R$ 50.000)
            │     → auto_approve → END                                   [status: approved]
            │
            └─ valor > limite
                  → await_decision → END                                 [status: pending_approval, PAUSA aqui]

  (mais tarde, fora de qualquer request do Telegram)
  POST /admin/loans/{id}/decide  (auth: X-Service-Api-Key)
        → LoanService.decide
              → graph.aupdate_state(thread_id, {human_decision: "approved"|"rejected"})
              → graph.ainvoke(None, thread_id)   # retoma do checkpoint
                    → route_by_decision → finalize → END      [status: disbursed | rejected]
        → notifica o cliente via Telegram (best-effort)
```

- **Checkpointer**: o mesmo checkpointer de processo do grafo principal
  (`MemorySaver` por padrão, `RedisSaver` com `USE_REDIS=true` — ver
  "Persistência de estado no Redis" acima); antes desta branch cada grafo
  tinha o seu próprio `MemorySaver` isolado.
- **Por que dois `ainvoke` e não um só bloqueando**: o primeiro roda dentro
  do request do Telegram e precisa retornar rápido — ele nunca espera um
  humano. O segundo acontece minutos, horas ou dias depois, disparado pelo
  endpoint admin, completamente fora do ciclo de vida daquela conversa.
- **`LoanRepository`** (`repositories/loan_repository.py`) existe *ao lado*
  do checkpointer do grafo: o checkpointer é indexado por `thread_id` e não
  foi feito para ser listado ("mostra todos os pedidos pendentes"); o
  repository é a view limpa e consultável que o endpoint admin usa.
- **Guardrail mais importante do prompt v4**: `requires_human_approval=True`
  significa pendente, não "praticamente aprovado" — o agente é instruído a
  nunca afirmar que um empréstimo foi aprovado/desembolsado a menos que o
  campo `status` retornado pela tool diga isso explicitamente.
- **Notificação ao cliente é best-effort**: se o envio pelo Telegram falhar
  depois da decisão humana, a decisão já está persistida — o cliente só não
  recebe o aviso imediato, não fica com um estado inconsistente.

### Testando o fluxo (com os usuários já seedados)

| `user_id` | Valor de teste | Caminho esperado |
|---|---|---|
| `123` (Ana Souza) | `R$ 10.000` | Aprovação automática — `status: approved` |
| `456` (Bruno Lima) | `R$ 15.000` | Aprovação automática — `status: approved` |
| `123` ou `456` | `R$ 80.000` | Pendente — `status: pending_approval`, aparece em `GET /admin/loans` |

Para decidir um pedido pendente (troque `<id>` pelo `application_id`
retornado pela conversa ou por `GET /admin/loans`):

```bash
curl -X POST "http://localhost:8000/admin/loans/<id>/decide" \
  -H "X-Service-Api-Key: <SERVICE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "decided_by": "ana.analista"}'
```

## Memória: curto prazo, resumo e longo prazo

Três mecanismos distintos, com ciclos de vida diferentes (ver
`domain/models/conversation.py` para o porquê de estarem em campos
separados em vez de uma lista única):

1. **Janela crua** (`ConversationHistory.messages`) — as últimas mensagens
   da conversa atual, carregadas via `ConversationRepository`.
2. **Resumo em rolagem** (`ConversationHistory.summary`) — quando a janela
   crua ultrapassa 20 mensagens, `ConversationService._maybe_compact` funde
   as 10 mais antigas num resumo (`agent/conversation_summarizer.py`,
   modelo utility do router) e as remove do armazenamento bruto via
   `ConversationRepository.compact`. Substitui o corte rígido que existia
   antes — nada é simplesmente descartado, é comprimido.
3. **Memória semântica de longo prazo** (`ConversationHistory.long_term_memories`)
   — cada vez que um resumo é gerado, ele também vira uma entrada num
   vector store por usuário (`gateways/semantic_memory_gateway.py`,
   `InMemoryVectorStore` isolado por `user_id` — sem filtro, isolamento
   estrutural). No início de cada turno, `ConversationService.get_history`
   busca semanticamente nessas memórias usando a mensagem atual como query,
   trazendo de volta contexto relevante mesmo que já tenha "rolado" para
   fora da janela crua e do resumo — memória que atravessa conversas, não
   só turnos.

Ambas as pontas de memória de longo prazo (`SemanticMemoryService.remember`
e `.recall`) e a compactação (`ConversationService._maybe_compact`) são
**best-effort**: uma falha no backend de embeddings ou no modelo de resumo é
logada e ignorada — nunca impede a entrega da resposta ao cliente, que já
foi computada nesse ponto do fluxo.

O System Prompt v3 recebe `{conversation_summary}` e `{long_term_memories}`
como variáveis de template e instrui o agente a tratá-los como contexto
aproximado, nunca como fato exato — para saldo, produtos ou recomendação
atuais, o agente sempre confirma via tool, nunca confia só na memória.

## Segurança

- **Webhook Telegram**: autenticado via header `X-Telegram-Bot-Api-Secret-Token`
  (configurado no `setWebhook`), comparado em tempo constante — Telegram não
  assina o payload por padrão, então este é o mecanismo recomendado.
- **user_id**: sempre lido de `message.from.id` (campo autenticado do próprio
  Telegram), nunca do texto da mensagem nem de parâmetros de Tool.
- **Rate limiting**: por `user_id`, antes de qualquer chamada ao LLM (fixed
  window, em memória ou Redis).
- **Auth de serviço**: endpoints internos (fora do webhook) exigem
  `X-Service-Api-Key` — inclui os endpoints admin de empréstimo
  (`api/routers/admin_loans.py`), sem exceção.
- **`request_loan` é a única Tool com efeito real**: todas as outras quatro
  são somente-leitura. O valor (`amount`) é o único parâmetro que o LLM
  controla livremente nela — mesma lógica do `query` em
  `search_knowledge_base`: não carrega identidade, só parametriza o próprio
  pedido do cliente autenticado.

## Observabilidade

- Logs estruturados (`structlog`), JSON em produção/staging, console legível
  em dev.
- `correlation_id`/`request_id` por requisição (`CorrelationIdMiddleware`),
  mesclados automaticamente em todo log.
- `traced_span` (`observability/tracing.py`) mede e loga a duração de cada
  chamada de Tool; emite spans OpenTelemetry reais se o SDK opcional estiver
  instalado e configurado.
- **LangSmith** (`LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`) para
  tracing ponta a ponta do LLM — escolhido em vez de LangFuse porque já
  existe um hook nativo (`agent/llm_factory.py::configure_langsmith_tracing`)
  e por ser o serviço gerenciado da própria LangChain, sem exigir subir
  infraestrutura nova no `docker-compose.yml`. Basta configurar as duas
  variáveis; nenhum código muda. O `LoanGraph` não chama LLM nenhum (é uma
  máquina de estados determinística), então não há nada dele para o
  LangSmith rastrear além do que os logs estruturados já cobrem — só as
  chamadas do grafo principal (`reason`) contra o `ChatOpenAI` aparecem no
  trace.

## Rodando localmente com Docker Compose

1. Copie o arquivo de exemplo e preencha os segredos:

   ```bash
   cp .env.example .env
   ```

   Preencha `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` (uma string
   aleatória sua escolha), `OPENAI_API_KEY`, `SERVICE_API_KEY`,
   `NGROK_AUTHTOKEN` e `NGROK_DOMAIN` (ver "Túnel de desenvolvimento" abaixo).

2. Suba os serviços:

   ```bash
   docker compose up --build
   ```

   A API sobe em `http://localhost:8000`; `GET /health` deve responder
   `{"status": "ok"}`. O serviço `ngrok` do compose já expõe essa porta
   publicamente — não é preciso rodar um túnel à parte.

### Túnel de desenvolvimento (domínio fixo)

Telegram precisa alcançar a API por HTTPS público, e o webhook precisa
sobreviver a restarts. Um túnel efêmero (`ngrok http 8000` avulso, ou
`trycloudflare.com`) sorteia uma **URL nova a cada restart**, o que invalida
silenciosamente o `setWebhook` anterior — o sintoma é "mando mensagem e nada
acontece", com `getWebhookInfo` mostrando `last_error_message` e updates
pendentes. Por isso o `docker-compose.yml` inclui um serviço `ngrok` fixo,
com **domínio estático reservado**, para que a URL nunca mude entre
restarts:

1. Crie uma conta gratuita em [ngrok.com](https://ngrok.com) e pegue o
   authtoken em [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   → `NGROK_AUTHTOKEN`.
2. Reserve um domínio estático gratuito em
   [dashboard.ngrok.com/domains](https://dashboard.ngrok.com/domains) (ex.:
   `seu-nome.ngrok-free.app`) → `NGROK_DOMAIN`.
3. `docker compose up` já sobe o túnel; confira a URL pública ativa (deve
   bater com `NGROK_DOMAIN`) no painel local em `http://localhost:4040`.

Com domínio fixo, o `setWebhook` do próximo passo só precisa ser feito
**uma vez** — reiniciar `docker compose` não quebra mais o webhook.

### Configurar o bot do Telegram

1. Crie o bot com [@BotFather](https://t.me/BotFather) e obtenha o token
   (`TELEGRAM_BOT_TOKEN`).
2. Escolha um segredo aleatório para `TELEGRAM_WEBHOOK_SECRET` (ex.:
   `openssl rand -hex 32`).
3. Registre o webhook, apontando para o domínio fixo do túnel:

   ```bash
   curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
     -H "Content-Type: application/json" \
     -d '{
           "url": "https://<NGROK_DOMAIN>/webhook/telegram",
           "secret_token": "<TELEGRAM_WEBHOOK_SECRET>"
         }'
   ```

   Para depurar (ver a URL atual, erros de entrega, updates pendentes):

   ```bash
   curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getWebhookInfo"
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

## Golden Transcripts — regressão de comportamento do agente

`pytest` sozinho não pega tudo: os testes acima nunca chamam um LLM de
verdade (fakes, `DeterministicFakeEmbedding`, `run_agent_turn` mockado nos
testes de API) — de propósito, pra suíte continuar rápida, determinística e
gratuita de rodar a cada commit. Isso deixa uma lacuna real: nada garante
que uma mudança de prompt, de descrição de Tool, ou de modelo, não quebrou o
*comportamento* do agente — só que o código continua funcionando.
**Golden Transcripts** cobrem essa lacuna: cenários de conversa com
asserções sobre comportamento (quais tools devem ou não ser chamadas, o que
a resposta final deve/não deve conter), pensados pra rodar contra o agente
de verdade.

```
tests/golden/
├── schema.py                     # GoldenTranscript, ExpectedToolCall, ResponseAssertions, rubric (pydantic)
├── judge.py                      # LLM-as-a-judge: avalia o campo rubric de cada cenário
├── transcripts.yaml              # os cenários — 12 exemplos cobrindo as 5 tools + guardrails + segurança
├── test_golden_transcripts_schema.py   # valida o YAML (sem LLM, roda no pytest normal)
└── run_golden_transcripts.py     # executa de verdade contra o agente (precisa de OPENAI_API_KEY, manual)
```

- **`response_assertions` é por substring, nunca por texto exato** — a
  mesma pergunta produz frases diferentes em execuções diferentes, mesmo em
  temperatura baixa. O que precisa ser constante é o *conteúdo*, não a
  redação: por exemplo, `loan_large_amount_requires_human_approval` (o
  cenário mais importante do arquivo) verifica que a resposta nunca contém
  "aprovado"/"desembolsado" quando o empréstimo está pendente de revisão
  humana — é uma regressão de compliance se isso falhar, não só de
  qualidade de texto.
- **`test_golden_transcripts_schema.py` roda no CI normal** (sem rede) e
  pega erros de autoria — nome de tool que não existe mais, id duplicado,
  categoria sem exemplo — antes mesmo de gastar uma chamada de API.
- **Rodar de verdade contra o agente é manual**, por design — custa dinheiro
  e não é determinístico:

  ```bash
  OPENAI_API_KEY=sk-... poetry run python tests/golden/run_golden_transcripts.py
  OPENAI_API_KEY=sk-... poetry run python tests/golden/run_golden_transcripts.py --id loan_large_amount_requires_human_approval
  ```

- **Adicionar um cenário novo**: um item em `transcripts.yaml`, sem tocar
  em código Python — o schema valida automaticamente. Rode `pytest
  tests/golden/` depois pra confirmar que o YAML está bem formado.

### LLM-as-a-judge — a terceira camada de asserção

`expected_tool_calls` e `response_assertions` são rápidos, determinísticos e
gratuitos — mas estruturalmente cegos pra qualidade qualitativa: tom
adequado, se uma explicação faz sentido, se a resposta é *fiel* ao que uma
tool retornou (não só cita a palavra certa). Pra isso existe o campo
opcional `rubric` em `GoldenTranscript`, avaliado por `judge.py`
(`GoldenTranscriptJudge`) — uma segunda chamada de LLM, separada da
conversa em si, que julga a resposta contra **um** critério em texto livre
e devolve `{passed: bool, reasoning: str}` estruturado.

- **Aditivo, não substituto**: as checagens determinísticas continuam
  sendo a primeira linha de defesa, sempre. O judge só entra quando
  `rubric` está definido, como uma segunda opinião pro que substring
  matching não alcança — ex.: em `guardrail_out_of_scope_legal_advice`
  (onde antes a nota dizia "melhor avaliado por leitura humana") e em
  `loan_large_amount_requires_human_approval`, pra pegar o caso em que a
  resposta contorna as palavras proibidas mas ainda assim *implica*
  aprovação.
- **Tier reasoning do Model Router** (`ModelActivity.JUDGE`) — julgar tom e
  fidelidade é julgamento de verdade, não tarefa mecânica; mesmo raciocínio
  de custo/qualidade que rege o resto do roteamento.
- **Escolha deliberada de escopo**: implementei como extensão do
  `run_golden_transcripts.py` existente, não como integração com LangSmith
  Datasets/evaluators — não exige conta/configuração externa pra ser útil
  agora. Migrar os cenários pra um Dataset do LangSmith e rodar via
  `evaluate()` continua um caminho natural depois, se quiser dashboard e
  histórico de tendência em vez do relatório no terminal.

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
