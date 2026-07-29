# Projeto: Financial AI Agent com LangChain, FastAPI e Telegram

Atue como um **Staff Machine Learning Engineer** e **AI Solutions Architect** especialista em Python, FastAPI, LangChain, engenharia de software, arquitetura de microsserviços e desenvolvimento de agentes de IA para produção.

O objetivo é projetar e implementar uma aplicação completa seguindo boas práticas de arquitetura, segurança, testes, observabilidade e desenvolvimento orientado a contratos.

---

# Objetivo

Desenvolver um **Financial AI Agent** que converse com o usuário pelo **Telegram** e recomende a próxima melhor ação financeira (**Next Best Action - NBA**).

A recomendação deverá ser obtida através de um modelo de Machine Learning.

Como este é um projeto demonstrativo, implemente inicialmente um modelo mockado que retorne respostas simuladas.

Exemplos de recomendações:

* Investir em renda fixa
* Aumentar limite do cartão
* Contratar seguro
* Antecipar parcelas
* Fazer portabilidade
* Aplicar em CDB
* Criar reserva de emergência

O agente deverá explicar a recomendação utilizando linguagem natural e responder dúvidas do usuário.

---

# Stack obrigatória

Utilize obrigatoriamente:

* Python 3.12+
* FastAPI
* LangChain
* Pydantic v2
* Docker
* Docker Compose
* Poetry (preferencialmente)
* Pytest
* HTTPX
* Ruff
* Mypy
* Pre-commit

Opcionalmente utilize:

* Redis para cache
* LangSmith para tracing
* OpenTelemetry
* Structlog para logs estruturados

---

# Arquitetura

A arquitetura deve seguir princípios de:

* Clean Architecture
* SOLID
* Domain Driven Design (DDD) quando fizer sentido
* Dependency Injection
* Separation of Concerns
* Repository Pattern (quando necessário)
* Service Layer
* Gateway Layer
* Configuration via Environment Variables

Explique todas as decisões arquiteturais.

---

# Fluxo da aplicação

1. Usuário envia mensagem no Telegram.

2. Telegram chama um webhook FastAPI.

3. Um Gateway valida:

* assinatura
* user_id
* autenticação
* autorização

4. O request é convertido para um modelo Pydantic.

5. O agente identifica o usuário. Chama um banco de dados com conversas antigas, produtos que o user já possui etc. 

6. O agente consulta uma Tool responsável pelo modelo NBA.

7. O modelo retorna uma recomendação mockada.

8. O agente utiliza o resultado para responder ao usuário em linguagem natural.

9. A resposta é enviada ao Telegram.

Enquanto o processo aguarda a resposta do NBA, deixe um agente para ir conversando com o usuário.
---

# Modelo de Machine Learning

Crie um serviço mockado.

Exemplo:

```
Input

user_id = 123

Output

{
    "action": "Investir em CDB",
    "confidence": 0.92,
    "reason": "Cliente possui saldo elevado parado em conta corrente."
}
```

Estruture o código para permitir substituir facilmente esse mock por um modelo real posteriormente.

---

# LangChain

Utilize LangChain para construir o agente.

Implemente:

* Chat Model
* PromptTemplate
* System Prompt
* Human Prompt
* Output Parser
* Tool Calling
* Runnable
* Agent Executor

Explique cada componente utilizado.

---

# Implementação das Tools

Implemente Tools seguindo boas práticas de engenharia.

Cada Tool deve possuir:

* responsabilidade única
* entrada mínima
* saída fortemente tipada
* schemas Pydantic
* documentação
* validação

Nunca permita que o LLM envie informações de identidade.

O user_id deve ser sempre obtido do contexto autenticado da requisição.

---

# Tool Design for LLM Consumption

Implemente ferramentas seguindo estas regras:

* Narrow input schemas
* Narrow output schemas
* Identity-required dispatch
* Scope-gated execution
* Strong typing
* Pydantic validation
* Structured responses

Cada Tool deverá representar apenas uma responsabilidade.

Exemplos:

* GetNextBestActionTool
* GetCustomerProfileTool
* GetProductsTool

---

# Contrato das Tools

Cada Tool deve possuir:

Input Schema

Output Schema

Structured Errors

Utilize erros estruturados como:

```
RATE_LIMITED

UPSTREAM_ERROR

NOT_FOUND

UNAUTHORIZED

VALIDATION_ERROR

UNKNOWN_ERROR
```

Esses erros devem ser tratados pelo agente.

Nunca permitir que uma exceção interrompa a conversa.

---

# Implementação dos handlers

Os handlers das Tools devem:

* obter user_id do contexto
* nunca confiar em parâmetros enviados pelo LLM
* validar entrada
* validar saída
* converter exceções para Structured Errors
* registrar logs

---

# Testes

Implemente testes completos.

Inclua:

## Unit Tests

* Happy Path
* RATE_LIMITED
* NOT_FOUND
* UPSTREAM_ERROR
* INVALID_INPUT

## Tool Contract Tests

Valide:

* Input Schema
* Output Schema
* Structured Errors
* Serialization

## API Tests

Teste:

* webhook Telegram
* autenticação
* autorização
* payload inválido
* payload válido

---

# Prompt Engineering

O System Prompt deve ser tratado como código.

Implemente:

* versionamento
* comentários
* documentação
* separação entre System Prompt e Human Prompt

Explique:

* persona
* audience
* tone
* compliance
* guardrails
* tool descriptions
* template variables
* allowed tools

O Prompt deve ser facilmente evolutivo.

---

# Segurança

Implemente:

* validação do Telegram
* autenticação
* autorização
* request validation
* rate limiting
* validação do user_id

Nunca aceite identidade enviada pelo modelo.

---

# Observabilidade

Adicionar:

* logs estruturados
* correlation id
* request id
* tracing
* tempo de execução das Tools
* métricas

---

# Docker

Criar:

* Dockerfile otimizado
* Docker Compose
* .dockerignore

Utilizar boas práticas:

* multi-stage build (se possível)
* imagem pequena
* cache eficiente

---

# Estrutura do projeto

Utilize a melhor organização:


---

# Entrega esperada

Gere o projeto completo.

Para cada arquivo:

* informe o caminho;
* explique sua responsabilidade;
* apresente o código completo;
* utilize tipagem estática;
* documente as funções;
* siga PEP 8;
* utilize Pydantic v2;
* utilize boas práticas de engenharia de software.

Ao final, explique como executar o projeto localmente usando Docker Compose, como configurar o bot do Telegram e como substituir posteriormente o modelo mockado por um modelo real de Next Best Action.

Essa versão aumenta bastante a qualidade da resposta porque:

* transforma requisitos em uma especificação técnica organizada;
* define claramente o papel que o modelo deve assumir (Staff ML Engineer/AI Architect);
* separa arquitetura, segurança, LangChain, Tools, testes e infraestrutura;
* inclui requisitos de qualidade (SOLID, Clean Architecture, tipagem, observabilidade);
* deixa explícito o formato da entrega, reduzindo respostas incompletas.

# AWS
Se possível adapte essa solução para usar o AWS SageMaker ou AWS Lambda.