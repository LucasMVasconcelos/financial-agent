"""AWS Lambda entrypoint, wrapping the same FastAPI app via Mangum.

Because every layer above (services, agent, routers) only depends on
FastAPI/Starlette abstractions — never on anything uvicorn-specific — the
whole application runs unchanged behind API Gateway/Lambda: Mangum adapts
the Lambda event/context into an ASGI call. To deploy this way:

  1. `poetry install --with aws` (adds `mangum`, `boto3`).
  2. Package `src/financial_agent` + dependencies into a Lambda deployment
     artifact (a container image built from this same `Dockerfile` works
     too — Lambda supports container images up to 10GB, just change the
     `CMD` to `financial_agent.infra.aws.lambda_handler.handler`).
  3. Point API Gateway's Telegram-webhook route at the Lambda; set
     `NBA_MODEL_PROVIDER=sagemaker` and `SAGEMAKER_ENDPOINT_NAME` as Lambda
     environment variables to use the real model gateway
     (`gateways/nba_model_gateway.SageMakerNBAModelGateway`) instead of the
     mock.
  4. Register the webhook with Telegram pointing at the API Gateway URL
     (see README "Configurar o bot do Telegram").

Cold starts: LangChain + the OpenAI client add measurable import time.
Keep the function warm (provisioned concurrency, or a scheduled ping) if
webhook latency matters, since Telegram expects a response within a few
seconds.
"""

from __future__ import annotations

from mangum import Mangum

from financial_agent.main import app

handler = Mangum(app, lifespan="on")
