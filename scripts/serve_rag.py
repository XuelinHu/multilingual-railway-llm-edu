from __future__ import annotations

import argparse

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from railway_llm_edu.rag.generator import RagGenerator


class QueryRequest(BaseModel):
    question: str


def create_app(config_path: str) -> FastAPI:
    app = FastAPI(title="multilingual-railway-llm-edu RAG API")
    generator = RagGenerator(config_path)

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        return generator.answer(req.question)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
