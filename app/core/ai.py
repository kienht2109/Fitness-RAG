from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings


def create_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
        chunk_size=settings.rag_embedding_batch_size,
    )


def create_chat_model(
    settings: Settings,
    *,
    model: str | None = None,
    temperature: float = 0,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.openai_chat_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=temperature,
    )
