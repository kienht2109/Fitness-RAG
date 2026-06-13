from app.core.ai import create_chat_model, create_embeddings
from app.core.config import Settings


def test_langchain_provider_factories_use_configured_models() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_chat_model="chat-test",
        openai_embedding_model="embedding-test",
        rag_embedding_batch_size=17,
    )

    embeddings = create_embeddings(settings)
    chat_model = create_chat_model(settings)

    assert embeddings.model == "embedding-test"
    assert embeddings.chunk_size == 17
    assert chat_model.model_name == "chat-test"
