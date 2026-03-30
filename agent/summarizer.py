from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    ANTHROPIC_API_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
)

MAX_ARTICLES = 50  # Cap sent to Claude to stay within token limits


def _format_articles(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles[:MAX_ARTICLES], 1):
        lines.append(f"{i}. [{a['source']}] {a['title']}")
        lines.append(f"   URL: {a['url']}")
        if a.get("summary"):
            lines.append(f"   Summary: {a['summary'][:300]}")
        lines.append("")
    return "\n".join(lines)


def _create_llm():
    """Create LLM client — LiteLLM proxy preferred, direct Anthropic as fallback."""
    if LITELLM_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="claude-sonnet",
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            max_tokens=1500,
            model_kwargs={
                "metadata": {
                    "product_area": "apps",
                    "workflow": "ai_news_agent",
                },
            },
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=1500,
    )


def _get_langfuse_handler():
    """Create Langfuse callback handler if configured."""
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            tags=["apps", "ai_news_agent"],
        )
    except Exception:
        return None


def summarize_news(articles: list[dict]) -> str:
    """Use Claude to summarize and curate articles into a Telegram-ready digest."""
    if not articles:
        return "No AI news articles were collected today."

    llm = _create_llm()

    date_str = datetime.now().strftime("%B %d, %Y")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(
                articles_text=_format_articles(articles),
                date=date_str,
            )
        ),
    ]

    kwargs = {}
    langfuse_handler = _get_langfuse_handler()
    if langfuse_handler:
        kwargs["config"] = {"callbacks": [langfuse_handler]}

    response = llm.invoke(messages, **kwargs)
    return response.content
