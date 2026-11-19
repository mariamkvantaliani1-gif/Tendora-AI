import os
from typing import Annotated, Any, TypedDict
import requests
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

DEFAULT_THREAD_ID = "creator-session-1"
TASKS_STORE: dict[str, list[dict[str, Any]]] = {}
IDEAS_STORE: dict[str, list[dict[str, Any]]] = {}


def _get_tasks(thread_id: str = DEFAULT_THREAD_ID) -> list[dict[str, Any]]:
    return TASKS_STORE.setdefault(thread_id, [])


def _get_ideas(thread_id: str = DEFAULT_THREAD_ID) -> list[dict[str, Any]]:
    return IDEAS_STORE.setdefault(thread_id, [])


@tool
def add_task(title: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """ამატებს ახალ დავალებას კონტენტ-კრეატორის სიაში."""
    tasks = _get_tasks(thread_id)
    item = {"id": len(tasks) + 1, "title": title.strip(), "done": False}
    tasks.append(item)
    return f"დავალება დაემატა: #{item['id']} — {item['title']}"


@tool
def list_tasks(thread_id: str = DEFAULT_THREAD_ID) -> str:
    """აბრუნებს მიმდინარე დავალებების სიას."""
    tasks = _get_tasks(thread_id)
    if not tasks:
        return "დავალებების სია ცარიელია."
    lines = [
        f"#{t['id']}. {t['title']} ({'შესრულებული' if t['done'] else 'აქტიური'})"
        for t in tasks
    ]
    return "თქვენი დავალებები:\n" + "\n".join(lines)


@tool
def delete_task(task_query: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """შლის დავალებას ID-ით ან სათაურით — წაშლამდე ითხოვს დადასტურებას interrupt()-ით."""
    tasks = _get_tasks(thread_id)
    q = task_query.strip().lower()
    matched = next(
        (t for t in tasks if str(t["id"]) == q or q in t["title"].lower()),
        None,
    )
    if matched is None:
        return f"დავალება '{task_query}' სიაში ვერ მოიძებნა."

    approved = interrupt({
        "action": "delete_task",
        "task_id": matched["id"],
        "task_title": matched["title"],
        "message": f"დაადასტურეთ დავალების წაშლა: #{matched['id']} — '{matched['title']}'?",
    })

    if approved is True:
        TASKS_STORE[thread_id] = [t for t in tasks if t["id"] != matched["id"]]
        return f"დავალება #{matched['id']} ('{matched['title']}') წაიშალა."

    return f"წაშლა გაუქმდა. დავალება #{matched['id']} ('{matched['title']}') უცვლელი დარჩა."


@tool
def add_idea(idea: str, category: str = "general", thread_id: str = DEFAULT_THREAD_ID) -> str:
    """Save a new content idea with its category."""
    ideas = _get_ideas(thread_id)
    item = {
        "id": len(ideas) + 1,
        "idea": idea.strip(),
        "category": category.strip() or "general",
    }
    ideas.append(item)
    return f"იდეა შენახულია: #{item['id']} [{item['category']}] — {item['idea']}"


@tool
def list_ideas(thread_id: str = DEFAULT_THREAD_ID) -> str:
    """Show all saved content ideas."""
    ideas = _get_ideas(thread_id)
    if not ideas:
        return "შენახული კონტენტ-იდეები არ არის."
    return "\n".join(f"#{i['id']} [{i['category']}] — {i['idea']}" for i in ideas)


@tool
def delete_idea(idea_id: int, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """Delete a saved content idea only after explicit user approval."""
    ideas = _get_ideas(thread_id)
    matched = next((i for i in ideas if i["id"] == idea_id), None)
    if matched is None:
        return f"იდეა ID #{idea_id} ვერ მოიძებნა."

    approved = interrupt({
        "action": "delete_idea",
        "idea_id": matched["id"],
        "idea": matched["idea"],
        "category": matched["category"],
        "message": f"დაადასტურეთ იდეის წაშლა: #{matched['id']} [{matched['category']}] — '{matched['idea']}'?",
    })

    if approved is True:
        IDEAS_STORE[thread_id] = [i for i in ideas if i["id"] != matched["id"]]
        return f"იდეა #{matched['id']} ('{matched['idea']}') წაიშალა."

    return f"იდეის წაშლა გაუქმდა. #{matched['id']} ('{matched['idea']}') უცვლელი დარჩა."


@tool
def search_ideas(query: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """Search saved ideas by title or category."""
    ideas = _get_ideas(thread_id)
    q = query.strip().lower()
    matches = [
        i for i in ideas
        if q in i["idea"].lower() or q in i["category"].lower()
    ]
    if not matches:
        return f"'{query}'-ის მიხედვით იდეები ვერ მოიძებნა."
    return "\n".join(f"#{i['id']} [{i['category']}] — {i['idea']}" for i in matches)


@tool
def get_tbilisi_weather() -> str:
    """Get current temperature in Tbilisi, in Celsius."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 41.7151, "longitude": 44.8271, "current_weather": True},
        timeout=10,
    ).json()
    current = resp.get("current_weather", {})
    return f"თბილისში ამჟამად ტემპერატურაა {current.get('temperature')}°C (ქარი: {current.get('windspeed')} კმ/სთ)."


@tool
def search_books(topic: str, limit: int = 5) -> str:
    """Search for books by topic using the Open Library API."""
    resp = requests.get(
        "https://openlibrary.org/search.json",
        params={"q": topic, "limit": limit},
        timeout=10,
    ).json()
    docs = resp.get("docs") or []
    if not docs:
        return f"'{topic}' თემაზე წიგნები ვერ მოიძებნა."
    lines = [
        f"• {d.get('title', 'უცნობი სათაური')} — {', '.join(d.get('author_name', ['უცნობი ავტორი'])[:2])} ({d.get('first_publish_year', '—')})"
        for d in docs[:limit]
    ]
    return "ნაპოვნი წიგნები:\n" + "\n".join(lines)


ALL_TOOLS = [
    add_task,
    list_tasks,
    delete_task,
    add_idea,
    list_ideas,
    delete_idea,
    search_ideas,
    get_tbilisi_weather,
    search_books,
]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "ხარ კონტენტ-კრეატორის დამხმარე AI ასისტენტი. "
        "დავალებებისთვის გამოიყენე add_task, list_tasks, delete_task; "
        "იდეებისთვის — add_idea, list_ideas, delete_idea, search_ideas; "
        "ამინდისთვის — get_tbilisi_weather; წიგნებისთვის — search_books. "
        "უპასუხე მკაფიოდ და ქართულ ენაზე."
    )
)


def build_graph(model_name: str = "llama-3.3-70b-versatile", api_key: str | None = None):
    """აწყობს და აბრუნებს LangGraph აგენტს InMemorySaver checkpointer-ით."""
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model_name, temperature=0.2, api_key=resolved_key)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=InMemorySaver())
