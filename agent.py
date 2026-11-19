"""
Trendora AI agent: LangGraph + Groq.
Keeps content tools, task tools, persistent idea storage, conversation memory,
and interrupt-based approval before deletions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, TypedDict
from urllib.parse import quote

import requests
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
IDEAS_FILE = Path(os.getenv("TRENDORA_IDEAS_FILE", BASE_DIR / "content_ideas.json"))
LANGUAGE_FILE = BASE_DIR / "trendora_language.json"

TRENDORA_PROMPT = """
შენ ხარ Trendora AI — პერსონალური კონტენტისა და მარკეტინგის ასისტენტი.
ეხმარები კონტენტ-კრეატორებსა და მცირე ბიზნესებს:
- კონტენტ იდეების შენახვას, მოძიებასა და ორგანიზებას;
- Instagram პოსტების, Reels-ისა და TikTok სცენარების შექმნას;
- ჰეშთეგებისა და კონტენტ-კალენდრის შედგენას;
- ტრენდებისა და მარკეტინგული სიახლეების მოძიებას;
- კონკურენტებისა და კონტენტის გაანალიზებას;
- გამოქვეყნების დროის შერჩევას;
- ამინდისა და წიგნების მოძიებას;
- დავალებების მართვას.

ყოველთვის უპასუხე მომხმარებლის ენაზე. ნაგულისხმევად გამოიყენე ქართული.
იყავი მეგობრული, კონკრეტული და პროფესიონალური.
არასოდეს წაშალო შენახული იდეა ან დავალება მომხმარებლის მკაფიო დადასტურების გარეშე.
თუ წაშლის Tool შეჩერდა დასადასტურებლად, აუხსენი მომხმარებელს ზუსტად რა იშლება და დაელოდე პასუხს.
შენი სახელია Trendora AI.
"""

class TrendoraState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    language: str
    business_niche: str
    target_audience: str
    content_ideas: list
    selected_platform: str


def _load_ideas() -> list[dict]:
    if IDEAS_FILE.exists():
        try:
            data = json.loads(IDEAS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_ideas(items: list[dict]) -> None:
    IDEAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    IDEAS_FILE.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@tool
def add_idea(idea: str, category: str = "General") -> str:
    """შეინახე ახალი კონტენტ იდეა და მიუთითე მისი კატეგორია."""
    items = _load_ideas()
    if any(x.get("idea", "").casefold() == idea.casefold() for x in items):
        return "ეს იდეა უკვე შენახულია."
    new_id = max((int(x.get("id", 0)) for x in items), default=0) + 1
    items.append({"id": new_id, "idea": idea.strip(), "category": category.strip()})
    _save_ideas(items)
    return f"იდეა წარმატებით შეინახა. ID: {new_id}"


@tool
def list_ideas() -> str:
    """აჩვენე ყველა შენახული კონტენტ იდეა."""
    items = _load_ideas()
    return json.dumps(items, ensure_ascii=False, indent=2) if items else "შენახული იდეები ჯერ არ არის."


@tool
def search_ideas(query: str) -> str:
    """მოძებნე შენახული იდეები სათაურის ან კატეგორიის მიხედვით."""
    q = query.casefold()
    matches = [x for x in _load_ideas()
               if q in x.get("idea", "").casefold() or q in x.get("category", "").casefold()]
    return json.dumps(matches, ensure_ascii=False, indent=2) if matches else "შესაბამისი იდეა ვერ მოიძებნა."


@tool
def delete_idea(idea_id: int) -> str:
    """წაშალე იდეა მხოლოდ მას შემდეგ, რაც მომხმარებელი interrupt approval-ს დაადასტურებს."""
    items = _load_ideas()
    item = next((x for x in items if int(x.get("id", -1)) == int(idea_id)), None)
    if item is None:
        return "მითითებული ID-ის იდეა ვერ მოიძებნა."
    approved = interrupt({
        "action": "delete_idea",
        "idea_id": idea_id,
        "idea": item,
        "message": f"ნამდვილად გსურთ ID {idea_id}-ის იდეის წაშლა: {item.get('idea')}?"
    })
    if approved is True:
        _save_ideas([x for x in items if int(x.get("id", -1)) != int(idea_id)])
        return f"იდეა წაიშალა: {item.get('idea')}"
    return f"წაშლა გაუქმდა. იდეა დარჩა შენახული: {item.get('idea')}"


# Separate in-memory task list, keyed by conversation/thread.
TASKS_STORE: dict[str, list[dict]] = {}
DEFAULT_THREAD_ID = "trendora-default"


def _thread_tasks(thread_id: str = DEFAULT_THREAD_ID) -> list[dict]:
    return TASKS_STORE.setdefault(thread_id, [])


@tool
def add_task(title: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """დაამატე ახალი დავალება კონტენტ-კრეატორის დავალებების სიაში."""
    tasks = _thread_tasks(thread_id)
    next_id = max((int(x["id"]) for x in tasks), default=0) + 1
    tasks.append({"id": next_id, "title": title.strip(), "done": False})
    return f"დავალება დაემატა: #{next_id} — {title.strip()}"


@tool
def list_tasks(thread_id: str = DEFAULT_THREAD_ID) -> str:
    """აჩვენე შენახული დავალებები."""
    tasks = _thread_tasks(thread_id)
    if not tasks:
        return "დავალებების სია ცარიელია."
    return "\n".join(
        f"#{x['id']}. {x['title']} — {'შესრულებული' if x['done'] else 'აქტიური'}"
        for x in tasks
    )


@tool
def delete_task(task_query: str, thread_id: str = DEFAULT_THREAD_ID) -> str:
    """წაშალე დავალება ID-ით ან სათაურით, მხოლოდ მომხმარებლის დადასტურების შემდეგ."""
    tasks = _thread_tasks(thread_id)
    q = task_query.strip().casefold()
    item = next((x for x in tasks if str(x["id"]) == q or q in x["title"].casefold()), None)
    if item is None:
        return "ასეთი დავალება ვერ მოიძებნა."
    approved = interrupt({
        "action": "delete_task",
        "task_id": item["id"],
        "task_title": item["title"],
        "message": f"დაადასტურეთ დავალების წაშლა: #{item['id']} — {item['title']}?"
    })
    if approved is True:
        TASKS_STORE[thread_id] = [x for x in tasks if x["id"] != item["id"]]
        return f"დავალება წაიშალა: {item['title']}"
    return f"წაშლა გაუქმდა. დავალება შენარჩუნებულია: {item['title']}"


@tool
def generate_post(topic: str, audience: str = "ზოგადი აუდიტორია", tone: str = "მეგობრული") -> str:
    """შექმენი Instagram პოსტის ტექსტი მითითებული თემის, აუდიტორიისა და ტონის მიხედვით."""
    return f"""📌 თემა: {topic}
👥 აუდიტორია: {audience}
✍️ Instagram პოსტი:

{topic} — იცოდი, რამდენად მნიშვნელოვანია ეს?
დღეს გაგიზიარებთ რამდენიმე საინტერესო რჩევას, რომლებიც დაგეხმარება სწორი არჩევანის გაკეთებაში.
შეინახე პოსტი და გაუზიარე მეგობრებს! ❤️

#Instagram #Marketing #ContentCreation
ტონი: {tone}"""


@tool
def generate_reels(topic: str, audience: str = "ზოგადი აუდიტორია", duration: str = "30 წამი") -> str:
    """შექმენი Instagram Reels-ის მოკლე სცენარი."""
    return f"""🎬 REELS სცენარი
თემა: {topic}
აუდიტორია: {audience}
ხანგრძლივობა: {duration}

1. Hook: იცოდი, რომ {topic}-თან დაკავშირებით ხშირად შეცდომებს ვუშვებთ?
2. მთავარი ნაწილი: გაუზიარე მაყურებელს 3 მნიშვნელოვანი რჩევა.
3. დასასრული: შეაჯამე მთავარი სათქმელი.
CTA: გამოიწერე გვერდი მეტი საინტერესო რჩევისთვის!"""


@tool
def generate_tiktok(topic: str, audience: str = "ზოგადი აუდიტორია", style: str = "ტრენდული") -> str:
    """შექმენი TikTok ვიდეოს სცენარი."""
    return f"""🎵 TIKTOK სცენარი
თემა: {topic}
აუდიტორია: {audience}
სტილი: {style}

1. Hook: „შენც უშვებ ამ შეცდომას?“
2. მთავარი ნაწილი: გააზიარე 3 საინტერესო ფაქტი ან რჩევა თემაზე — {topic}.
3. დასასრული: „შეინახე ეს ვიდეო, რომ არ დაგავიწყდეს!“
CTA: გამოიწერე მეტი საინტერესო კონტენტისთვის!"""


@tool
def generate_hashtags(topic: str, category: str = "General", platform: str = "Instagram") -> str:
    """შექმენი თემისა და კატეგორიის შესაბამისი ჰეშთეგები."""
    tags = {
        "Skincare": ["#skincare", "#skincareroutine", "#skincaretips", "#glowingskin", "#healthyskin", "#beauty"],
        "Beauty": ["#beauty", "#beautytips", "#makeup", "#selfcare", "#beautylover", "#glowup"],
        "Marketing": ["#marketing", "#digitalmarketing", "#socialmedia", "#contentcreator", "#marketingtips", "#branding"],
        "General": ["#trending", "#contentcreation", "#tips", "#explore", "#foryou", "#socialmedia"],
    }
    return f"თემა: {topic}\nპლატფორმა: {platform}\nჰეშთეგები: {' '.join(tags.get(category, tags['General']))}"


@tool
def generate_content_calendar(niche: str, platform: str = "Instagram") -> str:
    """შექმენი ერთი კვირის კონტენტ-კალენდარი."""
    calendar = {
        "ორშაბათი": "საგანმანათლებლო პოსტი — სასარგებლო რჩევები",
        "სამშაბათი": "Instagram Story — კითხვა აუდიტორიისთვის",
        "ოთხშაბათი": "Reels — საინტერესო ფაქტები",
        "ხუთშაბათი": "საინფორმაციო პოსტი",
        "პარასკევი": "ტრენდული ვიდეო",
        "შაბათი": "Behind the Scenes — კულისების მიღმა",
        "კვირა": "კვირის შეჯამება და აუდიტორიასთან ინტერაქცია",
    }
    return f"📅 {niche} — {platform}\n" + "\n".join(f"{d}: {v}" for d, v in calendar.items())


@tool
def repurpose_content(content: str, platform: str) -> str:
    """მოარგე არსებული კონტენტი სხვადასხვა სოციალურ პლატფორმას."""
    templates = {
        "Instagram": f"📸 INSTAGRAM POST\n\n{content}\n\nშეინახე პოსტი და გაუზიარე მეგობრებს!",
        "TikTok": f"🎵 TIKTOK SCRIPT\n\nHook: იცოდი ეს?\n\n{content}\n\nCTA: გამოიწერე მეტი საინტერესო ვიდეოსთვის!",
        "Facebook": f"📘 FACEBOOK POST\n\n{content}\n\nგაგვიზიარე შენი აზრი კომენტარებში!",
        "LinkedIn": f"💼 LINKEDIN POST\n\n{content}\n\nგაგვიზიარე შენი პროფესიული გამოცდილება ამ თემაზე.",
    }
    return templates.get(platform, "ეს პლატფორმა ჯერ მხარდაჭერილი არ არის.")


@tool
def search_trends(keyword: str, country: str = "Georgia") -> str:
    """მოძებნე ბოლო 7 დღის Google Trends-ის მონაცემები."""
    try:
        from pytrends.request import TrendReq
        client = TrendReq(hl="en-US", tz=240)
        client.build_payload([keyword], timeframe="now 7-d", geo="GE")
        related = client.related_queries()
        interest = client.interest_over_time()
        result = f"🔥 TREND SEARCH\nთემა: {keyword}\nქვეყანა: {country}\n"
        if not interest.empty:
            result += "\nბოლო 7 დღის ინტერესი:\n" + str(interest.tail(5))
        if related and keyword in related:
            rising = related[keyword].get("rising")
            if rising is not None and not rising.empty:
                result += "\nმზარდი ძიებები:\n" + rising.head(5).to_string(index=False)
        return result
    except Exception as exc:
        return f"ტრენდების მოძიება ვერ მოხერხდა: {exc}"


@tool
def search_marketing_news(topic: str = "Digital Marketing") -> str:
    """მოძებნე მარკეტინგის სიახლეები Google News RSS-იდან."""
    try:
        import feedparser
        url = f"https://news.google.com/rss/search?q={quote(topic)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries:
            return "სიახლეები ვერ მოიძებნა."
        return "\n\n".join(
            f"{i}. {a.title}\nთარიღი: {a.get('published', 'უცნობია')}\nბმული: {a.link}"
            for i, a in enumerate(feed.entries[:5], 1)
        )
    except Exception as exc:
        return f"სიახლეების მოძიება ვერ მოხერხდა: {exc}"


@tool
def analyze_competitor(competitor_name: str, followers: int, posts_per_week: int,
                       engagement_rate: float, niche: str) -> str:
    """გააანალიზე კონკურენტი გამომწერების, პოსტების სიხშირისა და ჩართულობის მიხედვით."""
    engagement = ("მაღალი" if engagement_rate >= 5 else
                  "საშუალო" if engagement_rate >= 2 else "დაბალი")
    frequency = "აქტიური" if posts_per_week >= 5 else "გასაზრდელი"
    return (f"კონკურენტი: {competitor_name}\nსფერო: {niche}\nგამომწერები: {followers}\n"
            f"პოსტები კვირაში: {posts_per_week}\nEngagement Rate: {engagement_rate}% ({engagement})\n"
            f"გამოქვეყნების სიხშირე: {frequency}\n\nრეკომენდაცია: შექმენი განსხვავებული კონტენტი, "
            "გამოიყენე Reels და გაზარდე აუდიტორიასთან ინტერაქცია.")


@tool
def suggest_posting_time(platform: str = "Instagram", audience: str = "ზოგადი აუდიტორია") -> str:
    """მიეცი ზოგადი რეკომენდაცია პოსტების გამოქვეყნების დროებზე."""
    schedules = {
        "Instagram": {"ორშაბათი": "12:00–14:00", "სამშაბათი": "11:00–13:00", "ოთხშაბათი": "12:00–14:00",
                      "ხუთშაბათი": "17:00–19:00", "პარასკევი": "11:00–13:00", "შაბათი": "10:00–12:00", "კვირა": "18:00–20:00"},
        "TikTok": {"ორშაბათი": "18:00–21:00", "სამშაბათი": "17:00–20:00", "ოთხშაბათი": "19:00–21:00",
                   "ხუთშაბათი": "18:00–21:00", "პარასკევი": "17:00–22:00", "შაბათი": "12:00–15:00", "კვირა": "18:00–21:00"},
        "Facebook": {"ორშაბათი": "09:00–11:00", "სამშაბათი": "10:00–12:00", "ოთხშაბათი": "09:00–11:00",
                     "ხუთშაბათი": "12:00–14:00", "პარასკევი": "10:00–12:00", "შაბათი": "11:00–13:00", "კვირა": "17:00–19:00"},
    }
    if platform not in schedules:
        return "ეს პლატფორმა ჯერ მხარდაჭერილი არ არის."
    return f"პლატფორმა: {platform}\nაუდიტორია: {audience}\n" + "\n".join(
        f"{d}: {t}" for d, t in schedules[platform].items()
    ) + "\n\nეს ზოგადი საწყისი რეკომენდაციაა — გადაამოწმე შენი აუდიტორიის Insights-ით."


@tool
def analyze_content(content: str, platform: str = "Instagram", goal: str = "Engagement") -> str:
    """შეაფასე სოციალური მედიის ტექსტი და მიეცი გაუმჯობესების რჩევები."""
    score, tips = 0, []
    if len(content) >= 100: score += 25
    else: tips.append("გაზარდე ტექსტის ინფორმაციულობა.")
    if any(w in content.casefold() for w in ["როგორ", "რატომ", "იცოდი", "რჩევა", "შეცდომა"]): score += 25
    else: tips.append("დაიწყე საინტერესო კითხვით ან Hook-ით.")
    if "?" in content: score += 25
    else: tips.append("დაუსვი აუდიტორიას კითხვა.")
    if any(w in content.casefold() for w in ["შეინახე", "გამოიწერე", "გაგვიზიარე", "კომენტარი"]): score += 25
    else: tips.append("დაამატე Call To Action (CTA).")
    if not tips: tips.append("კონტენტი კარგად არის სტრუქტურირებული!")
    return f"📊 შეფასება: {score}/100\nპლატფორმა: {platform}\nმიზანი: {goal}\n\nრეკომენდაციები:\n" + "\n".join(f"- {x}" for x in tips)


@tool
def get_weather(city: str) -> str:
    """მოიძიე მიმდინარე ამინდი Open-Meteo-ს უფასო API-ით."""
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                       params={"name": city, "count": 1, "language": "en"}, timeout=15)
    geo.raise_for_status()
    results = geo.json().get("results", [])
    if not results:
        return f"ქალაქი ვერ მოიძებნა: {city}"
    loc = results[0]
    weather = requests.get("https://api.open-meteo.com/v1/forecast",
                           params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                                   "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"},
                           timeout=15)
    weather.raise_for_status()
    cur = weather.json()["current"]
    return (f"{loc['name']}: {cur['temperature_2m']}°C; იგრძნობა {cur['apparent_temperature']}°C; "
            f"ტენიანობა {cur['relative_humidity_2m']}%; ქარი {cur['wind_speed_10m']} კმ/სთ.")


@tool
def search_books(topic: str, limit: int = 5) -> str:
    """მოძებნე წიგნები Open Library-ის API-ში თემის მიხედვით."""
    response = requests.get("https://openlibrary.org/search.json",
                            params={"q": topic, "limit": limit}, timeout=15)
    response.raise_for_status()
    docs = response.json().get("docs", [])
    if not docs:
        return "წიგნები ვერ მოიძებნა."
    return "\n".join(
        f"{i}. {b.get('title', 'Unknown title')} — {', '.join(b.get('author_name', ['Unknown author']))}"
        for i, b in enumerate(docs, 1)
    )


@tool
def save_language(language: str) -> str:
    """შეინახე Trendora-ს პასუხების ენა — ქართული ან ინგლისური."""
    mapping = {"ქართული": "Georgian", "ინგლისური": "English", "English": "English", "Georgian": "Georgian"}
    selected = mapping.get(language)
    if not selected:
        return "მხარდაჭერილია ქართული და ინგლისური."
    LANGUAGE_FILE.write_text(json.dumps({"language": selected}, ensure_ascii=False), encoding="utf-8")
    return f"ენა შეიცვალა: {selected}"


TOOLS = [
    add_idea, list_ideas, search_ideas, delete_idea,
    add_task, list_tasks, delete_task,
    generate_post, generate_reels, generate_tiktok, generate_hashtags,
    generate_content_calendar, repurpose_content, search_trends,
    search_marketing_news, analyze_competitor, suggest_posting_time,
    analyze_content, get_weather, search_books, save_language,
]


def build_graph():
    """ააგე და დააბრუნე Trendora AI-ის LangGraph გრაფი MemorySaver-ით."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY ვერ მოიძებნა. დაამატე გასაღები .env ფაილში ან გარემოს ცვლადებში."
        )

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.7,
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant_node(state: TrendoraState):
        messages = state.get("messages", [])
        response = llm_with_tools.invoke([SystemMessage(content=TRENDORA_PROMPT), *messages])
        return {"messages": [response]}

    builder = StateGraph(TrendoraState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", lambda state: (
        "tools" if state["messages"][-1].tool_calls else END
    ), {"tools": "tools", END: END})
    builder.add_edge("tools", "assistant")

    # In-memory conversation checkpointing survives Streamlit reruns while
    # this Python process remains alive. Use a database checkpointer for
    # durable memory across restarts/deployments.
    return builder.compile(checkpointer=InMemorySaver())
