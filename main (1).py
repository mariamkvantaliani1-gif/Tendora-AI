import os
from dotenv import load_dotenv
from langgraph.types import Command
from agent import DEFAULT_THREAD_ID, IDEAS_STORE, TASKS_STORE, build_graph


def print_new(result: dict, seen: int) -> int:
    msgs = result.get("messages", [])
    for m in msgs[seen:]:
        m.pretty_print()
    return len(msgs)


def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("გაფრთხილება: GROQ_API_KEY არ არის დაყენებული.")

    graph = build_graph()
    config = {"configurable": {"thread_id": DEFAULT_THREAD_ID}}
    TASKS_STORE[DEFAULT_THREAD_ID] = []
    IDEAS_STORE[DEFAULT_THREAD_ID] = []
    seen = 0

    for q in [
        "დაამატე დავალება — ალგებრის სწავლა",
        "მაჩვენე ჩემი დავალებები",
        "თბილისში ახლა რა ტემპერატურაა",
        "მომიძებნე Python-ის წიგნები",
    ]:
        print(f"\n--- {q} ---")
        res = graph.invoke({"messages": [("user", q)]}, config)
        seen = print_new(res, seen)

    print("\n--- წაშლა (True) ---")
    res = graph.invoke({"messages": [("user", "წაშალე დავალება — ალგებრის სწავლა")]}, config)
    seen = print_new(res, seen)
    if res.get("__interrupt__"):
        print("Interrupt:", res["__interrupt__"][0].value)
        res = graph.invoke(Command(resume=True), config)
        seen = print_new(res, seen)

    print("\n--- მეორე დავალება + წაშლა (False) ---")
    res = graph.invoke({"messages": [("user", "დაამატე დავალება — სცენარის წერა")]}, config)
    seen = print_new(res, seen)
    res = graph.invoke({"messages": [("user", "წაშალე დავალება — სცენარის წერა")]}, config)
    seen = print_new(res, seen)
    if res.get("__interrupt__"):
        print("Interrupt:", res["__interrupt__"][0].value)
        res = graph.invoke(Command(resume=False), config)
        seen = print_new(res, seen)

    print("დარჩენილი დავალებები:", TASKS_STORE[DEFAULT_THREAD_ID])


if __name__ == "__main__":
    main()
