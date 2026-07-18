# Agent Orchestrator
# A LangGraph agent that takes a user question and decides which tool(s)
# to call. Currently wired with the RAG tool (USDA programs). Vision and
# weather tools will be added the same way once ready.

import sys
import os

# allow importing from the rag/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "vision"))

from predict import predict_disease, format_prediction
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from generate import generate_answer  # reuse your existing RAG generation


# define tools the agent can call

@tool
def usda_program_lookup(query: str) -> str:
    """Look up information about USDA programs, eligibility, disaster
    assistance, crop reporting, loans, or payment rules. Use this when
    the user asks about USDA programs, forms, or farm assistance."""
    result = generate_answer(query)
    sources = ", ".join(result["sources"])
    return f"{result['answer']}\n\n(Sources: {sources})"


# placeholder tools — will be filled in once CV model / weather API are ready
@tool
def disease_detection(image_path: str) -> str:
    """Identify a crop disease from an uploaded image file. Use this when
    the user provides an image of a plant/leaf and wants to know what
    disease it has, or asks 'what's wrong with my plant/leaf'."""
    try:
        results = predict_disease(image_path, top_k=3)
        return format_prediction(results)
    except Exception as e:
        return f"Could not process the image: {str(e)}"


@tool
def weather_lookup(location: str) -> str:
    """Get weather forecast information for a location. NOT YET
    IMPLEMENTED — currently returns a placeholder."""
    return "Weather tool is not yet connected. Coming soon."


tools = [usda_program_lookup, disease_detection, weather_lookup]

# set up the LLM with tool calling

llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {t.name: t for t in tools}


# define the agent state

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# define graph nodes

def call_model(state: AgentState):
    # the planner node: LLM decides whether to respond directly or call a tool
    system = SystemMessage(content=(
        "You are Earthworm AI, a helpful assistant for farmers. "
        "You have tools for USDA program questions, crop disease detection, "
        "and weather. Use a tool when the question needs it. If no tool is "
        "needed, answer directly and briefly."
    ))
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState):
    # the tool-execution node: runs whichever tool(s) the model requested
    #NOTE: To avoid a small local model hallucinating on top of an already
    #grounded RAG answer, when exactly one tool is called we return that
    #tool's output directly as the final answer instead of routing back
    #through the LLM for re-synthesis. This guarantees the final answer
    #stays grounded in retrieved content

    last_message = state["messages"][-1]
    tool_messages = []
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_fn = tools_by_name[tool_name]
        result = tool_fn.invoke(tool_args)
        results.append(result)

        print(f"\n--- RAW TOOL OUTPUT ({tool_name}) ---")
        print(result)
        print("--- END RAW TOOL OUTPUT ---\n")

        tool_messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call["id"]
        })

    if len(results) == 1:
        # single tool call: skip re-synthesis, return grounded result directly
        from langchain_core.messages import AIMessage
        return {"messages": tool_messages + [AIMessage(content=results[0])]}

    # multiple tools called: fall back to letting the LLM combine them
    return {"messages": tool_messages}


def should_continue(state: AgentState):
    # router: decide whether to call a tool or end
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "call_tools"
    return END


def route_after_tools(state: AgentState):
    # after tools run: if we already produced a final AIMessage
    # (single-tool shortcut), end. Otherwise go back to the agent to
    # synthesize a combined answer from multiple tool results

    last_message = state["messages"][-1]
    if last_message.type == "ai":
        return END
    return "agent"


# build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("call_tools", call_tools)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "call_tools": "call_tools",
    END: END
})
graph.add_conditional_edges("call_tools", route_after_tools, {
    "agent": "agent",
    END: END
})

app_graph = graph.compile()


def ask_agent(question: str):
    # run a single question through the agent and return the final answer
    result = app_graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    test_question = "What disease does my plant have? The image is at ../vision/test_leaf.jpg"
    print(f"Question: {test_question}\n")
    answer = ask_agent(test_question)
    print("=== AGENT ANSWER ===")
    print(answer)