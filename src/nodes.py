"""
LangGraph Nodes - Individual Steps in the Workflow.
Each node is a simple function that processes the state.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.state import EventPlanningState
from src.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    EVENT_EXTRACTION_PROMPT
)
from src.structured_output import EventExtraction, IntentClassification
from src.vector_store import EventVectorStore
from src.rag import EventRAG
from src.tools import generate_budget, guest_counter
from typing import Dict


def intent_classification_node(state: EventPlanningState) -> Dict:
    """
    Node 1: Classify user intent.
    Simple classification to understand what type of event.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
    chain = prompt | llm
    
    response = chain.invoke({"user_input": state["user_input"]})
    intent = response.content.strip().lower()
    
    # Add message
    messages = state.get("messages", [])
    messages.append(f"Classified intent as: {intent}")
    
    return {
        "intent": intent,
        "messages": messages
    }


def event_extraction_node(state: EventPlanningState) -> Dict:
    """
    Node 2: Extract structured information from user input.
    Uses structured output to get clean data.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EventExtraction)
    
    prompt = ChatPromptTemplate.from_template(EVENT_EXTRACTION_PROMPT)
    chain = prompt | structured_llm
    
    extraction = chain.invoke({"user_input": state["user_input"]})
    
    messages = state.get("messages", [])
    messages.append(f"Extracted event: {extraction.event_type}, Guests: {extraction.guest_count}")
    
    return {
        "event_extraction": extraction,
        "messages": messages
    }


def semantic_retrieval_node(state: EventPlanningState, vector_store: EventVectorStore) -> Dict:
    """
    Node 3: Retrieve similar event templates using semantic search.
    Uses vector store to find relevant examples.
    """
    extraction = state.get("event_extraction")
    if not extraction:
        return {"retrieved_templates": [], "messages": state.get("messages", [])}
    
    # Search for similar templates
    query = f"{extraction.event_type} {state['user_input']}"
    templates = vector_store.search(query, k=3)
    
    messages = state.get("messages", [])
    messages.append(f"Retrieved {len(templates)} similar event templates")
    
    return {
        "retrieved_templates": templates,
        "messages": messages
    }


def rag_planning_node(state: EventPlanningState, rag_system: EventRAG) -> Dict:
    """
    Node 4: Use RAG to enhance the planning.
    Combines retrieved templates with LLM to create better plan.
    """
    extraction = state.get("event_extraction")
    if not extraction:
        return {"rag_enhanced_plan": {}, "messages": state.get("messages", [])}
    
    # Use RAG to generate enhanced plan
    enhanced = rag_system.enhance_with_rag(state["user_input"], extraction)
    
    messages = state.get("messages", [])
    messages.append("Generated RAG-enhanced plan using retrieved templates")
    
    return {
        "rag_enhanced_plan": enhanced,
        "messages": messages
    }


def budget_tool_node(state: EventPlanningState) -> Dict:
    """
    Node 5: Calculate budget using tool.
    Demonstrates tool calling in the workflow.
    """
    extraction = state.get("event_extraction")
    if not extraction:
        return {"budget_result": {}, "messages": state.get("messages", [])}
    
    # Call budget tool
    budget = generate_budget(
        guest_count=extraction.guest_count or 20,
        event_type=extraction.event_type,
        budget_constraint=extraction.budget
    )
    
    messages = state.get("messages", [])
    messages.append(f"Calculated budget: ₹{budget['total_budget']}")
    
    return {
        "budget_result": budget,
        "messages": messages
    }


def guest_list_tool_node(state: EventPlanningState) -> Dict:
    """
    Node 6: Validate guest list using tool.
    Another example of tool calling.
    """
    extraction = state.get("event_extraction")
    if not extraction:
        return {"guest_list_result": {}, "messages": state.get("messages", [])}
    
    # For demo, create a sample guest list
    # In real app, this would come from extraction or user input
    sample_guests = [f"Guest {i+1}" for i in range(extraction.guest_count or 20)]
    
    result = guest_counter(sample_guests, venue_capacity=50)
    
    messages = state.get("messages", [])
    messages.append(result["message"])
    
    return {
        "guest_list_result": result,
        "messages": messages
    }


def schedule_builder_node(state: EventPlanningState) -> Dict:
    """
    Node 7: Build the event schedule.
    Creates timeline based on event type and details.
    """
    extraction = state.get("event_extraction")
    if not extraction:
        return {"messages": state.get("messages", [])}
    
    # Simple schedule based on event type
    schedules = {
        "birthday_party": [
            {"time": "6:00 PM", "activity": "Welcome & Greetings"},
            {"time": "6:30 PM", "activity": "Games/Entertainment"},
            {"time": "7:00 PM", "activity": "Dinner"},
            {"time": "8:00 PM", "activity": "Cake Cutting"},
            {"time": "8:30 PM", "activity": "Music & Dancing"}
        ],
        "corporate_event": [
            {"time": "7:00 PM", "activity": "Cocktails & Networking"},
            {"time": "8:00 PM", "activity": "Welcome Address"},
            {"time": "8:30 PM", "activity": "Dinner"},
            {"time": "9:30 PM", "activity": "Speeches"},
            {"time": "10:00 PM", "activity": "Networking"}
        ]
    }
    
    schedule = schedules.get(extraction.event_type, [
        {"time": "6:00 PM", "activity": "Event Start"},
        {"time": "8:00 PM", "activity": "Main Activity"},
        {"time": "10:00 PM", "activity": "Event End"}
    ])
    
    messages = state.get("messages", [])
    messages.append(f"Created schedule with {len(schedule)} activities")
    
    return {
        "schedule": schedule,
        "messages": messages
    }


def structured_output_formatter_node(state: EventPlanningState, rag_system: EventRAG) -> Dict:
    """
    Node 8: Format final output as structured JSON.
    Uses RAG to generate complete structured plan.
    """
    extraction = state.get("event_extraction")
    if not extraction:
        return {"final_plan": None, "messages": state.get("messages", [])}
    
    # Generate final structured plan using RAG
    final_plan = rag_system.generate_plan_with_rag(
        state["user_input"],
        extraction
    )
    
    messages = state.get("messages", [])
    messages.append("Generated final structured event plan")
    
    return {
        "final_plan": final_plan,
        "messages": messages
    }

