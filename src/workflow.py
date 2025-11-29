"""
LangGraph Workflow - Connects all nodes together.
This is the main workflow that orchestrates the event planning process.
"""

from langgraph.graph import StateGraph, END
from src.state import EventPlanningState
from src.nodes import (
    intent_classification_node,
    event_extraction_node,
    semantic_retrieval_node,
    rag_planning_node,
    budget_tool_node,
    guest_list_tool_node,
    schedule_builder_node,
    structured_output_formatter_node
)
from src.vector_store import EventVectorStore, create_sample_templates
from src.rag import EventRAG


def create_event_planner():
    """
    Create the complete event planning workflow.
    This function sets up and returns the LangGraph workflow.
    """
    # Initialize vector store
    vector_store = EventVectorStore()
    
    # Always add sample templates (they won't duplicate if already exist)
    # In a real app, you'd check if templates exist first
    templates = create_sample_templates()
    try:
        vector_store.add_templates(templates)
    except Exception:
        # Templates might already exist, that's okay
        pass
    
    # Initialize RAG system
    rag_system = EventRAG(vector_store)
    
    # Create the graph
    workflow = StateGraph(EventPlanningState)
    
    # Add all nodes
    workflow.add_node("classify_intent", intent_classification_node)
    workflow.add_node("extract_event", event_extraction_node)
    workflow.add_node("retrieve_templates", 
                     lambda state: semantic_retrieval_node(state, vector_store))
    workflow.add_node("rag_planning", 
                     lambda state: rag_planning_node(state, rag_system))
    workflow.add_node("calculate_budget", budget_tool_node)
    workflow.add_node("validate_guests", guest_list_tool_node)
    workflow.add_node("build_schedule", schedule_builder_node)
    workflow.add_node("format_output", 
                     lambda state: structured_output_formatter_node(state, rag_system))
    
    # Define the workflow edges (how nodes connect)
    workflow.set_entry_point("classify_intent")
    
    workflow.add_edge("classify_intent", "extract_event")
    workflow.add_edge("extract_event", "retrieve_templates")
    workflow.add_edge("retrieve_templates", "rag_planning")
    workflow.add_edge("rag_planning", "calculate_budget")
    workflow.add_edge("calculate_budget", "validate_guests")
    workflow.add_edge("validate_guests", "build_schedule")
    workflow.add_edge("build_schedule", "format_output")
    workflow.add_edge("format_output", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# Simple function to run the planner
def plan_event(user_input: str):
    """
    Simple function to plan an event.
    Just pass in the user's request as a string.
    """
    planner = create_event_planner()
    
    result = planner.invoke({
        "user_input": user_input,
        "messages": []
    })
    
    return result

