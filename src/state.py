"""
LangGraph State Definition.
Simple state that holds all information as the workflow progresses.
"""

from typing import TypedDict, List, Dict, Any, Optional
from src.structured_output import EventExtraction, EventPlan


class EventPlanningState(TypedDict):
    """
    State that flows through the LangGraph workflow.
    Each node can read and update this state.
    """
    # User input
    user_input: str
    
    # Intent classification
    intent: Optional[str]
    
    # Extracted event information
    event_extraction: Optional[EventExtraction]
    
    # Retrieved templates from vector store
    retrieved_templates: Optional[List[Dict[str, Any]]]
    
    # RAG-enhanced planning
    rag_enhanced_plan: Optional[Dict[str, Any]]
    
    # Tool results
    budget_result: Optional[Dict[str, Any]]
    guest_list_result: Optional[Dict[str, Any]]
    menu_result: Optional[Dict[str, Any]]
    
    # Final structured output
    final_plan: Optional[EventPlan]
    
    # Any errors or messages
    messages: List[str]


