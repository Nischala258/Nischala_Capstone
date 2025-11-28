"""
RAG (Retrieval-Augmented Generation) implementation.
Demonstrates how to use retrieved templates to improve LLM responses.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.vector_store import EventVectorStore
from src.structured_output import EventPlan, EventExtraction
from typing import Dict, Any, List
import json


class EventRAG:
    """RAG system for event planning using retrieved templates."""
    
    def __init__(self, vector_store: EventVectorStore):
        """Initialize RAG with vector store."""
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    def retrieve_context(self, event_type: str, query: str = None) -> str:
        """
        Retrieve relevant templates for context.
        
        Args:
            event_type: Type of event
            query: Additional search query
            
        Returns:
            Formatted context string from retrieved templates
        """
        try:
            templates = self.vector_store.get_relevant_templates(event_type, query)
        except Exception:
            # If search fails, return empty context
            return "No templates found."
        
        if not templates:
            return "No templates found."
        
        context_parts = []
        for i, template in enumerate(templates, 1):
            context_parts.append(
                f"Template {i}:\n"
                f"Description: {template['text']}\n"
                f"Metadata: {json.dumps(template['metadata'], indent=2)}\n"
                f"Relevance Score: {template['score']:.3f}\n"
            )
        
        return "\n".join(context_parts)
    
    def enhance_with_rag(self, user_input: str, event_extraction: EventExtraction) -> Dict[str, Any]:
        """
        Enhance event planning with RAG context.
        
        Args:
            user_input: Original user input
            event_extraction: Extracted event information
            
        Returns:
            Enhanced planning information
        """
        # Retrieve relevant templates
        context = self.retrieve_context(
            event_extraction.event_type,
            user_input
        )
        
        # Load template data
        template_data = self._load_template_data(event_extraction.event_type)
        
        # Create RAG-enhanced prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert event planner. Use the retrieved templates and context to create a comprehensive event plan.
            
Retrieved Templates:
{context}

Template Guidelines:
{template_data}

User Request: {user_input}
Extracted Information: {extraction}

Create a detailed event plan using the templates as guidance, but customize it for the specific user requirements."""),
            ("human", "{user_input}")
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "context": context,
            "template_data": json.dumps(template_data, indent=2),
            "user_input": user_input,
            "extraction": event_extraction.model_dump_json()
        })
        
        return {
            "rag_context": context,
            "enhanced_plan": response.content,
            "templates_used": len(context.split("Template"))
        }
    
    def _load_template_data(self, event_type: str) -> Dict[str, Any]:
        """Load template data from JSON file."""
        try:
            with open("data/templates.json", "r") as f:
                data = json.load(f)
                for template in data["event_templates"]:
                    if template["event_type"] == event_type:
                        return template
        except FileNotFoundError:
            pass
        return {}
    
    def generate_plan_with_rag(
        self,
        user_input: str,
        event_extraction: EventExtraction,
        structured_output_model: type = EventPlan
    ) -> Any:
        """
        Generate structured event plan using RAG.
        
        Args:
            user_input: Original user input
            event_extraction: Extracted event information
            structured_output_model: Pydantic model for structured output
            
        Returns:
            Structured event plan
        """
        # Retrieve context
        context = self.retrieve_context(
            event_extraction.event_type,
            user_input
        )
        
        template_data = self._load_template_data(event_extraction.event_type)
        
        # Create structured output chain with RAG
        # Use a slightly higher temperature for more creative, detailed output
        creative_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
        structured_llm = creative_llm.with_structured_output(structured_output_model)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert event planner. Use the retrieved templates to create a comprehensive, structured event plan.

Retrieved Templates:
{context}

Template Guidelines:
{template_data}

Extracted Information:
{extraction}

CRITICAL REQUIREMENTS - YOU MUST FILL ALL FIELDS:

1. guest_count: ALWAYS provide an integer. If not specified, use a reasonable default (20-50 based on event type).

2. schedule: MUST include at least 5-8 time slots with activities. Include:
   - Welcome/Arrival time
   - Main activities (games, speeches, entertainment)
   - Meal times
   - Key moments (cake cutting, speeches, etc.)
   - Closing/Departure time
   Format: "HH:MM AM/PM" for times

3. budget_breakdown: MUST include at least 5-6 categories with realistic amounts in Indian Rupees:
   - Food & Catering
   - Venue/Rental
   - Decorations
   - Entertainment/Music
   - Photography/Videography
   - Miscellaneous

4. menu: MUST include at least 8-12 items across categories:
   - 2-3 Appetizers
   - 3-4 Main Course items
   - 2-3 Desserts
   - 2-3 Beverages
   Include estimated costs per item.

5. venue_suggestions: MUST provide 3-5 venue options with:
   - Name
   - Capacity (must accommodate guest_count)
   - Estimated cost
   - Location
   - Key features

6. decoration_plan: MUST include 5-8 decoration items with:
   - Item name
   - Quantity needed
   - Estimated cost
   - Priority (essential/optional)

7. shopping_list: MUST include 8-15 items needed for the event:
   - Food items
   - Decoration supplies
   - Party supplies
   - Each with quantity, estimated_price (use realistic prices in ₹), and priority

8. guests: If guest names are mentioned, include them. Otherwise, you can leave empty or create sample guest list.

9. recommendations: MUST include 3-5 helpful tips or suggestions for the event.

DO NOT leave any list empty. Create realistic, detailed plans based on the event type and guest count. Use Indian pricing and cultural context."""),
            ("human", "User Request: {user_input}\n\nGenerate a COMPLETE and DETAILED event plan. Fill ALL fields with realistic, comprehensive information. Do not leave any lists empty.")
        ])
        
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "context": context,
                "template_data": json.dumps(template_data, indent=2),
                "extraction": event_extraction.model_dump_json(),
                "user_input": user_input
            })
        except Exception as e:
            # If structured output parsing fails, create a minimal fallback
            print(f"Warning: Structured output parsing failed: {e}")
            result = None
        
        # Handle case where LLM returns None
        if result is None:
            # Create a minimal EventPlan as fallback
            result = EventPlan(
                event_type=event_extraction.event_type,
                date=event_extraction.date,
                guest_count=event_extraction.guest_count or 20,
                budget_total=event_extraction.budget
            )
        
        # Post-process to ensure guest_count is set from extraction if available
        if result.guest_count is None:
            if event_extraction.guest_count is not None:
                result = result.model_copy(update={"guest_count": event_extraction.guest_count})
            else:
                result = result.model_copy(update={"guest_count": 20})  # Default fallback
        
        # Ensure all shopping list items have estimated_price
        updated_shopping_list = []
        for item in result.shopping_list:
            if item.estimated_price is None:
                updated_item = item.model_copy(update={"estimated_price": 0.0})
                updated_shopping_list.append(updated_item)
            else:
                updated_shopping_list.append(item)
        
        if updated_shopping_list != result.shopping_list:
            result = result.model_copy(update={"shopping_list": updated_shopping_list})
        
        return result

