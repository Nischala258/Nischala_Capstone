"""
MCP Tools for Event Planning.
Simple tool functions that the LLM can call during planning.
"""

from typing import Dict, List, Any
import json


def add_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an event to the system.
    Simple function to store event information.
    """
    # In a real app, this would save to a database
    # For now, we just return confirmation
    return {
        "status": "success",
        "message": f"Event '{event_data.get('event_type', 'Unknown')}' added successfully",
        "event_id": f"event_{hash(str(event_data)) % 10000}"
    }


def generate_budget(guest_count: int, event_type: str, budget_constraint: float = None) -> Dict[str, Any]:
    """
    Generate a budget breakdown for an event.
    
    Args:
        guest_count: Number of guests
        event_type: Type of event
        budget_constraint: Optional maximum budget
    """
    # Simple per-person cost estimates
    cost_per_person = {
        "birthday_party": 500,
        "corporate_event": 1000,
        "baby_shower": 400,
        "farewell_party": 400,
        "anniversary": 750,
        "wedding": 2000
    }
    
    base_cost = cost_per_person.get(event_type, 600) * guest_count
    
    # If budget constraint is provided, adjust
    if budget_constraint and base_cost > budget_constraint:
        base_cost = budget_constraint
    
    # Simple budget breakdown percentages
    breakdown = {
        "food": round(base_cost * 0.4, 2),
        "venue": round(base_cost * 0.25, 2),
        "decor": round(base_cost * 0.2, 2),
        "entertainment": round(base_cost * 0.1, 2),
        "misc": round(base_cost * 0.05, 2)
    }
    
    return {
        "total_budget": round(base_cost, 2),
        "guest_count": guest_count,
        "breakdown": breakdown
    }


def guest_counter(guest_list: List[str], venue_capacity: int = 50) -> Dict[str, Any]:
    """
    Validate and count guests.
    
    Args:
        guest_list: List of guest names
        venue_capacity: Maximum venue capacity
    """
    count = len(guest_list)
    
    return {
        "guest_count": count,
        "venue_capacity": venue_capacity,
        "within_capacity": count <= venue_capacity,
        "message": f"{count} guests. {'Within' if count <= venue_capacity else 'Exceeds'} capacity of {venue_capacity}."
    }


def menu_price_estimator(menu_items: List[str], guest_count: int) -> Dict[str, Any]:
    """
    Estimate menu prices.
    
    Args:
        menu_items: List of menu items
        guest_count: Number of guests
    """
    # Simple price estimates per item per person
    item_prices = {
        "biryani": 150,
        "butter chicken": 200,
        "paneer tikka": 100,
        "naan": 30,
        "dal makhani": 80,
        "cake": 500,  # per cake
        "soft drinks": 50,
        "juice": 40
    }
    
    total_cost = 0
    item_breakdown = {}
    
    for item in menu_items:
        item_lower = item.lower()
        # Find matching price
        price = 100  # default
        for key, value in item_prices.items():
            if key in item_lower:
                price = value
                break
        
        # Calculate cost (some items are per person, some are fixed)
        if "cake" in item_lower:
            cost = price  # fixed cost
        else:
            cost = price * guest_count
        
        item_breakdown[item] = cost
        total_cost += cost
    
    return {
        "total_estimated_cost": round(total_cost, 2),
        "guest_count": guest_count,
        "item_breakdown": item_breakdown
    }


def shopping_list_generator(event_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a shopping list from event plan.
    
    Args:
        event_plan: Event plan dictionary
    """
    shopping_list = []
    
    # Add food items if menu exists
    if "menu" in event_plan:
        for item in event_plan["menu"]:
            shopping_list.append({
                "item": item.get("name", "Unknown"),
                "category": "food",
                "priority": "essential"
            })
    
    # Add decoration items
    if "decoration_plan" in event_plan:
        for decor in event_plan["decoration_plan"]:
            shopping_list.append({
                "item": decor.get("item", "Unknown"),
                "category": "decoration",
                "priority": decor.get("priority", "optional")
            })
    
    return shopping_list


# Tool registry for LangChain
def get_tools():
    """Return list of tools for LangChain tool calling."""
    from langchain.tools import tool
    
    @tool
    def add_event_tool(event_data: str) -> str:
        """Add an event. Input should be JSON string."""
        data = json.loads(event_data)
        result = add_event(data)
        return json.dumps(result)
    
    @tool
    def generate_budget_tool(guest_count: int, event_type: str, budget_constraint: float = None) -> str:
        """Generate budget breakdown for an event."""
        result = generate_budget(guest_count, event_type, budget_constraint)
        return json.dumps(result)
    
    @tool
    def guest_counter_tool(guest_list: str) -> str:
        """Count and validate guests. Input should be JSON array string."""
        guests = json.loads(guest_list)
        result = guest_counter(guests)
        return json.dumps(result)
    
    @tool
    def menu_price_estimator_tool(menu_items: str, guest_count: int) -> str:
        """Estimate menu prices. Input should be JSON array string."""
        items = json.loads(menu_items)
        result = menu_price_estimator(items, guest_count)
        return json.dumps(result)
    
    return [
        add_event_tool,
        generate_budget_tool,
        guest_counter_tool,
        menu_price_estimator_tool
    ]


