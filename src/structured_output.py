"""
Structured output models using Pydantic.
Demonstrates how to get consistent JSON output from LLMs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class Guest(BaseModel):
    """Represents a single guest."""
    name: str
    category: str = Field(description="family, friend, colleague, other")
    rsvp_status: Optional[str] = Field(default="pending", description="confirmed, pending, declined")


class ScheduleItem(BaseModel):
    """Represents a single activity in the event schedule."""
    time: str = Field(description="Time in HH:MM AM/PM format")
    activity: str
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None


class BudgetItem(BaseModel):
    """Represents a budget category."""
    category: str
    amount: float = Field(description="Amount in Indian Rupees")
    description: Optional[str] = None


class MenuItem(BaseModel):
    """Represents a menu item."""
    name: str
    category: str = Field(description="appetizer, main_course, dessert, beverage")
    quantity: Optional[str] = None
    estimated_cost: Optional[float] = None


class VenueSuggestion(BaseModel):
    """Represents a venue suggestion."""
    name: str
    capacity: int
    estimated_cost: float
    location: str
    features: List[str] = Field(default_factory=list)


class DecorationItem(BaseModel):
    """Represents a decoration item."""
    item: str
    quantity: int
    estimated_cost: float
    priority: str = Field(description="essential or optional")


class ShoppingListItem(BaseModel):
    """Represents an item in the shopping list."""
    item: str
    quantity: str
    estimated_price: Optional[float] = Field(default=0.0, description="Estimated price in Indian Rupees. If unknown, use 0.0")
    priority: str = Field(description="essential or optional")
    category: Optional[str] = None


class EventPlan(BaseModel):
    """Complete structured event plan output."""
    event_type: str
    date: Optional[str] = None
    guest_count: Optional[int] = Field(default=20, description="Number of guests. If not specified, use a reasonable default like 20")
    budget_total: Optional[float] = None
    
    # Structured components
    guests: List[Guest] = Field(default_factory=list)
    schedule: List[ScheduleItem] = Field(default_factory=list)
    budget_breakdown: List[BudgetItem] = Field(default_factory=list)
    menu: List[MenuItem] = Field(default_factory=list)
    venue_suggestions: List[VenueSuggestion] = Field(default_factory=list)
    decoration_plan: List[DecorationItem] = Field(default_factory=list)
    shopping_list: List[ShoppingListItem] = Field(default_factory=list)
    
    # Additional information
    notes: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)


class EventExtraction(BaseModel):
    """Extracted information from user input."""
    event_type: str
    date: Optional[str] = None
    guest_count: Optional[int] = None
    budget: Optional[float] = None
    preferences: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)


class IntentClassification(BaseModel):
    """Intent classification result."""
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


