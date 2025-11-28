"""
Prompt templates for the AI Event Planner.
Demonstrates various prompting techniques for event planning tasks.
"""

# Intent Classification Prompt
INTENT_CLASSIFICATION_PROMPT = """You are an AI event planning assistant. Analyze the user's input and classify their intent.

User Input: {user_input}

Classify the intent into one of these categories:
- birthday_party
- corporate_event
- wedding
- baby_shower
- farewell_party
- anniversary
- other

Respond with only the category name."""

# Event Extraction Prompt
EVENT_EXTRACTION_PROMPT = """Extract structured information from the user's event planning request.

User Input: {user_input}

Extract the following information:
1. Event type (e.g., birthday party, corporate dinner)
2. Date (if mentioned)
3. Number of guests (if mentioned)
4. Budget (if mentioned)
5. Any specific requirements or preferences

Format your response as a clear description that can be used for further processing."""

# Budget Planning Prompt
BUDGET_PLANNING_PROMPT = """You are a budget planning expert. Create a detailed budget breakdown for an event.

Event Details:
{event_details}

Budget Constraint: {budget_constraint}

Create a budget breakdown with the following categories:
- Food & Catering
- Venue
- Decorations
- Entertainment
- Miscellaneous

Provide realistic estimates in Indian Rupees (₹)."""

# Food Planning Prompt
FOOD_PLANNING_PROMPT = """Plan a menu for an event based on the following details:

Event Type: {event_type}
Number of Guests: {guest_count}
Budget: {budget}
Preferences: {preferences}

Suggest:
1. Appetizers
2. Main Course
3. Desserts
4. Beverages

Consider Indian cuisine preferences and dietary restrictions."""

# Schedule Creator Prompt
SCHEDULE_CREATOR_PROMPT = """Create a detailed timeline/schedule for an event.

Event Type: {event_type}
Date: {date}
Duration: {duration}
Number of Guests: {guest_count}

Create a timeline with:
- Time slots
- Activities
- Transitions
- Key moments

Format as a structured timeline."""

# Guest List Organizer Prompt
GUEST_LIST_ORGANIZER_PROMPT = """Organize and validate a guest list for an event.

Event Type: {event_type}
Initial Guest List: {guest_list}
Venue Capacity: {capacity}

Tasks:
1. Validate guest count against capacity
2. Organize guests by category (family, friends, colleagues, etc.)
3. Suggest seating arrangements if applicable
4. Identify any missing important guests

Provide an organized guest list with categories."""

# Venue Suggestion Prompt
VENUE_SUGGESTION_PROMPT = """Suggest suitable venues for an event.

Event Type: {event_type}
Number of Guests: {guest_count}
Budget: {budget}
Location Preference: {location}
Date: {date}

Suggest 3-5 venues with:
- Name
- Capacity
- Estimated cost
- Location
- Features/amenities"""

# Decoration Plan Prompt
DECORATION_PLAN_PROMPT = """Create a decoration plan for an event.

Event Type: {event_type}
Theme: {theme}
Venue Type: {venue_type}
Budget: {budget}

Suggest:
1. Color scheme
2. Decoration items needed
3. Setup requirements
4. Estimated costs"""

# Shopping List Generator Prompt
SHOPPING_LIST_PROMPT = """Generate a comprehensive shopping list for an event.

Event Details: {event_details}
Menu: {menu}
Decoration Plan: {decoration_plan}
Guest Count: {guest_count}

Create a shopping list with:
- Items
- Quantities
- Estimated prices
- Priority (essential/optional)"""


