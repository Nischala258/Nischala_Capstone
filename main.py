"""
Main Application - Simple entry point for the AI Event Planner.
Run this file to test the event planning system.
"""

import os
import sys
import contextlib
from dotenv import load_dotenv
import json


@contextlib.contextmanager
def _suppress_grpc_startup_logs_if_tty():
    """
    Suppress noisy gRPC/absl startup warnings *only* in an interactive terminal.
    This keeps the user's terminal clean without hiding errors in non-interactive runs.
    """
    if not sys.stderr.isatty():
        # In non-interactive environments (like this agent), don't hide anything.
        yield
        return
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        try:
            sys.stderr.close()
        finally:
            sys.stderr = original_stderr


def _init_app():
    """Load environment, set up LangSmith, and return the planner function."""
    # Load environment variables
    load_dotenv()

    # Import inside function so we can suppress startup logs cleanly
    from src.langsmith_setup import setup_langsmith
    from src.workflow import plan_event

    with _suppress_grpc_startup_logs_if_tty():
        # Setup LangSmith for debugging (optional)
        setup_langsmith()

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not found in .env file")
        print("Please create a .env file with your Google API key:")
        print("GOOGLE_API_KEY=your_key_here")
        print("\nContinuing anyway...\n")

    return plan_event


def main():
    """Main function to run the event planner."""
    plan_event = _init_app()
    print("=" * 60)
    print("🎉 AI Event Planner - Course Project")
    print("=" * 60)
    print("\nThis demonstrates:")
    print("  ✓ Prompting")
    print("  ✓ Structured Output")
    print("  ✓ Semantic Search (Vector Store)")
    print("  ✓ RAG (Retrieval-Augmented Generation)")
    print("  ✓ Tool Calling")
    print("  ✓ LangGraph Workflow")
    print("  ✓ LangSmith Debugging")
    print("\n" + "=" * 60 + "\n")
    
    # Example user inputs
    examples = [
        "Plan a birthday party for 30 people",
        "Organize a corporate dinner for 50 people on February 15th",
        "I need a baby shower plan for 25 people under ₹15,000",
        "Make a farewell party plan for 20 people"
    ]
    
    print("Example requests:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    
    print("\n" + "-" * 60)
    # In non-interactive environments (like this IDE-run), stdin isn't a TTY.
    # In that case, skip input() to avoid EOFError and use the default example.
    if sys.stdin.isatty():
        user_input = input(
            "\nEnter your event planning request (or press Enter for example): "
        ).strip()
    else:
        print("\n[Non-interactive environment detected, using default example request]\n")
        user_input = ""
    
    if not user_input:
        user_input = examples[0]  # Use first example
    
    print(f"\n📋 Processing: {user_input}")
    print("\n⏳ Planning your event...\n")
    
    try:
        # Run the planner
        result = plan_event(user_input)
        
        # Display results
        print("=" * 60)
        print("✅ Event Plan Generated!")
        print("=" * 60)
        
        # Show messages (workflow steps)
        if result.get("messages"):
            print("\n📝 Workflow Steps:")
            for msg in result["messages"]:
                print(f"  • {msg}")
        
        # Show final plan
        if result.get("final_plan"):
            plan = result["final_plan"]
            print("\n" + "=" * 60)
            print("🎯 FINAL EVENT PLAN")
            print("=" * 60)
            
            print(f"\nEvent Type: {plan.event_type}")
            print(f"Date: {plan.date or 'Not specified'}")
            print(f"Guest Count: {plan.guest_count}")
            print(f"Total Budget: ₹{plan.budget_total or 'Not calculated'}")
            
            if plan.schedule:
                print("\n📅 Schedule:")
                for item in plan.schedule:
                    print(f"  {item.time}: {item.activity}")
            
            if plan.budget_breakdown:
                print("\n💰 Budget Breakdown:")
                for item in plan.budget_breakdown:
                    print(f"  {item.category}: ₹{item.amount}")
            
            if plan.menu:
                print("\n🍽️  Menu:")
                for item in plan.menu:
                    print(f"  • {item.name} ({item.category})")
            
            # Show as JSON too
            print("\n" + "=" * 60)
            print("📄 Full Plan (JSON):")
            print("=" * 60)
            print(json.dumps(plan.model_dump(), indent=2, default=str))
        
        print("\n" + "=" * 60)
        print("✨ Planning Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Set GOOGLE_API_KEY in .env file")
        print("  2. Installed all requirements: pip install -r requirements.txt")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

