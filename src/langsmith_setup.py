"""
LangSmith Setup for Debugging.
Simple configuration to enable tracing and debugging.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_langsmith():
    """
    Setup LangSmith for debugging and tracing.
    This enables you to see the full workflow in LangSmith dashboard.
    """
    # Set environment variables if not already set
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        
        # Set project name
        project_name = os.getenv("LANGSMITH_PROJECT", "ai-event-planner")
        os.environ["LANGCHAIN_PROJECT"] = project_name
        
        print(f"✅ LangSmith tracing enabled for project: {project_name}")
        print("   View traces at: https://smith.langchain.com")
        return True
    else:
        print("ℹ️  LangSmith not configured (LANGSMITH_API_KEY not found)")
        print("   To enable debugging, add LANGSMITH_API_KEY to .env file")
        return False


# Auto-setup when imported
if __name__ != "__main__":
    setup_langsmith()


