# Capstone Project Report

## Title: AI Event Planner - Intelligent Event Planning System using LLMs

## Overview

This project is an AI-powered event planning assistant that helps users create complete event plans just by describing what they need in plain English. For example, if you say "Plan a birthday party for 30 people," the system automatically generates everything you need: a guest list, schedule, budget breakdown, menu suggestions, venue recommendations, and even a shopping list. It works by understanding your request, finding similar past events, and using that knowledge to create a personalized plan tailored to your needs.

## Reason for picking up this project

This project perfectly demonstrates all the key concepts we learned in this course. Event planning requires understanding natural language (Prompting), organizing information into structured formats (Structured Output), finding relevant examples from past events (Semantic Search), using those examples to improve suggestions (RAG), performing calculations like budget estimates (Tool Calling), managing a multi-step planning process (LangGraph), and debugging the entire workflow (LangSmith). 

I chose this project because event planning is something everyone can relate to, we've all struggled with organizing parties, meetings, or celebrations. By applying LLM concepts to this real-world problem, I can show how these technologies can make complex planning tasks much easier and more accessible to everyone.

## Plan

I plan to execute these steps to complete my project:

1. **Start with the scope**  DONE
   - Jot down the event types we want to support, what outputs each plan should include, and collect a few sample layouts so we aren’t starting from scratch.

2. **Teach the model to understand the user**  DONE
   - I’ll write two small prompts: one to classify the intent (“birthday party”, “corporate dinner”, etc.) and another to cleanly extract the date, guest count, budget, and any must-haves from the user text.

3. **Lock in the data shapes**  DONE
   - Define Pydantic models (guest, schedule item, budget line, etc.) and create a LangGraph `State` TypedDict so every node reads/writes structured information instead of raw strings.

4. **Give it some memory**  DONE
   - Build a tiny dataset of sample events, embed them, and store them in Chroma so we can fetch “similar past plans” whenever a new request comes in.

5. **Ground the planning with RAG**  DONE
   - Add a retrieval node that pulls the top templates, and a RAG node that blends those templates into the planning prompt so the model reuses good ideas instead of hallucinating.

6. **Add practical helpers**  
   - Implement a budget calculator, guest-capacity checker, and menu cost estimator, plus a simple schedule builder so every plan returns realistic numbers and a timeline.

7. **Stitch it together with LangGraph**  
   - Wire the nodes in order (intent → extraction → retrieval → RAG → tools → schedule → formatter) and make sure each step updates the shared state for the next one.

8. **Make it runnable for anyone**  
   - Build a friendly CLI (`main.py`) that loads `.env`, optionally enables LangSmith tracing, asks for a request, and prints both a human summary and the JSON output (with at least one demo input baked in).
