# SiliconCedars_agentic-ai-onboarding-Project

# Multi-Agent Supervisor

A hierarchical AI assistant built with LangGraph, where a main supervisor routes requests to specialized agents (SQL, web research, visualization, conversation, and RAG).

## Structure

- `agents/` — Main supervisor and specialized agents (SQL, visualization, conversation, RAG), plus the `research/` subgraph team (sub-supervisor, researcher, report writer).
- `graph/` — LangGraph orchestration: `workflow.py` builds the graph; `routing.py` centralizes routing functions and conditional decisions.
- `state/` — Shared LangGraph state definition used across the graph.
- `tools/` — Interfaces/wrappers for external capabilities: database access, web search, and visualization.
- `rag/` — Isolated RAG pipeline: indexing and retrieval over an internal document collection.
- `services/` — Infrastructure/service integrations that don't belong inside agents or tools (reserved).
- `evaluation/` — LangSmith-based evaluation of the system.
- `config/` — Project settings and environment configuration.
- `tests/` — Unit tests for routing, state, and graph.
