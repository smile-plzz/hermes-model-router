#!/usr/bin/env python3
"""
Hermes Model Router — Prototype.

Classifies incoming tasks by context and picks the best free-tier model
across Groq, Gemini, Mistral, Nous, and OpenRouter.

Usage:
  python routing-router.py "write a python function"
  python routing-router.py "analyze microservices" --json
  python routing-router.py "hello" --list-providers
"""
