"""Serving adapters that expose non-OpenAI inference backends through an
OpenAI-compatible HTTP surface, so the rest of aligne can eval them
unchanged via `aligne.client.ChatClient`.

Submodules import heavy, optional dependencies (e.g. tinker, fastapi, uvicorn)
LAZILY inside their entry points, so `import aligne` and the core `aligne`
CLI keep working with only the lean core dependencies installed.
"""
