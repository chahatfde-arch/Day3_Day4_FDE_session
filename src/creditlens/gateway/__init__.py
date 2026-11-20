"""One door to every LLM, plus the things that keep that door open.

    llm.py                the in-process gateway: cache, rate limit, fallbacks, offline stub
    litellm_config.yaml   the same routing as a shared server, for a room or for CI
    proxy.py              starts that server
    drill.py              the "cut the primary live" resilience drill
    costs.py              token cost estimates, with no hard-coded prices
"""
