# The CreditLens AI API (milestone 3)

Priya, the credit analyst, does not run Python. She uses a screen, and the screen talks
to this service. This page explains the five ideas the service is built on, in the
order you meet them.

```bash
python -m creditlens.api            # http://localhost:8000, docs at /docs
```

Open <http://localhost:8000/docs>. FastAPI generates that page from the type hints in
the route functions. Nobody writes it by hand, and it cannot go out of date.

---

## 1. Dependency injection

Look at a route handler:

```python
@router.get("/applications")
@observe(name="api.list_applications")
def list_applications(user: User = Depends(current_user), engine=Depends(get_engine)):
    ...
```

The handler says *what it needs* ("a current user, a database engine") and FastAPI
calls the small functions that build them. The handler never reads a header, never
opens a connection and never imports a global.

Three things that buys you:

- **Testing.** `tests/test_api.py` sets `DATABASE_URL` to a temporary SQLite file and
  calls `create_app()`. Every route now talks to that database. No route changed.
- **One place per concern.** There is exactly one function that turns headers into a
  `User` (`auth.current_user`). Fix a bug there, and every route is fixed.
- **Honest signatures.** You can read a handler and know everything it touches.

The rule that makes it work: **nothing is read at import time.** `create_app()` reads
`DATABASE_URL` when it is called, and parks the engine on `app.state`. If the engine
were a module-level global built on import, the test above would be impossible, which
is why `app.py` deliberately has no `app = create_app()` line at the bottom.

```python
# the factory, not a module-level app
uvicorn creditlens.api.app:create_app --factory
```

## 2. CORS

A browser refuses to let a page on `https://credit.goodbank.example` call an API on
`http://localhost:8000` unless the API sends a header saying it is allowed. That rule
is CORS, and it protects the *user's browser*, not the server. Anything that is not a
browser (curl, Python, the tests) ignores CORS entirely, so CORS is never a security
control for your API.

```python
app.add_middleware(CORSMiddleware, allow_origins=cors_origins(), ...)
```

`cors_origins()` reads `CORS_ORIGINS` from the environment and defaults to `*`, which
is right for a workshop on localhost and wrong for anything public. Set it to the real
origins before deploying:

```bash
CORS_ORIGINS=https://credit.goodbank.example,https://credit-staging.goodbank.example
```

Note `allow_credentials=False` in `app.py`. A browser refuses `*` together with
credentials, so if you ever need cookies, you must list the origins explicitly.

## 3. Bearer tokens

```bash
# 1. Get a token. There is no password: this is a mock login for the workshop.
curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U05"}'
# {"access_token":"eyJhbGciOiJIUzI1NiIs...","token_type":"bearer","user":{...}}

# 2. Send it with every other request.
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" -d '{"user_id":"U05"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/applications
```

A JWT is three base64 pieces joined by dots: a header, a payload of claims, and a
signature. Ours carries `sub` (the user id), `iat` and `exp`. Paste one into
<https://jwt.io> and read it: the payload is **not encrypted**, only signed. Never put
anything secret in a token. The signature, made with `JWT_SECRET`, is what proves the
token came from us and was not edited. Change one character and the server returns 401,
which `test_a_tampered_token_is_401` proves.

Why a token rather than sending `user_id` on every call? Because `user_id` can be typed
by anyone. A signed token cannot be forged without the secret.

### The dev shortcut

```bash
curl -H "X-User-Id: U05" http://localhost:8000/applications
```

This exists so you can demo RBAC without juggling tokens. It is a back door and it is
labelled as one: set `API_DEV_AUTH=0` and it disappears. Do that before anything leaves
your laptop.

In production, `auth.py` is the file you replace. The `security` profile in
`infra/docker-compose.yml` starts Keycloak with the same three roles and six users
(`infra/keycloak/creditlens-realm.json`), so you can swap the mock login for real OIDC
without touching a single route.

## 4. RBAC: role-based access control

Three roles, from `data/raw/registry/users.csv`:

| Role | Reads | Writes |
| --- | --- | --- |
| `analyst` | their own region only | drafts memos in their own region |
| `head_of_credit` | every region | the only role that may approve |
| `auditor` | every region | nothing: any POST except `/auth/token` is 403 |

```bash
# U01 is the North analyst. A South file does not exist as far as she is concerned.
curl -H "X-User-Id: U01" http://localhost:8000/applications | grep -c South     # 0
curl -i -H "X-User-Id: U01" http://localhost:8000/applications/A002             # 403
```

Two different behaviours, on purpose:

- A **list** is *filtered*. A short list is not an error, and filtering leaks nothing.
- A **single record** is *refused* with 403, and the message does not say which region
  the file belongs to, because that would already leak something.

The filter itself lives in `creditlens.guardrails.rbac.region_filter(user)`, which
returns a SQL fragment like `region = 'North'`. The API uses it, and so does the
agent's `run_sql` tool. One definition means the two can never drift apart, which is
the kind of bug that only shows up in an audit.

Approvals are stricter still. `ApprovalRequest` has no `decided_by` field, so a caller
cannot claim to be somebody else; the server fills it in from the token. Every
approval appends an `AuditEvent`:

```bash
curl -X POST http://localhost:8000/applications/A001/approve \
  -H "X-User-Id: U05" -H "Content-Type: application/json" \
  -d '{"decision": "approve", "comment": "DSCR exception accepted"}'

curl -H "X-User-Id: U06" http://localhost:8000/applications/A001/audit
```

## 5. Seeing the trace in Langfuse

Every handler is wrapped in `@observe` from `creditlens.observability`, so one HTTP
request becomes one Langfuse trace, with every LLM call the gateway makes nested
inside it. You see the prompt, the answer, the model, the token counts and the latency,
which is how you debug "why did the memo say that".

Without keys, `@observe` is a no-op. The code is identical either way, so there is
never a "tracing version" of a function.

### Option A: Langfuse Cloud (free project, nothing to install)

1. Sign up at <https://cloud.langfuse.com> and create a project.
2. **Settings → API keys → Create new API keys.**
3. Put them in `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com     # or https://eu.cloud.langfuse.com
```

### Option B: self-hosted

```bash
docker compose -f infra/docker-compose.yml --profile observability up -d
```

Six containers come up (web, worker, Postgres, ClickHouse, Redis, MinIO: see the
comments in the compose file for why). Open <http://localhost:3000>, create an account
and a project, copy the keys, and set `LANGFUSE_HOST=http://localhost:3000`.

### Then

```bash
python -m creditlens.api
curl -H "X-User-Id: U05" http://localhost:8000/applications
```

Refresh the **Traces** tab. You should see `api.list_applications`, tagged with
`role:head_of_credit` and `region:ALL` and carrying `user_id=U05`, because the handler
calls `update_trace(...)`. Click it to see the nested spans.

`settings.langfuse_enabled` is just `bool(public_key and secret_key)`, and `/health`
reports `"tracing": "langfuse"` or `"off"`, so you can check without guessing:

```bash
curl -s http://localhost:8000/health
```

If traces do not appear: the keys are from a different project, `LANGFUSE_HOST` has a
trailing slash, or the process exited before flushing. A short script should call
`creditlens.observability.flush()` before it ends; the long-running API does not need
to.

---

## Endpoint reference

| Method | Path | Who | What |
| --- | --- | --- | --- |
| GET | `/health` | anyone | Liveness, database, tracing and offline status |
| POST | `/auth/token` | anyone | Dev login by `user_id`, returns a bearer token |
| GET | `/me` | any user | Who the server thinks you are, and what you may do |
| GET | `/applications` | any user | Applications in your region, newest first |
| GET | `/applications/{id}` | any user | One application plus its statement features |
| POST | `/applications/{id}/memo` | analyst, head_of_credit | Runs the agent graph (501 until milestone 6) |
| POST | `/applications/{id}/approve` | head_of_credit | Records the decision, resumes the agent run |
| GET | `/applications/{id}/audit` | any user | The audit trail for one application |
| POST | `/risk/score` | any user | Mounted from `creditlens.ml.serving` when milestone 8 is installed |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///creditlens.db` | Read by `create_app()`, not at import |
| `JWT_SECRET` | `change-me-for-the-workshop` | Signs and verifies tokens |
| `API_DEV_AUTH` | `1` | `0` turns off the `X-User-Id` shortcut |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed browser origins |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | empty | Tracing; empty means no-op |
| `LLM_OFFLINE` | `0` | `1` answers from the cache or a stub, with no network |
