# Tearing down: how to get your laptop and your cloud account back

The habit this teaches is the one that matters most after a workshop: **the thing you
started is the thing you have to stop.** A forgotten ClickHouse container eats disk,
and a forgotten EC2 instance eats money.

## Local Docker

Stop the containers but keep the data:

```bash
docker compose -f infra/docker-compose.yml --profile core --profile observability \
  --profile ml --profile retrieval --profile security --profile legacy down
```

Stop them **and delete the volumes** (traces, MLflow runs, the Oracle database, the
Qdrant collections and the SQL Server data all go):

```bash
docker compose -f infra/docker-compose.yml --profile core --profile observability \
  --profile ml --profile retrieval --profile security --profile legacy down -v
```

`-v` is irreversible. The CreditLens data itself is safe either way, because
`data/` and `creditlens.db` are bind mounts from the repository, not Docker volumes.

### One profile at a time

```bash
docker compose -f infra/docker-compose.yml --profile observability down -v
```

### What is still running?

```bash
docker ps --filter "name=creditlens"
docker volume ls --filter "name=creditlens"
docker system df                 # how much disk Docker is actually using
```

### Reclaiming disk

```bash
docker image rm creditlens-api:local
docker system prune -a --volumes    # careful: this touches images from other projects too
```

Rough sizes to expect: SQL Server and Oracle are the heavy ones (gigabytes each), the
six Langfuse containers together are the next largest, and Qdrant, MLflow and Keycloak
are small. Do not run the `legacy` and `observability` profiles at the same time on a
laptop with 8 GB of RAM.

## Things that are not Docker

| Started by | Stopped by |
| --- | --- |
| `python -m creditlens.api` | Ctrl+C in that terminal |
| `python -m creditlens.integrations.soap_server` | Ctrl+C |
| `python -m creditlens.gateway.proxy` | Ctrl+C |
| A stray process on port 8000 | `lsof -ti:8000 \| xargs kill` (macOS, Linux) or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` (Windows) |

## Cloud

AWS is not covered here because it has its own, longer checklist and a script that
does it for you: see [`infra/aws/README.md`](aws/README.md) and run
[`infra/aws/teardown.sh`](aws/teardown.sh), which deletes everything tagged
`project=creditlens`.

Free SaaS accounts used by the project are deleted in their own dashboards, and none
of them bill by the hour: Langfuse Cloud, Neo4j AuraDB Free, Qdrant Cloud, Pinecone,
Jira Cloud, Slack. Revoking the API keys you pasted into `.env` is still worth doing
when a session ends, because those keys outlive the workshop.

## The one check worth doing every time

```bash
docker ps                                    # nothing of yours running
git status                                   # no secret accidentally staged
grep -r "API_KEY=sk-" .env 2>/dev/null       # and .env is still in .gitignore
```
