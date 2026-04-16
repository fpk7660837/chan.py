#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$ROOT_DIR/infra/clickhouse"
ENV_FILE="$INFRA_DIR/.env"
COMPOSE_FILE="$INFRA_DIR/docker-compose.yml"
PROJECT_NAME="chan-clickhouse"

usage() {
  cat <<'USAGE'
Usage: scripts/clickhouse_market.sh <command>

Commands:
  init-env   Create infra/clickhouse/.env and host data directories
  start      Start ClickHouse with docker compose
  stop       Stop ClickHouse
  restart    Restart ClickHouse
  status     Show container status
  logs       Follow ClickHouse logs
  init       Apply the market schema
  client     Open clickhouse-client inside the container
  query SQL  Run a ClickHouse SQL query
  backup     Back up the market database to MARKET_DATA_ROOT/backup/clickhouse
USAGE
}

generate_password() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 32 | tr -d '\n'
  else
    date +%s | shasum -a 256 | awk '{print $1}'
  fi
}

ensure_env() {
  if [[ -f "$ENV_FILE" ]]; then
    return
  fi

  local password
  password="$(generate_password)"
  umask 077
  {
    echo "MARKET_DATA_ROOT=/Users/kevinfu/market-data"
    echo "CLICKHOUSE_DB=market"
    echo "CLICKHOUSE_USER=chan"
    echo "CLICKHOUSE_PASSWORD=$password"
  } > "$ENV_FILE"
  echo "Created $ENV_FILE"
}

load_env() {
  ensure_env
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

ensure_dirs() {
  load_env
  mkdir -p \
    "$MARKET_DATA_ROOT/clickhouse/data" \
    "$MARKET_DATA_ROOT/clickhouse/logs" \
    "$MARKET_DATA_ROOT/clickhouse/config" \
    "$MARKET_DATA_ROOT/parquet" \
    "$MARKET_DATA_ROOT/backup/clickhouse"
}

compose() {
  docker compose \
    --env-file "$ENV_FILE" \
    -f "$COMPOSE_FILE" \
    -p "$PROJECT_NAME" \
    "$@"
}

wait_ready() {
  load_env
  local attempt
  for attempt in $(seq 1 60); do
    if compose exec -T clickhouse clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT 1" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ClickHouse did not become ready within 60 seconds" >&2
  return 1
}

apply_schema() {
  load_env
  wait_ready
  compose exec -T clickhouse clickhouse-client \
    --user "$CLICKHOUSE_USER" \
    --password "$CLICKHOUSE_PASSWORD" \
    --multiquery < "$INFRA_DIR/init/001_create_market.sql"
}

run_query() {
  load_env
  local sql="${1:-}"
  if [[ -z "$sql" ]]; then
    echo "query requires SQL text" >&2
    exit 2
  fi
  compose exec -T clickhouse clickhouse-client \
    --user "$CLICKHOUSE_USER" \
    --password "$CLICKHOUSE_PASSWORD" \
    --query "$sql"
}

backup_market() {
  load_env
  wait_ready
  local backup_name="${1:-market-$(date +%Y%m%d-%H%M%S).zip}"
  compose exec -T clickhouse clickhouse-client \
    --user "$CLICKHOUSE_USER" \
    --password "$CLICKHOUSE_PASSWORD" \
    --query "BACKUP DATABASE market TO Disk('backups', '$backup_name')"
  echo "Backup written under $MARKET_DATA_ROOT/backup/clickhouse/$backup_name"
}

command="${1:-}"
shift || true

case "$command" in
  init-env)
    ensure_dirs
    ;;
  start)
    ensure_dirs
    compose up -d
    wait_ready
    ;;
  stop)
    load_env
    compose down
    ;;
  restart)
    ensure_dirs
    compose up -d
    compose restart clickhouse
    wait_ready
    ;;
  status)
    load_env
    compose ps
    ;;
  logs)
    load_env
    compose logs -f --tail=200 clickhouse
    ;;
  init)
    ensure_dirs
    compose up -d
    apply_schema
    ;;
  client)
    load_env
    compose exec clickhouse clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD"
    ;;
  query)
    run_query "${1:-}"
    ;;
  backup)
    backup_market "${1:-}"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
