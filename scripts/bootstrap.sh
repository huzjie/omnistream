#!/usr/bin/env bash
#
# OmniStream bootstrap — get from a fresh clone to a running server.
#
# Usage:
#   ./scripts/bootstrap.sh              # core deps, mock mode, ready to run
#   ./scripts/bootstrap.sh --dev        # + linters, type checker, test tooling
#   ./scripts/bootstrap.sh --models     # + local ASR/TTS/vision model deps (heavy)
#   ./scripts/bootstrap.sh --all        # everything
#
# Safe to re-run: every step is idempotent.
#
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_DIR="${REPO_ROOT}/.venv"
readonly MIN_PY_MAJOR=3
readonly MIN_PY_MINOR=11

WANT_DEV=0
WANT_MODELS=0

# ------------------------------------------------------------------ output helpers
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'
else
  C_RESET=''; C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_BOLD=''
fi

step() { printf '%s==>%s %s%s%s\n' "${C_BLUE}" "${C_RESET}" "${C_BOLD}" "$*" "${C_RESET}"; }
ok()   { printf '%s  ok%s %s\n' "${C_GREEN}" "${C_RESET}" "$*"; }
warn() { printf '%swarn%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
die()  { printf '%s fail%s %s\n' "${C_RED}" "${C_RESET}" "$*" >&2; exit 1; }

trap 'die "bootstrap failed at line ${LINENO}"' ERR

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

# ------------------------------------------------------------------ arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)     WANT_DEV=1 ;;
    --models)  WANT_MODELS=1 ;;
    --all)     WANT_DEV=1; WANT_MODELS=1 ;;
    -h|--help) usage ;;
    *)         die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

cd "${REPO_ROOT}"

# ------------------------------------------------------------------ 1. python
step "Checking Python interpreter"
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    if "${candidate}" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"; then
      PYTHON_BIN="${candidate}"
      break
    fi
  fi
done
[[ -n "${PYTHON_BIN}" ]] || die "Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ is required but was not found on PATH"
ok "using $(${PYTHON_BIN} --version) at $(command -v "${PYTHON_BIN}")"

# ------------------------------------------------------------------ 2. virtualenv
step "Preparing virtual environment"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  ok "created ${VENV_DIR}"
else
  ok "reusing ${VENV_DIR}"
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  VENV_PY="${VENV_DIR}/bin/python"
else                       # Git Bash / MSYS on Windows
  # shellcheck disable=SC1091
  source "${VENV_DIR}/Scripts/activate"
  VENV_PY="${VENV_DIR}/Scripts/python"
fi

# ------------------------------------------------------------------ 3. dependencies
step "Upgrading pip toolchain"
"${VENV_PY}" -m pip install --upgrade --quiet pip setuptools wheel
ok "pip $(${VENV_PY} -m pip --version | awk '{print $2}')"

step "Installing core requirements"
"${VENV_PY}" -m pip install --quiet -r requirements.txt
ok "core dependencies installed"

if [[ ${WANT_DEV} -eq 1 ]]; then
  step "Installing development requirements"
  "${VENV_PY}" -m pip install --quiet -r requirements-dev.txt
  ok "dev dependencies installed"
fi

if [[ ${WANT_MODELS} -eq 1 ]]; then
  step "Installing local model requirements (this downloads a lot)"
  "${VENV_PY}" -m pip install --quiet -r requirements-local-models.txt
  ok "local model dependencies installed"
fi

step "Installing omnistream in editable mode"
"${VENV_PY}" -m pip install --quiet -e .
ok "omnistream is importable"

# ------------------------------------------------------------------ 4. runtime dirs & env
step "Preparing runtime directories"
mkdir -p data/objects data/recordings logs models
ok "data/, logs/, models/ ready"

step "Preparing .env"
if [[ ! -f .env ]]; then
  cp .env.example .env
  ok "created .env from .env.example (mock mode, no keys required)"
else
  ok ".env already exists — left untouched"
fi

# ------------------------------------------------------------------ 5. database
step "Applying database migrations"
if "${VENV_PY}" -m alembic upgrade head >/dev/null 2>&1; then
  ok "database schema is at head"
else
  warn "alembic upgrade failed — this is expected before the persistence layer is configured"
  warn "the server still runs in mock mode; re-run 'make migrate' once DATABASE_URL is set"
fi

# ------------------------------------------------------------------ 6. verification
step "Verifying installation"
"${VENV_PY}" - <<'PY'
import omnistream
from omnistream.config.settings import get_settings
from omnistream.providers import available_providers

settings = get_settings()
print(f"  version          : {omnistream.__version__}")
print(f"  env              : {settings.env}")
print(f"  mock_mode        : {settings.mock_mode}")
print(f"  default provider : {settings.providers.default}")
print(f"  providers        : {', '.join(available_providers())}")
PY
ok "import check passed"

# ------------------------------------------------------------------ done
cat <<EOF

${C_GREEN}${C_BOLD}Bootstrap complete.${C_RESET}

  Activate the environment:
      source ${VENV_DIR}/bin/activate

  Start the API (mock mode — no keys, no GPU):
      OMS_MOCK_MODE=true uvicorn omnistream.api.app:app --port 8080

  Then open:
      http://localhost:8080/docs      interactive API reference
      http://localhost:8080/healthz   liveness probe

  Other useful targets:
      make lint        ruff + mypy
      make docs        serve documentation locally
      make run         start API with reload

EOF
