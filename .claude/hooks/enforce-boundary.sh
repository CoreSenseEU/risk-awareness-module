#!/usr/bin/env bash
# enforce-boundary.sh — PreToolUse hook for Claude Code
#
# Hard-enforces that Claude's file operations and bash commands stay inside
# PROJECT_ROOT. Resolves symlinks and `..` traversal with realpath.
#
# Exit codes:
#   0  = allow (fall through to permission rules)
#   2  = block (Claude sees the stderr message and adjusts)
#   1  = hook error (non-blocking; Claude sees warning and continues)
#
# Input: JSON on stdin from Claude Code. See:
#   https://code.claude.com/docs/en/hooks

set -uo pipefail

# ---- Config ----------------------------------------------------------------

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$HOME/coresense/risk-awareness-module}"

# Canonicalize. If this fails, the project dir doesn't exist — bail loud.
if ! PROJECT_ROOT_REAL="$(realpath -e "$PROJECT_ROOT" 2>/dev/null)"; then
  echo "enforce-boundary: PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  exit 1
fi

# ---- Helpers ---------------------------------------------------------------

block() {
  echo "BLOCKED by enforce-boundary hook: $1" >&2
  exit 2
}

# Check whether an absolute or relative path resolves to inside PROJECT_ROOT.
# Handles: symlinks, .., non-existent files (resolves parent), ~ expansion.
is_inside_project() {
  local p="$1"
  # Expand leading ~
  p="${p/#\~/$HOME}"
  # Make relative paths relative to PROJECT_ROOT (Claude's cwd in most cases)
  if [[ "$p" != /* ]]; then
    p="$PROJECT_ROOT_REAL/$p"
  fi
  local abs
  if [[ -e "$p" ]]; then
    abs="$(realpath "$p")"
  else
    # Resolve the deepest existing ancestor, then append the remainder.
    local parent="$p"
    local tail=""
    while [[ ! -e "$parent" && "$parent" != "/" && "$parent" != "." ]]; do
      tail="/$(basename "$parent")${tail}"
      parent="$(dirname "$parent")"
    done
    if [[ -e "$parent" ]]; then
      abs="$(realpath "$parent")${tail}"
    else
      abs="$p"
    fi
  fi
  [[ "$abs" == "$PROJECT_ROOT_REAL" || "$abs" == "$PROJECT_ROOT_REAL"/* ]]
}

# ---- Parse input -----------------------------------------------------------

INPUT="$(cat)"

if ! command -v jq >/dev/null 2>&1; then
  echo "enforce-boundary: jq is required but not installed" >&2
  exit 1
fi

TOOL_NAME="$(printf '%s' "$INPUT" | jq -r '.tool_name // ""')"

# ---- Route by tool ---------------------------------------------------------

case "$TOOL_NAME" in

  Edit|Write|Read|MultiEdit|NotebookEdit)
    FILE_PATH="$(printf '%s' "$INPUT" | jq -r '
      .tool_input.file_path // .tool_input.path // .tool_input.notebook_path // ""
    ')"
    [[ -z "$FILE_PATH" ]] && exit 0
    if ! is_inside_project "$FILE_PATH"; then
      block "$TOOL_NAME on '$FILE_PATH' resolves outside project root ($PROJECT_ROOT_REAL)"
    fi
    ;;

  Bash)
    CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // ""')"
    [[ -z "$CMD" ]] && exit 0

    # Belt-and-braces blocks that duplicate settings.json deny rules.
    # Hooks catch shell-ism bypasses (quoting tricks, variable expansion, etc).
    if grep -qE '(^|[;&|[:space:]`(])(sudo|doas)([[:space:]]|$)' <<<"$CMD"; then
      block "privilege escalation (sudo/doas) not permitted"
    fi
    if grep -qE '(^|[;&|[:space:]])rm[[:space:]]+(-[a-zA-Z]+[[:space:]]+)*(/|~|\$HOME|\$\{HOME\}|\.\.[/[:space:]])' <<<"$CMD"; then
      block "rm targeting absolute, home, or parent paths not permitted"
    fi
    if grep -qE '(curl|wget)[^|;&]*\|[[:space:]]*(sh|bash|zsh|fish)([[:space:]]|$)' <<<"$CMD"; then
      block "piping remote content directly to a shell not permitted"
    fi
    if grep -qE '\bgit[[:space:]]+(commit|push|reset[[:space:]]+--hard|clean[[:space:]]+-[fFdDxX]+|rebase|checkout[[:space:]]+--)' <<<"$CMD"; then
      block "commits/pushes/destructive git operations are reserved for the human"
    fi

    # Detect any cd that leaves the project. Multiple cds in one line are
    # evaluated independently; the final working dir is what matters.
    if grep -qE '(^|[;&|[:space:]])cd([[:space:]]|$)' <<<"$CMD"; then
      # Pull every `cd <target>` occurrence out and validate each.
      while IFS= read -r target; do
        [[ -z "$target" ]] && continue
        # Strip surrounding quotes
        target="${target%\"}"; target="${target#\"}"
        target="${target%\'}"; target="${target#\'}"
        # `cd` with no arg goes to $HOME — always outside project
        if [[ "$target" == "-" || "$target" == "~" || "$target" == "$HOME" ]]; then
          block "cd to \$HOME or previous dir escapes project root"
        fi
        if ! is_inside_project "$target"; then
          block "cd target '$target' escapes project root"
        fi
      done < <(grep -oE '(^|[;&|[:space:]])cd[[:space:]]+[^;&|[:space:]]+' <<<"$CMD" \
               | sed -E 's/^[;&|[:space:]]*cd[[:space:]]+//')
    fi

    # Detect absolute path arguments that point outside the project for
    # common file-writing/redirecting commands. This is heuristic — real
    # enforcement lives in the Edit/Write rules above.
    if grep -qE '>[[:space:]]*(/|~)' <<<"$CMD"; then
      # Extract redirect targets
      while IFS= read -r target; do
        target="${target#>}"
        target="${target## }"
        target="${target%% *}"
        [[ -z "$target" ]] && continue
        if ! is_inside_project "$target"; then
          block "redirect target '$target' writes outside project root"
        fi
      done < <(grep -oE '>[[:space:]]*[^[:space:];&|]+' <<<"$CMD")
    fi
    ;;

  *)
    # Unknown tool — let permission rules handle it.
    exit 0
    ;;
esac

exit 0