# Autonomous Claude Code on Hetzner — Setup

This bundle locks a Claude Code session to `~/coresense/risk-awareness-module`,
runs autonomously (no permission prompts), reserves commits/pushes for you,
and bridges to **claude.ai/code** via Remote Control.

## Contents

```
.claude/
├── settings.json                  # permission rules + hook registration
└── hooks/
    └── enforce-boundary.sh        # hard directory-boundary enforcement
```

## Install

On the Hetzner box, as the user that will run Claude:

```bash
# 1. Make sure Claude Code is current (Remote Control needs ≥ 2.1.51)
npm install -g @anthropic-ai/claude-code
claude --version

# 2. Install the hook dependency
sudo apt-get install -y jq

# 3. Drop the .claude/ directory into the project root
mkdir -p ~/coresense/risk-awareness-module/.claude/hooks
cp settings.json  ~/coresense/risk-awareness-module/.claude/
cp enforce-boundary.sh ~/coresense/risk-awareness-module/.claude/hooks/
chmod +x ~/coresense/risk-awareness-module/.claude/hooks/enforce-boundary.sh

# 4. First-time auth (opens a browser URL you paste locally)
cd ~/coresense/risk-awareness-module
claude
# run /login, follow the URL, then /exit
```

## Daily operation

### Option A — Remote Control + web UI (recommended)

```bash
cd ~/coresense/risk-awareness-module
claude remote-control
```

Keep the terminal open (or launch inside `tmux` / `screen`). A session URL
prints; open it on claude.ai/code from any device. Press `space` in the
terminal for a QR code.

**Caveat:** As of early 2026 Remote Control doesn't fully honour
`--dangerously-skip-permissions`. That's why `settings.json` sets
`"defaultMode": "bypassPermissions"` instead — it sticks. If Anthropic
changes the Remote Control/skip-permissions interaction later, this
config still works.

### Option B — Tmux for survival across SSH drops

```bash
tmux new -s claude
cd ~/coresense/risk-awareness-module && claude remote-control
```

#### Tmux cheatsheet

Prefix is `Ctrl-b` (press and release, then the next key).

| Action                          | Keys / command                    |
| ------------------------------- | --------------------------------- |
| New session named `claude`      | `tmux new -s claude`              |
| List sessions                   | `tmux ls`                         |
| Attach to `claude`              | `tmux attach -t claude` (or `-a`) |
| Detach (session keeps running)  | `Ctrl-b d`                        |
| Kill session                    | `tmux kill-session -t claude`     |
| Scroll / search output          | `Ctrl-b [` → arrows or `/pattern`, `q` to exit |
| Rename current session          | `Ctrl-b $`                        |
| New window (tab)                | `Ctrl-b c`                        |
| Next / previous window          | `Ctrl-b n` / `Ctrl-b p`           |
| Split pane vertically           | `Ctrl-b %`                        |
| Split pane horizontally         | `Ctrl-b "`                        |
| Switch between panes            | `Ctrl-b` then arrow key           |
| Close current pane              | `Ctrl-b x`                        |

Typical workflow: SSH in, `tmux attach -t claude` (or `new -s claude` if none),
work, `Ctrl-b d` to detach, close SSH. Claude keeps running. If SSH drops
mid-session, reconnect and re-attach — no state is lost.

### Option C — Fully headless (no web UI, just run a task)

```bash
cd ~/coresense/risk-awareness-module
claude -p --dangerously-skip-permissions "implement the spec in TODO.md"
```

## Auto-updates

Claude Code auto-updates on launch when installed via npm. For belt-and-braces,
add a weekly cron:

```cron
0 4 * * 1  npm update -g @anthropic-ai/claude-code >/dev/null 2>&1
```

## How the safety layers stack

1. **`permissions.defaultMode: "bypassPermissions"`** — no prompts for
   anything that isn't explicitly denied.
2. **`permissions.deny`** — blocks whole categories regardless of allow
   rules (deny always wins). This is where `git commit/push`, `sudo`,
   `rm -rf /`, credential files, etc. live.
3. **`hooks/enforce-boundary.sh`** — OS-level boundary check that runs
   *before* permission rules evaluate. Catches symlink escapes, `..`
   traversal, `cd /etc`, redirects to outside paths, `curl | sh`,
   chained `sudo`, and doas. Exit code 2 hard-blocks the call; Claude
   sees the reason and adapts.
4. **Git** — you do all commits. `git reset --hard`, `git clean -fd`,
   `git checkout --`, and `git rebase` are denied, so Claude can't wipe
   your uncommitted work.

Defense in depth: if any one layer has a bug, the others catch it.

## Verify the hook before you trust it

From inside the project dir:

```bash
# Should print "BLOCKED..." and exit 2
echo '{"tool_name":"Edit","tool_input":{"file_path":"/etc/passwd"}}' \
  | .claude/hooks/enforce-boundary.sh; echo "exit=$?"

# Should exit 0 silently
echo '{"tool_name":"Edit","tool_input":{"file_path":"src/main.py"}}' \
  | CLAUDE_PROJECT_DIR=$(pwd) .claude/hooks/enforce-boundary.sh; echo "exit=$?"
```

## Optional: systemd user service for boot-start

Only sensible for headless runs (Option C) — Remote Control needs a foreground
process that surfaces the session URL.

`~/.config/systemd/user/claude-loop.service`:

```ini
[Unit]
Description=Claude Code autonomous task runner
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/coresense/risk-awareness-module
ExecStart=/usr/bin/env claude -p --dangerously-skip-permissions "%h/coresense/risk-awareness-module/TASK.md"
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now claude-loop
loginctl enable-linger $USER   # keep running across logout
journalctl --user -u claude-loop -f
```

## Tweaks you may want

- **Add more allow rules** for toolchains you actually use (the list
  assumes Python/Node/Rust — trim or extend in `settings.json`).
- **Narrow `Read`** further by removing `/usr/share/**` if Claude doesn't
  need to read system docs.
- **Enable Auto Mode** (Anthropic's classifier-gated alternative to bypass)
  by changing `defaultMode` to `"acceptEdits"` and adding
  `"autoApprove": true` — slightly safer, slightly slower.
- **Sandbox mode** (OS-level filesystem/network isolation for Bash) can be
  enabled in `settings.json` under a top-level `"sandbox"` key; worth
  checking `code.claude.com/docs/en/permissions` for the current schema.