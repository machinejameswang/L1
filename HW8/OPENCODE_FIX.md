# OpenCode Fix Notes

The error shown in Antigravity IDE:

```text
Unauthorized: Authentication Fails (governor)
```

means OpenCode is reaching the DeepSeek provider, but no valid API key is being sent.

## What Was Fixed

- `../opencode.json` default model was changed to:

```text
deepseek/deepseek-chat
```

- The custom DeepSeek model list was simplified to:

```text
deepseek-chat
deepseek-reasoner
```

- The filesystem MCP path encoding issue was corrected to:

```text
d:\AI人工智慧\Antigravity
```

## What You Still Need To Do

Set your DeepSeek API key in the Windows User environment:

```powershell
.\setup_opencode_deepseek.ps1 -ApiKey "YOUR_DEEPSEEK_API_KEY"
```

Then restart Antigravity IDE or open a new terminal.

Verify:

```powershell
opencode debug config
```

In the resolved config, this field should no longer be empty:

```json
"apiKey": "..."
```

Do not commit or paste your API key into `opencode.json`.

## Current Status

The API key has been configured in the Windows User environment.

Validation result:

```text
OpenCode can resolve the API key.
DeepSeek authentication no longer fails with Unauthorized.
DeepSeek now returns: Insufficient Balance.
```

This means the remaining blocker is DeepSeek account balance or quota. Add balance
to the DeepSeek account for this key, or replace it with another key that has
available credit.
