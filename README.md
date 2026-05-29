# Antigravity 2.0 — OpenCode Dev Workflow Framework

Welcome to **Antigravity 2.0**, an advanced developer workflow framework powered by the **OpenCode Core Engine**. This framework bridges high-level agentic AI coding environments with low-level physical analysis layers (such as the **Cynthion USB Analyzer**) and local workspace automation.

---

## 🚀 Key Features

*   **⚡ Zen Mode Model Routing**: Optimized model routing strategies leveraging benchmarks from Gemini 1.5 Pro/Flash, Claude 3.5 Sonnet, and OpenAI o1 to balance performance, cost, and speed.
*   **🛠️ Standardized Workflow (AGENTS.md)**: Establishes a highly structured three-phase development cycle:
    1.  **Plan Mode**: Read-only architecture validation and static ast analysis.
    2.  **Build Mode**: Rapid implementation with real-time git diff and sandbox commands.
    3.  **Review Mode**: Specialized compliance, safety, memory, and UX checks.
*   **📡 Model Context Protocol (MCP)**: Implements type-safe `stdio` bindings for external device communication (Cynthion USB Analysis hardware) and safe filesystem sandboxing.
*   **🤖 Role-Specific Custom Agents**: Configured via `.opencode/agent/` for targeted development phases (`@review` and `@architect`).
*   **🎨 Premium Glassmorphic Web Dashboard**: An index entrypoint (`index.html`) featuring an animated live slate-indigo theme clock showing dynamic timezones, greetings, and time ticking indicators.

---

## 📁 Repository Structure

```
.
├── opencode.json             # Core global configuration and MCP server registry
├── AGENTS.md                 # Universal project development guidelines & standards
├── index.html                # Premium high-tech status & live clock dashboard
├── README.md                 # Repository introduction and guides
└── .opencode/
    ├── agent/
    │   ├── review.md         # Instructions and system prompts for the @review auditor
    │   └── architect.md      # Design guidelines for the @architect designer
    └── command/
        └── test-cynthion.js  # Node.js command script verifying stdio JSON-RPC handshake
```

---

## 🛠️ Getting Started

### 1. Requirements
- **Node.js** (v18+)
- **Git**
- **OpenCode CLI** / TUI Engine

### 2. Running the Live Dashboard
Simply open the `index.html` file in any modern browser to view the high-tech, responsive status dashboard with dynamic live timezone ticking:
```bash
start index.html
```

### 3. Testing the MCP Interface
You can run the mock custom command to verify the standard JSON-RPC communication channel:
```bash
node .opencode/command/test-cynthion.js
```

---

*Formulated with care to elevate agentic engineering standards. Powered by the OpenCode AI Standard.*
