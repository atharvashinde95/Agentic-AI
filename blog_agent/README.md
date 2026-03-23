# AI Blog Writing Agent

A fully agentic, LangGraph-powered blog writing system with image generation.

## Architecture

```
User Input
   │
   ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│   Planner   │───▶│    Writer    │───▶│ SEO Optimizer │
│   Agent     │    │   Agent      │    │    Agent      │
└─────────────┘    └──────────────┘    └───────┬───────┘
                                               │
                              ┌────────────────┴──────────────────┐
                              │ (if images requested)              │
                              ▼                                    │
                    ┌──────────────────┐                          │
                    │  Image Prompt    │                          │
                    │     Agent        │                          │
                    └────────┬─────────┘                          │
                             │                                    │
                             ▼                                    │
                    ┌──────────────────┐                          │
                    │  Image Generator │                          │
                    │     Agent        │                          │
                    └────────┬─────────┘                          │
                             └────────────────────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │    Assembler      │
                                    │     Agent         │
                                    └────────┬──────────┘
                                             │
                                             ▼
                              Markdown + HTML + Images
```

## Setup

```bash
# 1. Clone / copy the project
cd blog_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API credentials
cp .env.example .env
# Edit .env with your Capgemini LLM and Image Gen URLs + API keys

# 4. Run Streamlit UI
streamlit run app.py

# OR use CLI
python main.py --topic "Future of AI in Healthcare" --length medium --images 3
```

## Project Structure

```
blog_agent/
├── app.py                  # Streamlit UI (5 tabs)
├── main.py                 # CLI entry point
├── graph.py                # LangGraph pipeline definition
├── state.py                # Shared Pydantic state schema
├── config.py               # Centralised configuration
├── agents/
│   ├── planner.py          # Outline generation
│   ├── writer.py           # Full content writing
│   ├── seo_optimizer.py    # SEO enhancement
│   ├── image_prompt.py     # Image prompt crafting
│   ├── image_generator.py  # Nova-Canvas API calls
│   └── assembler.py        # Final output assembly
├── utils/
│   ├── llm_client.py       # Capgemini LLM wrapper
│   └── image_client.py     # Capgemini image API wrapper
├── outputs/                # Generated blogs + images
├── .env.example
├── requirements.txt
└── README.md
```

## Models Used
- **LLM**: `amazon.nova-lite-v1:0` via Capgemini GenEngine
- **Image Gen**: `amazon.nova-canvas-v1:0` via Capgemini GenEngine

## Streamlit UI Tabs
| Tab | Description |
|-----|-------------|
| 🚀 Generate | Input form + live agent pipeline status + log viewer |
| 📄 Blog Post | Rendered Markdown output with download |
| 🌐 HTML Preview | Live HTML iframe with download |
| 🖼️ Images | Image gallery with per-image download |
| ⚙️ Settings | API URL/key configuration saved to .env |

## Logs
All agent logs stream to the **terminal** in real time with timestamps.
The Streamlit UI also shows a live log panel on the Generate tab.
